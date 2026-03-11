bl_info = {
    "name": "Material Toolkit",
    "author": "kharand",
    "version": (1, 0, 0),
    "blender": (3, 0, 0),
    "location": "View3D > Sidebar > Material Toolkit",
    "description": "Complete toolkit for textures, materials and batch operations",
    "category": "Material",
}

import bpy
import os
import shutil
import re
import bmesh
from bpy.props import StringProperty, EnumProperty, IntProperty, BoolProperty
from bpy.types import Operator, Panel
from bpy_extras.io_utils import ExportHelper
from mathutils import Matrix
from collections import defaultdict


def sanitize(name: str) -> str:
    name = re.sub(r"[\\/:*?\"<>|]+", "_", name)
    name = name.strip().rstrip(".")
    return name or "image"


def choose_ext(img, fallback=".png"):
    p = (img.filepath_raw or img.filepath or "").strip()
    root, ext = os.path.splitext(p)
    return ext if ext else fallback


def abspath(img):
    try:
        return bpy.path.abspath(img.filepath_raw or img.filepath, library=img.library)
    except Exception:
        return bpy.path.abspath(img.filepath_raw or img.filepath)


class EXPORT_OT_textures(Operator, ExportHelper):
    bl_idname = "export.textures"
    bl_label = "Export Textures"
    bl_description = "Export all textures to selected directory"
    bl_options = {'REGISTER'}
    
    filename_ext = ""
    use_filter_folder = True
    
    directory: StringProperty(
        name="Directory Path",
        description="Directory to export textures to",
        subtype='DIR_PATH'
    )
    
    def execute(self, context):
        target_dir = self.directory
        
        if not target_dir:
            self.report({'ERROR'}, "No directory selected")
            return {'CANCELLED'}
        
        os.makedirs(target_dir, exist_ok=True)
        
        saved, skipped = 0, 0
        
        for img in bpy.data.images:
            if getattr(img, "source", None) in {"VIEWER"}:
                print(f"Skip VIEWER image: {img.name}")
                skipped += 1
                continue
            
            base = sanitize(img.name)
            ext = choose_ext(img)
            out_path = os.path.join(target_dir, base + ext)
            
            packed_ok = False
            try:
                if hasattr(img, "packed_files") and len(img.packed_files) > 0:
                    img.filepath_raw = out_path
                    try:
                        img.reload()
                    except Exception:
                        pass
                    img.save()
                    print(f"Saved PACKED image: {out_path}")
                    saved += 1
                    packed_ok = True
                elif getattr(img, "packed_file", None):
                    img.filepath_raw = out_path
                    try:
                        img.reload()
                    except Exception:
                        pass
                    img.save()
                    print(f"Saved PACKED image: {out_path}")
                    saved += 1
                    packed_ok = True
            except Exception as e:
                print(f"Failed saving packed '{img.name}': {e}")
            
            if packed_ok:
                continue
            
            src = abspath(img)
            if src and os.path.isfile(src):
                src_root, src_ext = os.path.splitext(os.path.basename(src))
                dst_name = sanitize(src_root) + (src_ext if src_ext else ext)
                dst_path = os.path.join(target_dir, dst_name)
                if os.path.exists(dst_path) and sanitize(src_root) != base:
                    dst_path = out_path
                try:
                    shutil.copy2(src, dst_path)
                    print(f"Copied FILE image: {dst_path}")
                    saved += 1
                except Exception as e:
                    print(f"Failed copying '{img.name}' from '{src}': {e}")
                    skipped += 1
                continue
            
            try:
                img.filepath_raw = os.path.join(target_dir, base + ".png")
                img.file_format = 'PNG'
                try:
                    img.reload()
                except Exception:
                    pass
                img.save()
                print(f"Saved GENERATED image: {img.filepath_raw}")
                saved += 1
            except Exception as e:
                print(f"Skip '{img.name}' (no data / cannot save): {e}")
                skipped += 1
        
        self.report({'INFO'}, f"Export complete. Saved: {saved}, Skipped: {skipped}")
        print(f"Done. Saved: {saved}, Skipped: {skipped}")
        
        return {'FINISHED'}
    
    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}


class MATERIAL_OT_convert_to_principled(Operator):
    bl_idname = "material.convert_to_principled"
    bl_label = "Convert All to Principled BSDF"
    bl_description = "Convert all materials to Principled BSDF"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        import time
        
        print("\n" + "="*60)
        print("STARTING MATERIAL CONVERSION")
        print("="*60 + "\n")
        
        print("PHASE 1: Scanning and classifying textures from all materials...")
        material_cache = {}
        total_materials = len([m for m in bpy.data.materials if m.use_nodes])
        cached_count = 0
        
        for mat in bpy.data.materials:
            if not mat.use_nodes:
                continue
            
            nodes = mat.node_tree.nodes
            links = mat.node_tree.links
            
            texture_data = self.scan_all_textures(nodes, links, mat.name)
            
            if texture_data['textures']:
                material_cache[mat.name] = texture_data
                cached_count += 1
                roles = [r for r in texture_data['textures'].keys()]
                family = texture_data.get('family_root', 'unknown')
                print(f"[{cached_count}/{total_materials}] Cached: '{mat.name}' -> family: '{family}' roles: {roles}")
                time.sleep(0.01)
        
        print(f"\nCaching complete: {cached_count} materials with textures cached")
        print(f"Waiting 0.5 seconds before conversion...\n")
        time.sleep(0.5)
        
        print("PHASE 2: Converting materials to Principled BSDF...")
        converted_count = 0
        skipped_count = 0
        error_count = 0
        
        for mat in bpy.data.materials:
            if not mat.use_nodes:
                continue
            
            try:
                nodes = mat.node_tree.nodes
                links = mat.node_tree.links
                
                output_node = None
                for node in nodes:
                    if node.type == 'OUTPUT_MATERIAL':
                        output_node = node
                        break
                
                if output_node and output_node.inputs['Surface'].is_linked:
                    linked_node = output_node.inputs['Surface'].links[0].from_node
                    if linked_node.type == 'BSDF_PRINCIPLED':
                        skipped_count += 1
                        continue
                
                if mat.name not in material_cache:
                    print(f"No texture cached for '{mat.name}', skipping")
                    skipped_count += 1
                    continue
                
                texture_data = material_cache[mat.name]
                uv_info = texture_data['uv_info']
                textures = texture_data['textures']
                
                nodes.clear()
                time.sleep(0.005)
                
                principled = nodes.new(type='ShaderNodeBsdfPrincipled')
                principled.location = (200, 0)
                
                output = nodes.new(type='ShaderNodeOutputMaterial')
                output.location = (500, 0)
                links.new(principled.outputs['BSDF'], output.inputs['Surface'])
                
                uv_chain_output = None
                if uv_info['vector_connected']:
                    if uv_info['has_uvmap'] and not uv_info['has_mapping']:
                        uvmap = nodes.new(type='ShaderNodeUVMap')
                        uvmap.location = (-800, 0)
                        uvmap.uv_map = uv_info['uvmap_name']
                        uv_chain_output = uvmap.outputs['UV']
                    elif uv_info['has_mapping']:
                        mapping = nodes.new(type='ShaderNodeMapping')
                        mapping.location = (-600, 0)
                        mapping.inputs['Location'].default_value = uv_info['mapping_location']
                        mapping.inputs['Rotation'].default_value = uv_info['mapping_rotation']
                        mapping.inputs['Scale'].default_value = uv_info['mapping_scale']
                        
                        if uv_info['has_uvmap']:
                            uvmap = nodes.new(type='ShaderNodeUVMap')
                            uvmap.location = (-900, 0)
                            uvmap.uv_map = uv_info['uvmap_name']
                            links.new(uvmap.outputs['UV'], mapping.inputs['Vector'])
                        elif uv_info['has_texcoord']:
                            tex_coord = nodes.new(type='ShaderNodeTexCoord')
                            tex_coord.location = (-900, 0)
                            links.new(tex_coord.outputs[uv_info['coord_output']], mapping.inputs['Vector'])
                        
                        uv_chain_output = mapping.outputs['Vector']
                    elif uv_info['has_texcoord']:
                        tex_coord = nodes.new(type='ShaderNodeTexCoord')
                        tex_coord.location = (-800, 0)
                        uv_chain_output = tex_coord.outputs[uv_info['coord_output']]
                
                y_offset = 300
                has_alpha = False
                diffuse_node = None
                ao_node = None
                
                if 'diffuse' in textures:
                    tex_info = textures['diffuse']
                    tex_node = nodes.new(type='ShaderNodeTexImage')
                    tex_node.image = tex_info['image']
                    tex_node.location = (-300, y_offset)
                    tex_node.interpolation = tex_info['interpolation']
                    tex_node.projection = tex_info['projection']
                    tex_node.extension = tex_info['extension']
                    if tex_node.image and tex_node.image.colorspace_settings:
                        tex_node.image.colorspace_settings.name = 'sRGB'
                    if uv_chain_output:
                        links.new(uv_chain_output, tex_node.inputs['Vector'])
                    diffuse_node = tex_node
                    y_offset -= 300
                
                if 'ao' in textures:
                    tex_info = textures['ao']
                    tex_node = nodes.new(type='ShaderNodeTexImage')
                    tex_node.image = tex_info['image']
                    tex_node.location = (-300, y_offset)
                    tex_node.interpolation = tex_info['interpolation']
                    tex_node.projection = tex_info['projection']
                    tex_node.extension = tex_info['extension']
                    if tex_node.image and tex_node.image.colorspace_settings:
                        tex_node.image.colorspace_settings.name = 'Non-Color'
                    if uv_chain_output:
                        links.new(uv_chain_output, tex_node.inputs['Vector'])
                    ao_node = tex_node
                    y_offset -= 300
                
                if diffuse_node and ao_node:
                    mix_node = nodes.new(type='ShaderNodeMix')
                    mix_node.data_type = 'RGBA'
                    mix_node.blend_type = 'MULTIPLY'
                    mix_node.location = (0, 200)
                    mix_node.inputs['Factor'].default_value = 1.0
                    links.new(diffuse_node.outputs['Color'], mix_node.inputs[6])
                    links.new(ao_node.outputs['Color'], mix_node.inputs[7])
                    links.new(mix_node.outputs[2], principled.inputs['Base Color'])
                elif diffuse_node:
                    links.new(diffuse_node.outputs['Color'], principled.inputs['Base Color'])
                
                if 'normal' in textures:
                    tex_info = textures['normal']
                    tex_node = nodes.new(type='ShaderNodeTexImage')
                    tex_node.image = tex_info['image']
                    tex_node.location = (-300, y_offset)
                    tex_node.interpolation = tex_info['interpolation']
                    tex_node.projection = tex_info['projection']
                    tex_node.extension = tex_info['extension']
                    if tex_node.image and tex_node.image.colorspace_settings:
                        tex_node.image.colorspace_settings.name = 'Non-Color'
                    if uv_chain_output:
                        links.new(uv_chain_output, tex_node.inputs['Vector'])
                    
                    normal_map = nodes.new(type='ShaderNodeNormalMap')
                    normal_map.location = (0, y_offset)
                    links.new(tex_node.outputs['Color'], normal_map.inputs['Color'])
                    links.new(normal_map.outputs['Normal'], principled.inputs['Normal'])
                    y_offset -= 300
                
                if 'packed_rm' in textures:
                    tex_info = textures['packed_rm']
                    tex_node = nodes.new(type='ShaderNodeTexImage')
                    tex_node.image = tex_info['image']
                    tex_node.location = (-300, y_offset)
                    tex_node.interpolation = tex_info['interpolation']
                    tex_node.projection = tex_info['projection']
                    tex_node.extension = tex_info['extension']
                    if tex_node.image and tex_node.image.colorspace_settings:
                        tex_node.image.colorspace_settings.name = 'Non-Color'
                    if uv_chain_output:
                        links.new(uv_chain_output, tex_node.inputs['Vector'])
                    
                    sep_rgb = nodes.new(type='ShaderNodeSeparateColor')
                    sep_rgb.location = (0, y_offset)
                    links.new(tex_node.outputs['Color'], sep_rgb.inputs['Color'])
                    
                    if 'roughness' not in textures:
                        links.new(sep_rgb.outputs['Red'], principled.inputs['Roughness'])
                    if 'metallic' not in textures:
                        links.new(sep_rgb.outputs['Green'], principled.inputs['Metallic'])
                    
                    y_offset -= 300
                
                if 'roughness' in textures:
                    tex_info = textures['roughness']
                    tex_node = nodes.new(type='ShaderNodeTexImage')
                    tex_node.image = tex_info['image']
                    tex_node.location = (-300, y_offset)
                    tex_node.interpolation = tex_info['interpolation']
                    tex_node.projection = tex_info['projection']
                    tex_node.extension = tex_info['extension']
                    if tex_node.image and tex_node.image.colorspace_settings:
                        tex_node.image.colorspace_settings.name = 'Non-Color'
                    if uv_chain_output:
                        links.new(uv_chain_output, tex_node.inputs['Vector'])
                    links.new(tex_node.outputs['Color'], principled.inputs['Roughness'])
                    y_offset -= 300
                
                if 'metallic' in textures:
                    tex_info = textures['metallic']
                    tex_node = nodes.new(type='ShaderNodeTexImage')
                    tex_node.image = tex_info['image']
                    tex_node.location = (-300, y_offset)
                    tex_node.interpolation = tex_info['interpolation']
                    tex_node.projection = tex_info['projection']
                    tex_node.extension = tex_info['extension']
                    if tex_node.image and tex_node.image.colorspace_settings:
                        tex_node.image.colorspace_settings.name = 'Non-Color'
                    if uv_chain_output:
                        links.new(uv_chain_output, tex_node.inputs['Vector'])
                    links.new(tex_node.outputs['Color'], principled.inputs['Metallic'])
                    y_offset -= 300
                
                if 'specular' in textures:
                    tex_info = textures['specular']
                    tex_node = nodes.new(type='ShaderNodeTexImage')
                    tex_node.image = tex_info['image']
                    tex_node.location = (-300, y_offset)
                    tex_node.interpolation = tex_info['interpolation']
                    tex_node.projection = tex_info['projection']
                    tex_node.extension = tex_info['extension']
                    if tex_node.image and tex_node.image.colorspace_settings:
                        tex_node.image.colorspace_settings.name = 'Non-Color'
                    if uv_chain_output:
                        links.new(uv_chain_output, tex_node.inputs['Vector'])
                    if 'Specular IOR Level' in principled.inputs:
                        links.new(tex_node.outputs['Color'], principled.inputs['Specular IOR Level'])
                    elif 'Specular' in principled.inputs:
                        links.new(tex_node.outputs['Color'], principled.inputs['Specular'])
                    y_offset -= 300
                
                if 'emission' in textures:
                    tex_info = textures['emission']
                    tex_node = nodes.new(type='ShaderNodeTexImage')
                    tex_node.image = tex_info['image']
                    tex_node.location = (-300, y_offset)
                    tex_node.interpolation = tex_info['interpolation']
                    tex_node.projection = tex_info['projection']
                    tex_node.extension = tex_info['extension']
                    if tex_node.image and tex_node.image.colorspace_settings:
                        tex_node.image.colorspace_settings.name = 'sRGB'
                    if uv_chain_output:
                        links.new(uv_chain_output, tex_node.inputs['Vector'])
                    if 'Emission Color' in principled.inputs:
                        links.new(tex_node.outputs['Color'], principled.inputs['Emission Color'])
                    elif 'Emission' in principled.inputs:
                        links.new(tex_node.outputs['Color'], principled.inputs['Emission'])
                    if 'Emission Strength' in principled.inputs:
                        principled.inputs['Emission Strength'].default_value = 1.0
                    y_offset -= 300
                
                if 'alpha' in textures:
                    tex_info = textures['alpha']
                    tex_node = nodes.new(type='ShaderNodeTexImage')
                    tex_node.image = tex_info['image']
                    tex_node.location = (-300, y_offset)
                    tex_node.interpolation = tex_info['interpolation']
                    tex_node.projection = tex_info['projection']
                    tex_node.extension = tex_info['extension']
                    if tex_node.image and tex_node.image.colorspace_settings:
                        tex_node.image.colorspace_settings.name = 'Non-Color'
                    if uv_chain_output:
                        links.new(uv_chain_output, tex_node.inputs['Vector'])
                    links.new(tex_node.outputs['Color'], principled.inputs['Alpha'])
                    has_alpha = True
                    y_offset -= 300
                elif diffuse_node and textures.get('diffuse', {}).get('has_alpha', False):
                    links.new(diffuse_node.outputs['Alpha'], principled.inputs['Alpha'])
                    has_alpha = True
                
                if has_alpha:
                    mat.blend_method = 'CLIP'
                    if hasattr(mat, 'shadow_method'):
                        mat.shadow_method = 'CLIP'
                else:
                    mat.blend_method = 'OPAQUE'
                    if hasattr(mat, 'shadow_method'):
                        mat.shadow_method = 'OPAQUE'
                mat.use_backface_culling = False
                
                converted_count += 1
                roles = list(textures.keys())
                print(f"[{converted_count}] '{mat.name}' -> {roles}")
                time.sleep(0.01)
                
            except Exception as e:
                error_count += 1
                print(f"[ERROR] Failed to convert '{mat.name}': {str(e)}")
                import traceback
                traceback.print_exc()
                continue
        
        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()
        
        print("\n" + "="*60)
        print("CONVERSION COMPLETE")
        print("="*60)
        print(f"Converted: {converted_count}")
        print(f"Skipped: {skipped_count}")
        if error_count > 0:
            print(f"Errors: {error_count}")
        print("="*60 + "\n")
        
        if error_count > 0:
            self.report({'WARNING'}, f"Converted {converted_count}, skipped {skipped_count}, {error_count} errors")
        else:
            self.report({'INFO'}, f"Successfully converted {converted_count} materials, skipped {skipped_count}")
        
        return {'FINISHED'}
    
    def extract_family_root(self, texture_name):
        name = texture_name.lower()
        name = re.sub(r'\.[^.]+$', '', name)
        
        pbr_suffixes = [
            r'[_-]diffuse$', r'[_-]albedo$', r'[_-]basecolor$', r'[_-]base_color$',
            r'[_-]normal$', r'[_-]norm$', r'[_-]nrm$',
            r'[_-]roughness$', r'[_-]rough$', r'[_-]rgh$',
            r'[_-]metallic$', r'[_-]metalness$', r'[_-]metal$', r'[_-]mtl$',
            r'[_-]specular$', r'[_-]spec$',
            r'[_-]emission$', r'[_-]emissive$', r'[_-]emit$', r'[_-]glow$',
            r'[_-]occlusion$', r'[_-]occ$',
            r'[_-]opacity$', r'[_-]alpha$', r'[_-]trans$', r'[_-]transparency$',
            r'[_-]orm$', r'[_-]rma$', r'[_-]arm$', r'[_-]mra$', r'[_-]rm$',
            r'[_-]ao$', r'[_-]d$', r'[_-]n$', r'[_-]m$', r'[_-]r$', r'[_-]s$',
            r'[_-]e$', r'[_-]a$', r'[_-]col$', r'[_-]nor$', r'[_-]diff$',
        ]
        
        for suffix in pbr_suffixes:
            name = re.sub(suffix, '', name)
        
        name = re.sub(r'[_-][ab]$', '', name, flags=re.IGNORECASE)
        
        return name
    
    def texture_belongs_to_family(self, texture_name, family_root):
        if not family_root:
            return True
        
        tex_lower = texture_name.lower()
        tex_lower = re.sub(r'\.[^.]+$', '', tex_lower)
        
        if tex_lower.startswith(family_root):
            return True
        
        tex_root = self.extract_family_root(texture_name)
        if tex_root == family_root:
            return True
        
        min_match = int(len(family_root) * 0.6)
        if min_match > 5 and family_root[:min_match] == tex_root[:min_match]:
            return True
        
        return False
    
    def find_diffuse_texture(self, tex_nodes, nodes, links):
        for tex_node in tex_nodes:
            role = self.get_role_from_connections(tex_node, links)
            if role == 'diffuse':
                return tex_node
        
        for tex_node in tex_nodes:
            if tex_node.image:
                name_lower = tex_node.image.name.lower()
                if re.search(r'[_-](d|diff|diffuse|albedo|basecolor|base_color|col)\.', name_lower):
                    return tex_node
                if re.search(r'[_-](d|diff|diffuse|albedo|basecolor|base_color|col)$', re.sub(r'\.[^.]+$', '', name_lower)):
                    return tex_node
        
        for tex_node in tex_nodes:
            if tex_node.image:
                name_lower = tex_node.image.name.lower()
                if any(kw in name_lower for kw in ['diffuse', 'albedo', 'basecolor', 'base_color']):
                    return tex_node
        
        non_diffuse_indicators = ['_n.', '_n_', '_m.', '_m_', '_r.', '_r_', '_rm', '_orm', 
                                   '_ao', '_occ', 'normal', 'rough', 'metal', 'spec']
        for tex_node in tex_nodes:
            if tex_node.image:
                name_lower = tex_node.image.name.lower()
                is_connected = any(out.is_linked for out in tex_node.outputs)
                if is_connected and not any(ind in name_lower for ind in non_diffuse_indicators):
                    return tex_node
        
        for tex_node in tex_nodes:
            if tex_node.image:
                name_lower = tex_node.image.name.lower()
                if not any(ind in name_lower for ind in non_diffuse_indicators):
                    return tex_node
        
        return tex_nodes[0] if tex_nodes else None
    
    def scan_all_textures(self, nodes, links, material_name):
        result = {
            'textures': {},
            'uv_info': {
                'vector_connected': False,
                'has_mapping': False,
                'has_texcoord': False,
                'has_uvmap': False,
                'uvmap_name': '',
                'coord_output': 'UV',
                'mapping_location': (0.0, 0.0, 0.0),
                'mapping_rotation': (0.0, 0.0, 0.0),
                'mapping_scale': (1.0, 1.0, 1.0)
            },
            'family_root': None
        }
        
        tex_nodes = [n for n in nodes if n.type == 'TEX_IMAGE' and n.image]
        if not tex_nodes:
            return result
        
        diffuse_tex_node = self.find_diffuse_texture(tex_nodes, nodes, links)
        
        if not diffuse_tex_node or not diffuse_tex_node.image:
            return result
        
        family_root = self.extract_family_root(diffuse_tex_node.image.name)
        if not family_root or len(family_root) < 3:
            family_root = self.extract_family_root(material_name)
        
        result['family_root'] = family_root
        
        family_tex_nodes = []
        for tex_node in tex_nodes:
            if tex_node.image:
                if self.texture_belongs_to_family(tex_node.image.name, family_root):
                    family_tex_nodes.append(tex_node)
                else:
                    print(f"    [FILTERED] '{tex_node.image.name}' excluded - doesn't match family '{family_root}'")
        
        if not family_tex_nodes:
            family_tex_nodes = [diffuse_tex_node]
        
        variant_key = self.detect_variant_key(family_tex_nodes)
        if variant_key:
            family_tex_nodes = self.filter_textures_by_variant(family_tex_nodes, variant_key)
        
        uv_extracted = False
        classified = {}
        
        if diffuse_tex_node in family_tex_nodes:
            tex_info = {
                'image': diffuse_tex_node.image,
                'interpolation': diffuse_tex_node.interpolation,
                'projection': diffuse_tex_node.projection,
                'extension': diffuse_tex_node.extension,
                'has_alpha': self.image_has_alpha(diffuse_tex_node.image)
            }
            classified['diffuse'] = tex_info
            
            uv_info = self.extract_texture_info(diffuse_tex_node)
            result['uv_info'] = {
                'vector_connected': uv_info['vector_connected'],
                'has_mapping': uv_info['has_mapping'],
                'has_texcoord': uv_info['has_texcoord'],
                'has_uvmap': uv_info['has_uvmap'],
                'uvmap_name': uv_info['uvmap_name'],
                'coord_output': uv_info['coord_output'],
                'mapping_location': uv_info['mapping_location'],
                'mapping_rotation': uv_info['mapping_rotation'],
                'mapping_scale': uv_info['mapping_scale']
            }
            uv_extracted = True
        
        for tex_node in family_tex_nodes:
            if tex_node == diffuse_tex_node:
                continue
            
            role = self.classify_texture_role(tex_node, nodes, links)
            
            if role and role not in classified:
                tex_info = {
                    'image': tex_node.image,
                    'interpolation': tex_node.interpolation,
                    'projection': tex_node.projection,
                    'extension': tex_node.extension,
                    'has_alpha': self.image_has_alpha(tex_node.image)
                }
                classified[role] = tex_info
                
                if not uv_extracted:
                    uv_info = self.extract_texture_info(tex_node)
                    result['uv_info'] = {
                        'vector_connected': uv_info['vector_connected'],
                        'has_mapping': uv_info['has_mapping'],
                        'has_texcoord': uv_info['has_texcoord'],
                        'has_uvmap': uv_info['has_uvmap'],
                        'uvmap_name': uv_info['uvmap_name'],
                        'coord_output': uv_info['coord_output'],
                        'mapping_location': uv_info['mapping_location'],
                        'mapping_rotation': uv_info['mapping_rotation'],
                        'mapping_scale': uv_info['mapping_scale']
                    }
                    uv_extracted = True
        
        result['textures'] = classified
        return result
    
    def detect_variant_key(self, tex_nodes):
        variant_pattern = re.compile(r'[_-]([AB])[_-]', re.IGNORECASE)
        variant_counts = defaultdict(int)
        
        for tex_node in tex_nodes:
            if tex_node.image:
                name = tex_node.image.name
                match = variant_pattern.search(name)
                if match:
                    key = match.group(0).upper()
                    variant_counts[key] += 1
        
        if variant_counts:
            return max(variant_counts, key=variant_counts.get)
        return None
    
    def filter_textures_by_variant(self, tex_nodes, variant_key):
        filtered = []
        variant_key_lower = variant_key.lower()
        
        for tex_node in tex_nodes:
            if tex_node.image:
                name_lower = tex_node.image.name.lower()
                if variant_key_lower in name_lower:
                    filtered.append(tex_node)
                elif not re.search(r'[_-][ab][_-]', name_lower, re.IGNORECASE):
                    filtered.append(tex_node)
        
        return filtered if filtered else tex_nodes
    
    def classify_texture_role(self, tex_node, nodes, links):
        name_lower = (tex_node.image.name if tex_node.image else '').lower()
        node_name_lower = tex_node.name.lower()
        combined = name_lower + ' ' + node_name_lower
        
        connection_role = self.get_role_from_connections(tex_node, links)
        if connection_role:
            if connection_role == 'normal':
                if self.is_blacklisted_for_normal(combined):
                    return None
            return connection_role
        
        role = self.classify_by_suffix(name_lower)
        if role:
            if role == 'normal' and self.is_blacklisted_for_normal(combined):
                return None
            return role
        
        role = self.classify_by_keywords(combined)
        if role == 'normal' and self.is_blacklisted_for_normal(combined):
            return None
        return role
    
    def is_blacklisted_for_normal(self, combined):
        blacklist = [
            '_m.', '_m_', '_metal', '_metallic', '_mtl',
            '_rm.', '_rm_', '_orm', '_rma', '_arm', '_mra',
            'multi', '_multi',
            'mask', '_mask',
            '_ao', 'ambientocclusion', 'occlusion', '_occ',
            '_r.', '_r_', '_rough', '_rgh', 'roughness',
            '_s.', '_s_', '_spec', 'specular',
            '_d.', '_d_', '_diff', 'diffuse', 'albedo', 'basecolor',
            '_a.', '_a_', 'alpha', 'opacity', 'trans'
        ]
        return any(bl in combined for bl in blacklist)
    
    def classify_by_suffix(self, name_lower):
        suffix_patterns = [
            (r'[_-]n\.', 'normal'),
            (r'[_-]n$', 'normal'),
            (r'[_-]nor\.', 'normal'),
            (r'[_-]nor$', 'normal'),
            (r'[_-]norm\.', 'normal'),
            (r'[_-]norm$', 'normal'),
            (r'[_-]normal\.', 'normal'),
            (r'[_-]normal$', 'normal'),
            (r'[_-]nrm\.', 'normal'),
            (r'[_-]nrm$', 'normal'),
            
            (r'[_-]d\.', 'diffuse'),
            (r'[_-]d$', 'diffuse'),
            (r'[_-]diff\.', 'diffuse'),
            (r'[_-]diff$', 'diffuse'),
            (r'[_-]diffuse\.', 'diffuse'),
            (r'[_-]diffuse$', 'diffuse'),
            (r'[_-]albedo\.', 'diffuse'),
            (r'[_-]albedo$', 'diffuse'),
            (r'[_-]basecolor\.', 'diffuse'),
            (r'[_-]basecolor$', 'diffuse'),
            (r'[_-]base_color\.', 'diffuse'),
            (r'[_-]base_color$', 'diffuse'),
            (r'[_-]col\.', 'diffuse'),
            (r'[_-]col$', 'diffuse'),
            
            (r'[_-]rm\.', 'packed_rm'),
            (r'[_-]rm$', 'packed_rm'),
            (r'[_-]orm\.', 'packed_rm'),
            (r'[_-]orm$', 'packed_rm'),
            (r'[_-]rma\.', 'packed_rm'),
            (r'[_-]rma$', 'packed_rm'),
            (r'[_-]arm\.', 'packed_rm'),
            (r'[_-]arm$', 'packed_rm'),
            (r'[_-]mra\.', 'packed_rm'),
            (r'[_-]mra$', 'packed_rm'),
            
            (r'[_-]r\.', 'roughness'),
            (r'[_-]r$', 'roughness'),
            (r'[_-]rough\.', 'roughness'),
            (r'[_-]rough$', 'roughness'),
            (r'[_-]rgh\.', 'roughness'),
            (r'[_-]rgh$', 'roughness'),
            (r'[_-]roughness\.', 'roughness'),
            (r'[_-]roughness$', 'roughness'),
            
            (r'[_-]m\.', 'metallic'),
            (r'[_-]m$', 'metallic'),
            (r'[_-]metal\.', 'metallic'),
            (r'[_-]metal$', 'metallic'),
            (r'[_-]mtl\.', 'metallic'),
            (r'[_-]mtl$', 'metallic'),
            (r'[_-]metallic\.', 'metallic'),
            (r'[_-]metallic$', 'metallic'),
            (r'[_-]metalness\.', 'metallic'),
            (r'[_-]metalness$', 'metallic'),
            
            (r'[_-]s\.', 'specular'),
            (r'[_-]s$', 'specular'),
            (r'[_-]spec\.', 'specular'),
            (r'[_-]spec$', 'specular'),
            (r'[_-]specular\.', 'specular'),
            (r'[_-]specular$', 'specular'),
            
            (r'[_-]e\.', 'emission'),
            (r'[_-]e$', 'emission'),
            (r'[_-]emit\.', 'emission'),
            (r'[_-]emit$', 'emission'),
            (r'[_-]emiss\.', 'emission'),
            (r'[_-]emiss$', 'emission'),
            (r'[_-]emission\.', 'emission'),
            (r'[_-]emission$', 'emission'),
            (r'[_-]glow\.', 'emission'),
            (r'[_-]glow$', 'emission'),
            
            (r'[_-]a\.', 'alpha'),
            (r'[_-]a$', 'alpha'),
            (r'[_-]alpha\.', 'alpha'),
            (r'[_-]alpha$', 'alpha'),
            (r'[_-]opacity\.', 'alpha'),
            (r'[_-]opacity$', 'alpha'),
            (r'[_-]trans\.', 'alpha'),
            (r'[_-]trans$', 'alpha'),
            (r'[_-]mask\.', 'alpha'),
            (r'[_-]mask$', 'alpha'),
            
            (r'[_-]ao\.', 'ao'),
            (r'[_-]ao$', 'ao'),
            (r'[_-]occ\.', 'ao'),
            (r'[_-]occ$', 'ao'),
            (r'[_-]occlusion\.', 'ao'),
            (r'[_-]occlusion$', 'ao'),
            (r'ambientocclusion', 'ao'),
            (r'ambient_occlusion', 'ao'),
        ]
        
        base_name = re.sub(r'\.[^.]+$', '', name_lower)
        
        for pattern, role in suffix_patterns:
            if re.search(pattern, base_name) or re.search(pattern, name_lower):
                return role
        
        return None
    
    def classify_by_keywords(self, combined):
        if self.has_strong_normal_indicator(combined):
            if not self.is_blacklisted_for_normal(combined):
                return 'normal'
        
        packed_kw = ['_rm', '_orm', '_rma', '_arm', '_mra', 'roughnessmetallic']
        if any(kw in combined for kw in packed_kw):
            return 'packed_rm'
        
        rough_kw = ['rough', 'rgh', 'roughness']
        if any(kw in combined for kw in rough_kw):
            if not any(pk in combined for pk in ['_rm', '_orm', '_rma', '_arm', '_mra']):
                return 'roughness'
        
        metal_kw = ['metal', 'mtl', 'metallic', 'metalness']
        if any(kw in combined for kw in metal_kw):
            if not any(pk in combined for pk in ['_rm', '_orm', '_rma', '_arm', '_mra']):
                return 'metallic'
        
        spec_kw = ['spec', 'specular']
        if any(kw in combined for kw in spec_kw):
            return 'specular'
        
        emit_kw = ['emiss', 'emit', 'glow', 'emission']
        if any(kw in combined for kw in emit_kw):
            return 'emission'
        
        alpha_kw = ['alpha', 'opacity', 'trans', 'transparency']
        if any(kw in combined for kw in alpha_kw):
            return 'alpha'
        
        ao_kw = ['ambientocclusion', 'ambient_occlusion', 'occlusion']
        if any(kw in combined for kw in ao_kw):
            return 'ao'
        if '_ao' in combined or combined.startswith('ao') or ' ao' in combined:
            return 'ao'
        
        diff_kw = ['diffuse', 'diff', 'albedo', 'basecolor', 'base_color']
        if any(kw in combined for kw in diff_kw):
            return 'diffuse'
        
        return None
    
    def has_strong_normal_indicator(self, combined):
        strong_patterns = [
            r'[_-]n[_.\s]',
            r'[_-]n$',
            r'[_-]nor[_.\s]',
            r'[_-]nor$',
            r'[_-]norm[_.\s]',
            r'[_-]norm$',
            r'[_-]normal[_.\s]',
            r'[_-]normal$',
            r'[_-]nrm[_.\s]',
            r'[_-]nrm$',
            r'\bnormal\b',
            r'\bnrm\b',
            r'\bnorm\b',
            r'normalmap',
            r'normal_map',
        ]
        return any(re.search(p, combined) for p in strong_patterns)
    
    def get_role_from_connections(self, tex_node, links):
        for output in tex_node.outputs:
            for link in output.links:
                to_node = link.to_node
                to_socket = link.to_socket
                
                if to_node.type == 'BSDF_PRINCIPLED':
                    socket_name = to_socket.name.lower()
                    if 'base color' in socket_name or socket_name == 'color':
                        return 'diffuse'
                    elif 'normal' in socket_name:
                        return 'normal'
                    elif 'roughness' in socket_name:
                        return 'roughness'
                    elif 'metallic' in socket_name:
                        return 'metallic'
                    elif 'specular' in socket_name:
                        return 'specular'
                    elif 'emission' in socket_name:
                        return 'emission'
                    elif 'alpha' in socket_name:
                        return 'alpha'
                
                elif to_node.type == 'NORMAL_MAP':
                    return 'normal'
                
                elif to_node.type == 'BUMP':
                    return 'normal'
                
                elif to_node.type in ['MIX', 'MIX_RGB', 'MIXRGB', 'MIX_SHADER']:
                    continue
                
                elif to_node.type == 'SEPARATE_RGB' or to_node.type == 'SEPARATE_COLOR':
                    return 'packed_rm'
        
        return None
    
    def image_has_alpha(self, image):
        if not image:
            return False
        if image.channels >= 4:
            if image.depth in [32, 128]:
                return True
            name_lower = image.name.lower()
            if any(kw in name_lower for kw in ['alpha', 'rgba', 'transparent']):
                return True
        return False
    
    def find_diffuse_texture_node(self, nodes, material_name):
        shader = None
        for node in nodes:
            if node.type.startswith('BSDF_') or node.type in ['EMISSION', 'EEVEE_SPECULAR', 'GROUP']:
                shader = node
                break
        
        if shader:
            for input_socket in shader.inputs:
                if input_socket.name in ['Base Color', 'Color', 'Diffuse Color', 'texture_diffuse']:
                    if input_socket.is_linked:
                        result = self.trace_to_texture_node(input_socket)
                        if result:
                            return result
        
        diffuse_keywords = ['diffuse', 'diff', 'color', 'albedo', 'base', '_d']
        texture_nodes = [n for n in nodes if n.type == 'TEX_IMAGE' and n.image]
        
        for keyword in diffuse_keywords:
            for node in texture_nodes:
                if keyword in node.image.name.lower() or keyword in node.name.lower():
                    return self.extract_texture_info(node)
        
        for node in texture_nodes:
            for output in node.outputs:
                if output.is_linked:
                    return self.extract_texture_info(node)
        
        if texture_nodes:
            return self.extract_texture_info(texture_nodes[0])
        
        return None
    
    def trace_to_texture_node(self, socket):
        if not socket or not socket.is_linked:
            return None
        
        visited = set()
        
        def trace(current_socket):
            if not current_socket or not current_socket.is_linked:
                return None
            
            node = current_socket.links[0].from_node
            if id(node) in visited:
                return None
            visited.add(id(node))
            
            if node.type == 'TEX_IMAGE' and node.image:
                return self.extract_texture_info(node)
            
            if node.type in ['MIX', 'MIX_RGB', 'MIX_COLOR', 'MIXRGB']:
                for inp_name in ['A', 'B', 'Color1', 'Color2']:
                    if inp_name in node.inputs and node.inputs[inp_name].is_linked:
                        result = trace(node.inputs[inp_name])
                        if result:
                            return result
            
            for inp in node.inputs:
                if inp.is_linked:
                    result = trace(inp)
                    if result:
                        return result
            
            return None
        
        return trace(socket)
    
    def extract_texture_info(self, tex_node):
        info = {
            'node': tex_node,
            'vector_connected': False,
            'has_mapping': False,
            'has_texcoord': False,
            'has_uvmap': False,
            'uvmap_name': '',
            'coord_output': 'UV',
            'mapping_location': (0.0, 0.0, 0.0),
            'mapping_rotation': (0.0, 0.0, 0.0),
            'mapping_scale': (1.0, 1.0, 1.0)
        }
        
        vector_input = tex_node.inputs.get('Vector')
        if vector_input and vector_input.is_linked:
            info['vector_connected'] = True
            connected = vector_input.links[0].from_node
            
            if connected.type == 'UVMAP':
                info['has_uvmap'] = True
                info['uvmap_name'] = connected.uv_map
            
            elif connected.type == 'MAPPING':
                info['has_mapping'] = True
                info['mapping_location'] = tuple(connected.inputs['Location'].default_value)
                info['mapping_rotation'] = tuple(connected.inputs['Rotation'].default_value)
                info['mapping_scale'] = tuple(connected.inputs['Scale'].default_value)
                
                map_vec = connected.inputs.get('Vector')
                if map_vec and map_vec.is_linked:
                    coord_node = map_vec.links[0].from_node
                    
                    if coord_node.type == 'UVMAP':
                        info['has_uvmap'] = True
                        info['uvmap_name'] = coord_node.uv_map
                    
                    elif coord_node.type == 'TEX_COORD':
                        info['has_texcoord'] = True
                        for output in coord_node.outputs:
                            if output.is_linked and output.links[0].to_node == connected:
                                info['coord_output'] = output.name
                                break
            
            elif connected.type == 'TEX_COORD':
                info['has_texcoord'] = True
                for output in connected.outputs:
                    if output.is_linked and output.links[0].to_node == tex_node:
                        info['coord_output'] = output.name
                        break
        
        return info


def ensure_collection(name: str):
    col = bpy.data.collections.get(name)
    if not col:
        col = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(col)
    return col

def iter_mesh_objects(source_mode: str, source_collection_name: str):
    if source_mode == 'SELECTED':
        return [o for o in bpy.context.selected_objects if o.type == 'MESH']
    elif source_mode == 'COLLECTION' and source_collection_name:
        col = bpy.data.collections.get(source_collection_name)
        if not col:
            return []
        def walk(c):
            for o in c.objects:
                if o.type == 'MESH':
                    yield o
            for ch in c.children:
                yield from walk(ch)
        return list(walk(col))
    else:
        return [o for o in bpy.context.scene.objects if o.type == 'MESH']

def evaluated_mesh_copy(obj: bpy.types.Object, apply_modifiers: bool):
    mw = obj.matrix_world.copy()
    if apply_modifiers:
        deps = bpy.context.evaluated_depsgraph_get()
        eval_obj = obj.evaluated_get(deps)
        real_mesh = bpy.data.meshes.new_from_object(
            eval_obj,
            preserve_all_data_layers=True,
            depsgraph=deps,
        )
        return real_mesh, mw
    else:
        return obj.data.copy(), mw

def chunk_faces_by_vertex_limit(mesh: bpy.types.Mesh, max_vertices: int):
    mesh.calc_loop_triangles()
    face_to_verts = []
    for poly in mesh.polygons:
        face_to_verts.append(set(mesh.loops[v].vertex_index for v in range(poly.loop_start, poly.loop_start + poly.loop_total)))

    batches = []
    current_batch = []
    current_verts = set()

    for f_idx, vset in enumerate(face_to_verts):
        new_count = len(current_verts | vset)
        if new_count <= max_vertices:
            current_batch.append(f_idx)
            current_verts |= vset
        else:
            if current_batch:
                batches.append(current_batch)
            current_batch = [f_idx]
            current_verts = set(vset)
    if current_batch:
        batches.append(current_batch)
    return batches

def build_object_from_face_batch(src_mesh: bpy.types.Mesh, face_indices: list, name: str):
    bm = bmesh.new()
    bm.from_mesh(src_mesh)
    bm.verts.ensure_lookup_table()
    bm.faces.ensure_lookup_table()

    keep = set(face_indices)
    for i, f in enumerate(bm.faces):
        f.select = (i in keep)

    bmesh.ops.delete(bm, geom=[f for i, f in enumerate(bm.faces) if i not in keep], context='FACES')
    bmesh.ops.delete(bm, geom=[e for e in bm.edges if not e.link_faces], context='EDGES')
    bmesh.ops.delete(bm, geom=[v for v in bm.verts if not v.link_edges], context='VERTS')

    new_mesh = bpy.data.meshes.new(name)
    bm.to_mesh(new_mesh)
    bm.free()
    new_obj = bpy.data.objects.new(name, new_mesh)
    return new_obj

def vertex_count_of_mesh(mesh: bpy.types.Mesh) -> int:
    return len(mesh.vertices)

def pack_bins(items, max_per_bin: int, max_bins: int):
    sorted_items = sorted(items, key=lambda x: x[1], reverse=True)
    bins = []
    bin_vert_counts = []

    for item_id, v in sorted_items:
        placed = False
        for b_idx, current in enumerate(bin_vert_counts):
            if current + v <= max_per_bin:
                bins[b_idx].append(item_id)
                bin_vert_counts[b_idx] += v
                placed = True
                break
        if not placed:
            bins.append([item_id])
            bin_vert_counts.append(v)

    if len(bins) > max_bins:
        while len(bins) > max_bins:
            pairs = sorted([(i, c) for i, c in enumerate(bin_vert_counts)], key=lambda t: t[1])
            i1 = pairs[0][0]
            i2 = pairs[1][0]
            bins[i1].extend(bins[i2])
            bin_vert_counts[i1] += bin_vert_counts[i2]
            del bins[i2]
            del bin_vert_counts[i2]
    return bins, bin_vert_counts

def add_decimate_if_needed(obj: bpy.types.Object, max_vertices: int):
    v = len(obj.data.vertices)
    if v > max_vertices and v > 0:
        ratio = max_vertices / float(v)
        mod = obj.modifiers.new(name="AutoDecimate", type='DECIMATE')
        mod.ratio = max(0.01, min(0.99, ratio))
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.modifier_apply(modifier=mod.name)

def copy_materials_to_object(target_obj: bpy.types.Object, source_objs: list):
    material_set = set()
    materials_list = []
    
    for src_obj in source_objs:
        if src_obj.data and hasattr(src_obj.data, 'materials'):
            for mat in src_obj.data.materials:
                if mat and mat not in material_set:
                    material_set.add(mat)
                    materials_list.append(mat)
    
    target_obj.data.materials.clear()
    for mat in materials_list:
        target_obj.data.materials.append(mat)

def join_objects_with_materials(context, objs: list, final_name: str):
    if not objs:
        return None
    
    for o in context.selected_objects:
        o.select_set(False)
    
    for o in objs:
        o.select_set(True)
    
    context.view_layer.objects.active = objs[0]
    
    bpy.ops.object.join()
    
    joined = context.view_layer.objects.active
    joined.name = final_name
    
    return joined


class OBJECT_OT_merge_by_vertex_budget(Operator):
    bl_idname = "object.merge_by_vertex_budget"
    bl_label = "Merge to Bins"
    bl_description = "Merge meshes into bins respecting vertex budget"
    bl_options = {'REGISTER', 'UNDO'}
    
    source_mode: EnumProperty(
        name="Source",
        items=[
            ('ALL', "All", "All meshes in scene"),
            ('COLLECTION', "Collection", "Specific collection"),
            ('SELECTED', "Selected", "Selected objects only")
        ],
        default='ALL'
    )
    source_collection: StringProperty(name="Source Collection", default="")
    max_vertices_per_mesh: IntProperty(name="Max Vertices", default=20000, min=1000)
    max_output_meshes: IntProperty(name="Max Outputs", default=10, min=1)
    output_collection_name: StringProperty(name="Output Collection", default="Merged_Output")
    apply_modifiers: BoolProperty(name="Apply Modifiers", default=True)
    
    def execute(self, context):
        wm = context.window_manager
        self.source_mode = wm.vb_source_mode
        self.source_collection = wm.vb_source_collection
        self.max_vertices_per_mesh = wm.vb_max_vertices
        self.max_output_meshes = wm.vb_max_outputs
        self.output_collection_name = wm.vb_output_collection
        self.apply_modifiers = wm.vb_apply_modifiers
        
        src_objs = iter_mesh_objects(self.source_mode, self.source_collection)
        
        if not src_objs:
            if self.source_mode == 'SELECTED':
                self.report({'WARNING'}, "No mesh objects selected in the viewport.")
            else:
                self.report({'WARNING'}, "No mesh objects found for the selected source.")
            return {'CANCELLED'}

        total_vertices = 0
        for obj in src_objs:
            mesh_copy, _ = evaluated_mesh_copy(obj, self.apply_modifiers)
            vcount = vertex_count_of_mesh(mesh_copy)
            total_vertices += vcount
            bpy.data.meshes.remove(mesh_copy, do_unlink=True)
        
        max_possible_vertices = self.max_vertices_per_mesh * self.max_output_meshes
        
        if total_vertices > max_possible_vertices:
            self.report({'ERROR'}, 
                f"Cannot fit {total_vertices:,} total vertices from {len(src_objs)} objects into {self.max_output_meshes} meshes "
                f"with {self.max_vertices_per_mesh:,} vertices each (max capacity: {max_possible_vertices:,}). "
                f"Increase Max Vertices per Mesh or Max Output Meshes.")
            return {'CANCELLED'}

        out_col = ensure_collection(self.output_collection_name)

        parts = []
        temp_objs = []
        obj_to_original = {}

        for obj in src_objs:
            mesh_copy, _mw = evaluated_mesh_copy(obj, self.apply_modifiers)
            vcount = vertex_count_of_mesh(mesh_copy)

            if vcount <= self.max_vertices_per_mesh:
                new_obj = bpy.data.objects.new(obj.name + "_part", mesh_copy)
                new_obj.matrix_world = obj.matrix_world.copy()
                
                if obj.data and hasattr(obj.data, 'materials'):
                    for mat in obj.data.materials:
                        if mat:
                            new_obj.data.materials.append(mat)
                
                bpy.context.scene.collection.objects.link(new_obj)
                temp_objs.append(new_obj)
                obj_to_original[new_obj.name] = obj
                parts.append((new_obj, vcount))
            else:
                batches = chunk_faces_by_vertex_limit(mesh_copy, self.max_vertices_per_mesh)
                
                if obj.data and hasattr(obj.data, 'materials'):
                    for mat in obj.data.materials:
                        if mat:
                            mesh_copy.materials.append(mat)
                
                for i, face_batch in enumerate(batches):
                    part_obj = build_object_from_face_batch(mesh_copy, face_batch, f"{obj.name}_chunk_{i:03d}")
                    part_obj.matrix_world = obj.matrix_world.copy()
                    
                    if obj.data and hasattr(obj.data, 'materials'):
                        for mat in obj.data.materials:
                            if mat:
                                part_obj.data.materials.append(mat)
                    
                    bpy.context.scene.collection.objects.link(part_obj)
                    temp_objs.append(part_obj)
                    obj_to_original[part_obj.name] = obj
                    parts.append((part_obj, len(part_obj.data.vertices)))
                
                bpy.data.meshes.remove(mesh_copy, do_unlink=True)

        items = [(o.name, v) for (o, v) in parts]
        name_to_obj = {o.name: o for (o, _) in parts}
        bins, bin_vert_counts = pack_bins(items, self.max_vertices_per_mesh, self.max_output_meshes)

        result_objs = []
        for idx, bin_names in enumerate(bins, start=1):
            objs = [name_to_obj[n] for n in bin_names]
            
            joined = join_objects_with_materials(context, objs, f"Combined_{idx:02d}")
            
            if joined:
                for col in list(joined.users_collection):
                    col.objects.unlink(joined)
                out_col.objects.link(joined)
                
                add_decimate_if_needed(joined, self.max_vertices_per_mesh)
                result_objs.append(joined)

        for o in temp_objs:
            try:
                if not o or not o.name:
                    continue
                if o.name in [r.name for r in result_objs]:
                    continue
                if o.name not in bpy.data.objects:
                    continue
                    
                for col in list(o.users_collection):
                    col.objects.unlink(o)
                    
                if o.data and o.data.users == 1:
                    bpy.data.meshes.remove(o.data, do_unlink=True)
                    
                if o.name in bpy.data.objects:
                    bpy.data.objects.remove(o, do_unlink=True)
            except ReferenceError:
                continue
            except Exception:
                continue

        self.report({'INFO'}, 
            f"Created {len(result_objs)} merged objects in '{out_col.name}' "
            f"({total_vertices:,} total vertices).")
        return {'FINISHED'}


class VIEW3D_PT_material_toolkit(Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Material Toolkit'
    bl_label = "Material Toolkit"
    
    def draw(self, context):
        layout = self.layout
        wm = context.window_manager
        
        box = layout.box()
        row = box.row()
        row.label(text="Texture Export", icon='EXPORT')
        box.operator("export.textures", text="Export All Textures", icon='IMAGE_DATA')
        
        layout.separator()
        
        box = layout.box()
        row = box.row()
        row.label(text="Material Converter", icon='MATERIAL')
        box.operator("material.convert_to_principled", text="Convert to Principled BSDF", icon='SHADING_RENDERED')
        box.label(text="• Maintains UV/projection setup", icon='BLANK1')
        box.label(text="• Blender may be unresponsive", icon='BLANK1')
        
        layout.separator()
        
        box = layout.box()
        row = box.row()
        row.label(text="Batch Merger", icon='AUTOMERGE_ON')
        
        col = box.column(align=True)
        col.label(text="Source")
        
        row = col.row(align=True)
        row.prop(wm, "vb_source_mode", expand=True)
        
        if wm.vb_source_mode == 'COLLECTION':
            col.prop_search(wm, "vb_source_collection", bpy.data, "collections", text="")
        elif wm.vb_source_mode == 'SELECTED':
            selected_meshes = [o for o in context.selected_objects if o.type == 'MESH']
            col.label(text=f"Selected: {len(selected_meshes)}", icon='INFO')

        col.separator()
        col.prop(wm, "vb_max_vertices", text="Max Vertices per Mesh")
        col.prop(wm, "vb_max_outputs", text="Max Output Meshes")
        col.prop(wm, "vb_output_collection", text="Output Collection")
        col.prop(wm, "vb_apply_modifiers", text="Apply Modifiers")

        col.separator()
        col.operator(OBJECT_OT_merge_by_vertex_budget.bl_idname, text="Merge to Bins", icon='AUTOMERGE_ON')


def register_props():
    wm = bpy.types.WindowManager
    wm.vb_source_mode = EnumProperty(
        name="Source",
        items=[
            ('ALL', "All", "All meshes in scene"),
            ('COLLECTION', "Collection", "Specific collection"),
            ('SELECTED', "Selected", "Selected objects only")
        ],
        default='ALL'
    )
    wm.vb_source_collection = StringProperty(name="Source Collection", default="")
    wm.vb_max_vertices = IntProperty(name="Max Vertices", default=20000, min=1000)
    wm.vb_max_outputs = IntProperty(name="Max Outputs", default=10, min=1)
    wm.vb_output_collection = StringProperty(name="Output Collection", default="Merged_Output")
    wm.vb_apply_modifiers = BoolProperty(name="Apply Modifiers", default=True)

def unregister_props():
    wm = bpy.types.WindowManager
    del wm.vb_source_mode
    del wm.vb_source_collection
    del wm.vb_max_vertices
    del wm.vb_max_outputs
    del wm.vb_output_collection
    del wm.vb_apply_modifiers

def invoke_with_wm_props(self, context, event):
    wm = context.window_manager
    self.source_mode = wm.vb_source_mode
    self.source_collection = wm.vb_source_collection
    self.max_vertices_per_mesh = wm.vb_max_vertices
    self.max_output_meshes = wm.vb_max_outputs
    self.output_collection_name = wm.vb_output_collection
    self.apply_modifiers = wm.vb_apply_modifiers
    return self.execute(context)


classes = (
    EXPORT_OT_textures,
    MATERIAL_OT_convert_to_principled,
    OBJECT_OT_merge_by_vertex_budget,
    VIEW3D_PT_material_toolkit,
)


def register():
    register_props()
    for cls in classes:
        bpy.utils.register_class(cls)
    OBJECT_OT_merge_by_vertex_budget.invoke = invoke_with_wm_props


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    unregister_props()


if __name__ == "__main__":
    register()
