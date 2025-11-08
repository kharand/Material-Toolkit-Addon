# Material Toolkit for Blender

A comprehensive Blender addon for texture management, material conversion, and batch mesh operations.

## Features

### Texture Export
Export all textures from your Blender scene to a selected directory with a single click. Handles packed textures, file-based textures, and generated images automatically.

### Material Converter
Convert materials to Principled BSDF shader while preserving texture mappings and UV setups.

**Currently Supported Materials:**
- Diffuse BSDF
- Emission Shader
- Mix Shader (Diffuse/Emission combinations)

The converter intelligently detects texture nodes, maintains projection settings (UV, Box, etc.), and reconstructs the material with proper base color connections.

### Batch Mesh Merger
Merge multiple objects into optimized meshes based on vertex budget constraints. Perfect for game engine optimization and performance management.

- Set maximum vertices per output mesh
- Control number of output meshes
- Choose source: All meshes, specific collection, or selected objects
- Optional modifier application before merging
- Smart bin-packing algorithm for efficient distribution

## Installation

### Method 1: Direct ZIP Installation (Recommended)

1. Download the latest `Material_Toolkit.zip` from the [Releases](../../releases) page
2. Open Blender
3. Go to `Edit` → `Preferences` → `Add-ons`
4. Click `Install...` and select the downloaded ZIP file
5. Enable "Material Toolkit" in the addon list

### Method 2: Manual Installation

1. Download `Material_Toolkit.py`
2. Copy it to your Blender addons folder:
   - **Windows:** `%APPDATA%\Blender Foundation\Blender\{version}\scripts\addons\`
   - **macOS:** `~/Library/Application Support/Blender/{version}/scripts/addons/`
   - **Linux:** `~/.config/blender/{version}/scripts/addons/`
3. Enable the addon in Blender preferences

## Usage

Access the addon from the 3D Viewport sidebar:
**View3D → Sidebar (N key) → Material Toolkit tab**

### Export Textures
1. Click "Export All Textures"
2. Select destination folder
3. All textures (packed and file-based) will be saved

### Convert Materials
1. Click "Convert to Principled BSDF"
2. Wait for conversion (Blender may appear unresponsive during batch processing)
3. All compatible materials will be converted while maintaining their texture connections

### Merge Objects
1. Select source mode (All/Collection/Selected)
2. Set vertex budget parameters
3. Specify output collection name
4. Click "Merge to Bins"

## Requirements

- Blender 3.0.0 or higher (Works with versions up to 4.5.4 LTS currently)

## Author

Created by **kharand**

## License

This addon is provided as-is for use in Blender projects.

## Support

For issues, questions, or feature requests, please use the [Issues](../../issues) tab on GitHub.
