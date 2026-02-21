# SpectralUtils

This is a package for basic manipulation of imaging spectroscopy data. It is designed to accommodate data from a variety of instruments, abstracting out the specifics of the file delivery. Currently, the package supports data from the following instruments / product levels:

- [AVIRIS-3 L1B Radiance](https://doi.org/10.3334/ORNLDAAC/2356)
- [AVIRIS-3 L2A Reflectance](https://daac.ornl.gov/cgi-bin/dsviewer.pl?ds_id=2357)
- AVIRIS-NG L2A Reflectance
- [EMIT L1B Radiance](https://lpdaac.usgs.gov/products/emitl1bradv001/)
- [EMIT L2A Reflectance](https://lpdaac.usgs.gov/products/emitl2arflv001/)
- Any data in ENVI format

## Installation

To install the package, we recommend using [pixi](https://pixi.sh):

```bash
pixi install
```

Or you can install via pip, though depending on the system you may experience gdal-related issues:

```bash
pip install spectral_util
```

## CLI Interface

The package provides a unified CLI interface with the following commands:

### Quicklooks (RGB, NDVI, NBR)

```bash
# Standard RGB
spectral_util quicklooks rgb input_file.tif output_rgb.tif

# RGB with custom wavelengths
spectral_util quicklooks rgb input_file.tif output_rgb.tif --red_wl 660 --green_wl 560 --blue_wl 460

# NDVI calculation
spectral_util quicklooks ndvi input_file.tif output_ndvi.tif

# NBR calculation
spectral_util quicklooks nbr input_file.tif output_nbr.tif
```

### Mosaic Operations

```bash
# Build observation mosaic
spectral_util mosaic build-obs-nc output_file.nc input_file_list.txt --x_resolution 10 --y_resolution -10

# Apply GLT to files
spectral_util mosaic apply-glt glt_file.txt raw_files.txt output_file.tif

# Stack GLTs
spectral_util mosaic stack-glts glt_files.txt obs_file_lists.txt output_glt_file.txt output_file_list.txt
```

### Data Download (AV3 and EMIT)

```bash
# Download AV3 data
spectral_util av3-download /path/to/output --temporal 2024-10-04T16:00:00 2024-10-04T17:00:00 --bounding_box -103.74460188 32.22680624 -103.74481188 32.22700624

# Download EMIT data
spectral_util emit-download /path/to/output --temporal 2024-10-04T16:00:00 2024-10-04T17:00:00 --bounding_box -103.74460188 32.22680624 -103.74481188 32.22700624
```

### Reformat an EMIT file
```
spectral_util reformat nc-to-envi EMIT_L2A_RFL_001_20250525T090616_2514506_004.nc test_emit_rfl --ortho
```

## Python API

For programmatic use, the package can be imported directly:

```python
from spectral_util.common import quicklooks
from spectral_util.mosaic import mosaic
from spectral_util.ea_assist import earthaccess_helpers_AV3, earthaccess_helpers_EMIT
```

## Utilities

A series of utility scripts are provided to help with common tasks, such as:

### Standard RGB, with stretching
```bash
spectral_util quicklooks rgb EMIT_L1B_RAD_001_20240715T195403_2419712_015.nc emit_rgb.tif
```

### Standard RGB, custom wavelengths
```bash
spectral_util quicklooks rgb EMIT_L1B_RAD_001_20240715T195403_2419712_015.nc emit_rgb.tif --red_wl 2360 --green_wl 1800 --blue_wl 1000
```

### Spectral Indices
```bash
spectral_util quicklooks nbr EMIT_L1B_RAD_001_20240715T195403_2419712_015.nc emit_nbr.tif
spectral_util quicklooks ndvi EMIT_L1B_RAD_001_20240715T195403_2419712_015.nc emit_ndvi.tif
```

## Command Reference

### Quicklooks Commands
- `rgb`: Create RGB composite
- `ndvi`: Calculate NDVI (Normalized Difference Vegetation Index)
- `nbr`: Calculate NBR (Normalized Burn Ratio)

### Mosaic Commands
- `build-obs-nc`: Build observation mosaic from input files
- `apply-glt`: Apply Geolocation Transformation to files
- `stack-glts`: Stack multiple GLTs into a single output

### Download Commands
- `get-fid`: Download a specific FID for a given product via CMR.
- `av3-download`: Download AV3 data from Earthdata - currently niche usage, to be updated
- `emit-download`: Download EMIT data from Earthdata - currently niche usage, to be updated

## Reformat Commands
- `nc-to-envi`: Convert a netcdf data file to an envi binary file, with options

## Options

Most commands have custom options - call the given utility with --help to see all options.

## Examples

### RGB with custom wavelengths and stretching
```bash
spectral_util quicklooks rgb input.nc output.tif --red_wl 660 --green_wl 560 --blue_wl 460 --stretch 2 98
```

### Download with specific parameters
```bash
spectral_util emit-download /data/output --temporal 2024-01-01T00:00:00 2024-12-31T23:59:59 --bounding_box -120 30 -110 40
```

## Help

For detailed help on any command:
```bash
spectral_util <command> --help
```

For example:
```bash
spectral_util quicklooks rgb --help
spectral_util mosaic build-obs-nc --help
spectral_util av3-download --help
```