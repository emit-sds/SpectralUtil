# SpectralUtil

SpectralUtil provides command-line and Python tools for common imaging spectroscopy workflows, including quicklooks, plotting, mosaicking, Earthdata download helpers, and format conversion.

Supported products include:

- [AVIRIS-3 L1B Radiance](https://doi.org/10.3334/ORNLDAAC/2356)
- [AVIRIS-3 L2A Reflectance](https://daac.ornl.gov/cgi-bin/dsviewer.pl?ds_id=2357)
- AVIRIS-NG L2A Reflectance
- [EMIT L1B Radiance](https://lpdaac.usgs.gov/products/emitl1bradv001/)
- [EMIT L2A Reflectance](https://lpdaac.usgs.gov/products/emitl2arflv001/)
- [PRISM Reflectance](https://daacweb-prod.ornl.gov/BIOSCAPE/guides/BioSCape_PRISM_L2A_RFL.html)
- ENVI format inputs

## Installation

Recommended with [pixi](https://pixi.sh):

```bash
pixi install
```

Or with pip:

```bash
pip install spectral_util
```

## Python API

Common IO access patterns - probably the single most useful thing in this repo - are provided acorss datasets. Use:
```
from spectral_util.spec_io import load_data
meta, rfl = load_data('AV320250809t182459_000_L1B_RDN_4842d6a3_RDN.nc')
print(len(meta.wl))
print(rfl.shape)
```

Will return:
```
284
(1280, 1234, 284)
```

load_data supports options for orthoing nc files that are not natively orthod (e.g. radiance .nc files) during read, and lazy loading (still only partially supported).  All CLI options used below (and more) have supporint api function calls.

## Running the CLI

If installed into your current environment:

```bash
spectral_util --help
```

If using the local pixi environment:

```bash
pixi run spectral_util --help
```

The top-level command groups are:

- download
- mosaic
- quicklooks
- plot
- reformat

## CLI Examples by Section

### download

#### download get-fid
```bash
# Download files for one specific FID and product short-name
spectral_util download get-fid ./downloads AV3_L1B_RDN_2356 AV320250809t182459_000

# Download only the RDN granule component (used in examples below)
spectral_util download get-fid ./downloads AV3_L1B_RDN_2356 AV320250809t182459_000 --subfile RDN --version 1

# Overwrite previously-downloaded files
spectral_util download get-fid ./downloads AV3_L1B_RDN_2356 AV320250809t182459_000 --subfile RDN --version 1 --overwrite
```

Assume the downloaded file used in subsequent examples is:

```text
./downloads/AV320250809t182459_000_L1B_RDN_4842d6a3_RDN.nc
```

### quicklooks

#### quicklooks rgb
```bash
# Basic RGB quicklook
spectral_util quicklooks rgb ./downloads/AV320250809t182459_000_L1B_RDN_4842d6a3_RDN.nc output_rgb.tif

# RGB with custom wavelengths and percentile stretch
spectral_util quicklooks rgb ./downloads/AV320250809t182459_000_L1B_RDN_4842d6a3_RDN.nc output_rgb.tif --red_wl 660 --green_wl 560 --blue_wl 460 --stretch 2 98

# RGB with explicit per-channel scaling (disable stretch)
spectral_util quicklooks rgb ./downloads/AV320250809t182459_000_L1B_RDN_4842d6a3_RDN.nc output_rgb_scaled.tif --stretch -1 -1 --scale 0 0.2 0 0.2 0 0.2

# Orthorectified RGB output when GLT is available
spectral_util quicklooks rgb ./downloads/AV320250809t182459_000_L1B_RDN_4842d6a3_RDN.nc output_rgb_ortho.tif --ortho
```

#### quicklooks ndvi
```bash
# Default NDVI
spectral_util quicklooks ndvi ./downloads/AV320250809t182459_000_L1B_RDN_4842d6a3_RDN.nc output_ndvi.tif

# NDVI with custom wavelengths and band widths
spectral_util quicklooks ndvi ./downloads/AV320250809t182459_000_L1B_RDN_4842d6a3_RDN.nc output_ndvi_custom.tif --red_wl 665 --nir_wl 842 --red_width 20 --nir_width 20

# Orthorectified NDVI
spectral_util quicklooks ndvi ./downloads/AV320250809t182459_000_L1B_RDN_4842d6a3_RDN.nc output_ndvi_ortho.tif --ortho
```

#### quicklooks nbr
```bash
# Default NBR
spectral_util quicklooks nbr ./downloads/AV320250809t182459_000_L1B_RDN_4842d6a3_RDN.nc output_nbr.tif

# NBR with custom NIR/SWIR settings
spectral_util quicklooks nbr ./downloads/AV320250809t182459_000_L1B_RDN_4842d6a3_RDN.nc output_nbr_custom.tif --nir_wl 860 --swir_wl 2200 --nir_width 20 --swir_width 40

# Orthorectified NBR
spectral_util quicklooks nbr ./downloads/AV320250809t182459_000_L1B_RDN_4842d6a3_RDN.nc output_nbr_ortho.tif --ortho
```

### plot

#### plot plot-basic-overview
```bash
# Show RGB + selected spectra interactively
spectral_util plot plot-basic-overview ./downloads/AV320250809t182459_000_L1B_RDN_4842d6a3_RDN.nc

# Save basic overview plot and choose random sampling
spectral_util plot plot-basic-overview ./downloads/AV320250809t182459_000_L1B_RDN_4842d6a3_RDN.nc --output_file spectra_overview.png --n_points 8 --method random

# K-means representative spectra
spectral_util plot plot-basic-overview ./downloads/AV320250809t182459_000_L1B_RDN_4842d6a3_RDN.nc --output_file spectra_kmeans.png --n_points 10 --method kmeans
```

#### plot plot-pcs
```bash
# Plot first 20 principal components
spectral_util plot plot-pcs ./downloads/AV320250809t182459_000_L1B_RDN_4842d6a3_RDN.nc --output_file pcs.png

# Plot a custom PC range with controlled sampling
spectral_util plot plot-pcs ./downloads/AV320250809t182459_000_L1B_RDN_4842d6a3_RDN.nc --output_file pcs_10_29.png --first_pc 10 --last_pc 29 --n_points 20000 --seed 42
```

#### plot plot-mnf
```bash
# Plot first 20 MNF components
spectral_util plot plot-mnf ./downloads/AV320250809t182459_000_L1B_RDN_4842d6a3_RDN.nc --output_file mnf.png

# Custom MNF count and covariance sampling options
spectral_util plot plot-mnf ./downloads/AV320250809t182459_000_L1B_RDN_4842d6a3_RDN.nc --output_file mnf_30.png --n_mnf 30 --n_points 20000 --seed 42

# Estimate noise along rows instead of columns
spectral_util plot plot-mnf ./downloads/AV320250809t182459_000_L1B_RDN_4842d6a3_RDN.nc --output_file mnf_diff_rows.png --diff_dim 0
```

### mosaic

#### mosaic build-obs-nc
```bash
# Build GLT mosaic from a text file of OBS inputs
spectral_util mosaic build-obs-nc mosaic_glt.tif obs_files.txt --x_resolution 30 --y_resolution -30 --output_epsg 4326

# Build with explicit extent and criteria selection
spectral_util mosaic build-obs-nc mosaic_glt_utm.tif obs_files.txt \
	--x_resolution 60 --y_resolution -60 --output_epsg 32611 \
	--target_extent_ul_lr 3800000 420000 3700000 520000 \
	--criteria_band 0 --criteria_mode min --n_cores 8

# Exclude selected files and set max nearest-neighbor distance
spectral_util mosaic build-obs-nc mosaic_filtered.tif obs_files.txt \
	--ignore_file_list ignore_files.txt --x_resolution 30 --y_resolution -30 --max_distance 45
```

#### mosaic apply-glt
```bash
# Apply GLT to a single raw file
spectral_util mosaic apply-glt mosaic_glt.tif ./downloads/AV320250809t182459_000_L1B_RDN_4842d6a3_RDN.nc output_ortho.tif

# Apply GLT to a list of raw files and subset bands
spectral_util mosaic apply-glt mosaic_glt.tif raw_files.txt output_rgb.tif --bands 34 --bands 22 --bands 10

# Write ENVI output with custom nodata handling
spectral_util mosaic apply-glt mosaic_glt.tif raw_files.txt output_envi --output_format envi --nodata_value -9999 --glt_nodata_value 0
```

#### mosaic stack-glts
```bash
# Merge several GLTs and their file lists into one stack
spectral_util mosaic stack-glts glt_files.txt obs_file_lists.txt stacked_glt.tif stacked_file_list.txt
```

### reformat

#### reformat nc-to-envi
```bash
# Convert NetCDF to ENVI
spectral_util reformat nc-to-envi ./downloads/AV320250809t182459_000_L1B_RDN_4842d6a3_RDN.nc output_envi

# Convert NetCDF to orthorectified ENVI output
spectral_util reformat nc-to-envi ./downloads/AV320250809t182459_000_L1B_RDN_4842d6a3_RDN.nc output_envi_ortho --ortho

# Overwrite existing output
spectral_util reformat nc-to-envi ./downloads/AV320250809t182459_000_L1B_RDN_4842d6a3_RDN.nc output_envi --overwrite
```



## Help

Use help at any level:
spectral_util reformat nc-to-envi ./downloads/AV320250809t182459_000_L1B_RDN_4842d6a3_RDN.nc output_envi
```bash
spectral_util --help
spectral_util reformat nc-to-envi ./downloads/AV320250809t182459_000_L1B_RDN_4842d6a3_RDN.nc output_envi_ortho --ortho
spectral_util quicklooks rgb --help
spectral_util plot --help
spectral_util reformat nc-to-envi ./downloads/AV320250809t182459_000_L1B_RDN_4842d6a3_RDN.nc output_envi --overwrite