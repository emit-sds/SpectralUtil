#!/usr/bin/env python3
"""Quicklooks module for spectral utilities."""

import click
import numpy as np
from spectral_util.spec_io import load_data, write_cog

# Define common arguments
def common_arguments(f):
    f = click.argument('output_file')(f)
    f = click.argument('input_file')(f)
    f = click.option('--ortho', is_flag=True, help='Orthorectify the output; only relevant if the input format is non-orthod')(f)
    return f

def shared_options(f):
    f = click.option('--ortho', is_flag=True, help='Orthorectify the output; only relevant if the input format is non-orthod')
    return f

def calc_index(data, meta, first_wl, second_wl, first_width=0, second_width=0, nodata=-9999):
    """
    Calculate a spectral index.
    Args:
        data (numpy like): Input data array.
        meta (Metadata): Metadata object containing wavelength information.
        first_wl (int): Wavelength for the first band [nm].
        second_wl (int): Wavelength for the second band [nm].
        first_width (int): Width for the first band [nm]; 0 = single wavelength.
        second_width (int): Width for the second band [nm]; 0 = single wavelength.
    """
    first = data[..., meta.wl_index(first_wl, first_width)]
    second = data[..., meta.wl_index(second_wl, second_width)]
    if len(first.shape) == 3:
        first = np.mean(first, axis=-1)
    if len(second.shape) == 3:
        second = np.mean(second, axis=-1)

    index = (first - second) / (first + second)
    index = index.squeeze()
    index[first == meta.nodata_value] = nodata
    index[np.isfinite(index) == False] = nodata

    return index


@click.command()
@common_arguments
@click.option('--red_wl', default=660, help='Red band wavelength [nm]')
@click.option('--nir_wl', default=800, help='NIR band wavelength [nm]')
@click.option('--red_width', default=0, help='Red band width [nm]; 0 = single wavelength')
@click.option('--nir_width', default=0, help='NIR band width [nm]; 0 = single wavelength')
def ndvi(input_file, output_file, ortho, red_wl, nir_wl, red_width, nir_width):
    """
    Calculate NDVI.

    Args:
        input_file (str): Path to the input file.
        output_file (str): Path to the output file.
        ortho (bool): Orthorectify the output.
        red_wl (int): Red band wavelength [nm].
        nir_wl (int): NIR band wavelength [nm].
        red_width (int): Red band width [nm]; 0 = single wavelength.
        nir_width (int): NIR band width [nm]; 0 = single wavelength.
    """
    click.echo(f"Running NDVI Calculation on {input_file}")
    meta, rfl = load_data(input_file, lazy=True, load_glt=ortho)
    ndvi = calc_index(rfl, meta, red_wl, nir_wl, red_width, nir_width)
    ndvi = ndvi.reshape((ndvi.shape[0], ndvi.shape[1], 1))
    write_cog(output_file, ndvi, meta, ortho=ortho)

@click.command()
@common_arguments
@click.option('--nir_wl', default=866, help='Red band wavelength [nm]')
@click.option('--swir_wl', default=2198, help='NIR band wavelength [nm]')
@click.option('--nir_width', default=0, help='Red band width [nm]; 0 = single wavelength')
@click.option('--swir_width', default=0, help='NIR band width [nm]; 0 = single wavelength')
def nbr(input_file, output_file, ortho, nir_wl, swir_wl, nir_width, swir_width):
    """
    Calculate NBR.

    Args:
        input_file (str): Path to the input file.
        output_file (str): Path to the output file.
        ortho (bool): Orthorectify the output.
        nir_wl (int): NIR band wavelength [nm].
        swir_wl (int): SWIR band wavelength [nm].
        nir_width (int): NIR band width [nm]; 0 = single wavelength.
        swir_width (int): SWIR band width [nm]; 0 = single wavelength.
    """

    click.echo(f"Running NBR Calculation on {input_file}")
    meta, rfl = load_data(input_file, lazy=True, load_glt=ortho)
    nbr = calc_index(rfl, meta, nir_wl, swir_wl, nir_width, swir_width)
    nbr = nbr.reshape((nbr.shape[0], nbr.shape[1], 1))
    write_cog(output_file, nbr, meta, ortho=ortho, nodata_value=-9999)


def get_rgb(rfl, meta, red_wl=650, green_wl=560, blue_wl=460, percentile_stretch=[2,98], scale=[-1,-1,-1,-1,-1,-1]):
    """
    Get RGB composite from reflectance data.

    Args:
        rfl (numpy like): Reflectance data array.
        meta (Metadata): Metadata object containing wavelength information.
        red_wl (int): Red band wavelength [nm].
        green_wl (int): Green band wavelength [nm].
        blue_wl (int): Blue band wavelength [nm].
        percentile_stretch [(int), (int)]: Stretch the RGB values to the percentile min & max listed here.  Set to -1, -1 to not stretch.
        scale [(int), (int), (int), (int), (int), (int)]: Scale the RGB values to the min & max listed here.  Set to -1s to not scale (default).

    Returns:
        numpy array: RGB composite array.

        IF percentile_stretch or scale is used, return is a uint8, otherwise it is the same dtype as the input reflectance data.
    """
    rgb = rfl[..., np.array([meta.wl_index(x) for x in [red_wl, green_wl, blue_wl]])]
    if percentile_stretch[0] != -1 and percentile_stretch[1] != -1:
        rgb[rgb == meta.nodata_value] = np.nan
        rgb -= np.nanpercentile(rgb, percentile_stretch[0], axis=(0, 1))
        rgb /= np.nanpercentile(rgb, percentile_stretch[1], axis=(0, 1))
        rgb[rgb < 0] = 0
        rgb[rgb > 1] = 1
        mask = np.isfinite(rgb[...,0]) == False
        rgb[mask,:] = 0
        rgb = (rgb * 255).astype(np.uint8)

        rgb[rgb == 0] = 1
        rgb[mask,:] = 0
    elif np.all(np.array(scale) != -1):
        mask = rgb[...,0] == meta.nodata_value
        rgb[...,0] = (np.clip(rgb[...,0], scale[0], scale[1]) - scale[0]) / (scale[1] - scale[0])
        rgb[...,1] = (np.clip(rgb[...,1], scale[2], scale[3]) - scale[2]) / (scale[3] - scale[2])
        rgb[...,2] = (np.clip(rgb[...,2], scale[4], scale[5]) - scale[4]) / (scale[5] - scale[4])

        rgb = (rgb * 255).astype(np.uint8)
        rgb[rgb == 0] = 1
        rgb[mask,:] = 0

    return rgb

@click.command()
@common_arguments
@click.option('--red_wl', default=650, help='Red band wavelength [nm]')
@click.option('--green_wl', default=560, help='Green band wavelength [nm]')
@click.option('--blue_wl', default=460, help='Blue band width [nm]')
@click.option('--stretch', default=[2,98], nargs=2, type=int, help='stretch the rgb; set to -1 -1 to not stretch')
@click.option('--scale', default=[-1,-1,-1,-1,-1,-1], nargs=6, type=float, help='scale the rgb to these min, max pairs')
def rgb(input_file, output_file, ortho, red_wl, green_wl, blue_wl, stretch, scale):
    """
    Calculate RGB composite.

    Args:
        input_file (str): Path to the input file.
        output_file (str): Path to the output file.
        ortho (bool): Orthorectify the output.
        red_wl (int): Red band wavelength [nm].
        green_wl (int): Green band wavelength [nm].
        blue_wl (int): Blue band wavelength [nm].
        stretch [(int), (int)]: Stretch the RGB values to the percentile min & max listed here.  Set to -1, -1 to not stretch.
        scale [(int), (int), (int), (int), (int), (int)]: Scale the RGB values to the min & max listed here.  Set to -1s to not scale (default).
    """
    if np.all(np.array(scale) != -1) and np.all(np.array(stretch) != -1):
        raise ValueError("Cannot set both stretch and scale")

    click.echo(f"Running RGB Calculation on {input_file}")
    meta, rfl = load_data(input_file, lazy=True, load_glt=ortho)
    rgb = get_rgb(rfl, meta, red_wl, green_wl, blue_wl, stretch, scale)

    nodata_value = meta.nodata_value
    if np.all(np.array(stretch) != -1) or np.all(np.array(scale) != -1):
        nodata_value = 0

    write_cog(output_file, rgb, meta, ortho=ortho, nodata_value=nodata_value)

