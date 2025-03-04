import click
import numpy as np

from spectral_util import cli
from spectral_util.io import load_data, write_cog

# Define common arguments
def common_arguments(f):
    # put this (counter-intuitively) in reverse order
    f = click.argument('output_file')(f)
    f = click.argument('input_file')(f)
    f = click.option('--ortho', is_flag=True, help='Orthorectify the output; only relevant if the input format is non-orthod')(f)
    return f


def shared_options(f):
    f = click.option('--ortho', is_flag=True, help='Orthorectify the output; only relevant if the input format is non-orthod')
    return f


@cli.command(no_args_is_help=True)
@common_arguments
@click.option('--red_wl', default=660, help='Red band wavelength [nm]')
@click.option('--nir_wl', default=800, help='NIR band wavelength [nm]')
@click.option('--red_width', default=0, help='Red band width [nm]; 0 = single wavelength')
@click.option('--nir_width', default=0, help='NIR band width [nm]; 0 = single wavelength')
def ndvi(input_file, output_file, ortho, red_wl, nir_wl, red_width, nir_width):
    """
    Calculate NDVI.

    \b
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

    red = rfl[..., meta.wl_index(red_wl, red_width)]
    nir = rfl[..., meta.wl_index(nir_wl, nir_width)]

    ndvi = (nir - red) / (nir + red)
    ndvi = ndvi.squeeze()
    ndvi[nir == meta.nodata_value] = -9999
    ndvi[np.isfinite(ndvi) == False] = -9999
    ndvi = ndvi.reshape((ndvi.shape[0], ndvi.shape[1], 1))

    write_cog(output_file, ndvi, meta, ortho=ortho)


@cli.command(no_args_is_help=True)
@common_arguments
@click.option('--nir_wl', default=866, help='Red band wavelength [nm]')
@click.option('--swir_wl', default=2198, help='NIR band wavelength [nm]')
@click.option('--nir_width', default=0, help='Red band width [nm]; 0 = single wavelength')
@click.option('--swir_width', default=0, help='NIR band width [nm]; 0 = single wavelength')
def nbr(input_file, output_file, ortho, nir_wl, swir_wl, nir_width, swir_width):
    """
    Calculate NBR.

    \b
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

    nir = rfl[..., meta.wl_index(nir_wl)]
    swir = rfl[..., meta.wl_index(swir_wl)]

    nbr = (nir - swir) / (swir + nir)
    nbr = nbr.squeeze().astype(np.float32)
    nbr[nir == meta.nodata_value] = -9999
    nbr[np.isfinite(nbr) == False] = -9999
    nbr = nbr.reshape((nbr.shape[0], nbr.shape[1], 1))

    write_cog(output_file, nbr, meta, ortho=ortho, nodata_value=-9999)


@cli.command(no_args_is_help=True)
@common_arguments
@click.option('--red_wl', default=650, help='Red band wavelength [nm]')
@click.option('--green_wl', default=560, help='Green band wavelength [nm]')
@click.option('--blue_wl', default=460, help='Blue band width [nm]')
@click.option('--stretch', default=[2,98], nargs=2, type=int, help='stretch the rgb; set to -1 -1 to not stretch')
@click.option('--scale', default=[-1,-1,-1,-1,-1,-1], nargs=6, type=float, help='scale the rgb to these min, max pairs')
def rgb(input_file, output_file, ortho, red_wl, green_wl, blue_wl, stretch, scale):
    """
    Calculate RGB composite.

    \b
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

    rgb = rfl[..., np.array([meta.wl_index(x) for x in [red_wl, green_wl, blue_wl]])]
    if stretch[0] != -1 and stretch[1] != -1:
        rgb[rgb == meta.nodata_value] = np.nan
        rgb -= np.nanpercentile(rgb, stretch[0], axis=(0, 1))
        rgb /= np.nanpercentile(rgb, stretch[1], axis=(0, 1))
        rgb[rgb < 0] = 0
        rgb[rgb > 1] = 1
        mask = np.isfinite(rgb[...,0]) == False
        rgb[mask,:] = 0
        rgb = (rgb * 255).astype(np.uint8)

        rgb[rgb == 0] = 1
        rgb[mask,:] = 0
        nodata_value = 0
    elif np.all(np.array(scale) != -1):
        mask = rgb[...,0] == meta.nodata_value
        rgb[...,0] = (np.clip(rgb[...,0], scale[0], scale[1]) - scale[0]) / (scale[1] - scale[0])
        rgb[...,1] = (np.clip(rgb[...,1], scale[2], scale[3]) - scale[2]) / (scale[3] - scale[2])
        rgb[...,2] = (np.clip(rgb[...,2], scale[4], scale[5]) - scale[4]) / (scale[5] - scale[4])

        rgb = (rgb * 255).astype(np.uint8)
        rgb[rgb == 0] = 1
        rgb[mask,:] = 0

        nodata_value = 0
    else:
        nodata_value = meta.nodata_value


    write_cog(output_file, rgb, meta, ortho=ortho, nodata_value=nodata_value)


# @cli.command(no_args_is_help=True)
# @common_arguments
# def ndwi(input_file, output_file):
#     click.echo("This doesn't work yet.")
