import click
import os
import numpy as np
from scipy.spatial import KDTree
from scipy.signal import convolve2d

import time
import spectral_util.spec_io as spec_io
from osgeo import osr
import pyproj
import logging
from tqdm import tqdm
from copy import deepcopy
osr.UseExceptions()

@click.command()
@click.argument('nc_file')
@click.argument('output_file')
@click.option('--ortho', is_flag=True)
@click.option('--overwrite', is_flag=True)
def nc_to_envi(nc_file: str, output_file: str, ortho=False, overwrite=False):
    """Convert a NetCDF file to ENVI format.

    \b
    Args:
        nc_file (str): Path to the input NetCDF file.
        output_file (str): Path to the output ENVI file.
        ortho (bool, optional): Whether to orthorectify the data. Defaults to False.
    """
    meta, dat = spec_io.load_data(nc_file, load_glt=True)
    dat = np.array(dat)

    if ortho and meta.orthoable:
        dat = spec_io.ortho_data(dat, meta.glt)

    if os.path.isfile(output_file) and not overwrite:
        raise FileExistsError(f'Output file {output_file} already exists. Use --overwrite to overwrite it.')

    spec_io.write_envi_file(dat, meta, output_file)





@click.group()
def cli():
    pass


cli.add_command(nc_to_envi)

if __name__ == '__main__':
    cli()
