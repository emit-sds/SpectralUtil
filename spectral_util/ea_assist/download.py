import numpy as np
import os
import pdb
import click
import re
from typing import Tuple
import os
import requests

import earthaccess
import spectral_util.spec_io as spec_io





def get_scene(outdir: str, 
              short_name: str, 
              subfile: str = '.*',
              granule_name:str = None, 
              version: str= None, 
              temoporal: str= None, 
              bounding_box:Tuple[float, float, float, float] = None, 
              count: int = None, 
              overwrite: bool = False):
    earthaccess.login(persist=True)
    kargs = {
        'temporal': temoporal,
        'bounding_box': bounding_box,
        'count': count,
        'version': version,
        'granule_name': granule_name,
    }
    kargs = {key: value for key, value in kargs.items() if value is not None}

    scenes = earthaccess.search_data(short_name=short_name, **kargs)
    if len(scenes) == 0:
        print('No scenes found with the specified parameters.')
        return []
    
    urls = []
    for scene in scenes:
        for link in scene.data_links():
            if re.search(subfile, os.path.basename(link)):
                urls.append(link)
    

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    outfiles = []
    for url in urls:
        filename = os.path.join(outdir, os.path.basename(url))
        if not os.path.isfile(filename) or overwrite:
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                with open(filename, 'wb') as f:
                    # iterate over response.content with a progress bar
                    with click.progressbar(length=int(response.headers.get('content-length', 0)), label=f'Downloading {os.path.basename(url)}') as bar:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                            bar.update(len(chunk))
            else:
                print(f'Failed to download {url}: {response.status_code}')
        outfiles.append(filename)

    return outfiles


@click.command()
@click.argument('outdir', type=click.Path(), required=True)
@click.argument('short_name', required=True)
@click.argument('fid', required=True)
@click.option('--subfile', default='.*', help='Regex for subfile matching')
@click.option('--version', '-v', default=None, help='Collection version')
@click.option('--overwrite', is_flag=True, help='Overwrite existing files')
def get_fid(outdir, short_name, fid, subfile, version, overwrite):
    """
    Download Earth Data granules based on search criteria.
    Args:
        outdir (str): Directory to save downloaded files.
        short_name (str): Short name of the dataset to search for.
        fid (str): FID string to filter granules.
        subfile (str): Regex pattern to filter specific subfiles from the granules.
        version (str): Collection version to filter by.
        overwrite (bool): Whether to overwrite existing files in the output directory.

    """
    fid_format = fid.upper()
    if subfile != '.*':
        if subfile[0] != '_':
            subfile = '_' + subfile
        if subfile[-1] != '_':
            subfile = subfile + '_'
    get_scene(outdir, short_name, subfile=subfile, granule_name=f'*{fid_format}*', 
              version=version, overwrite=overwrite)


