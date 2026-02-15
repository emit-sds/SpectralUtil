# ea_assist module initialization
import click
from spectral_util import ea_assist
from spectral_util.ea_assist import download, earthaccess_helpers_AV3, earthaccess_helpers_EMIT

@click.group()
def cli():
    pass

cli.add_command(ea_assist.download.get_fid, name='get-fid')
cli.add_command(ea_assist.earthaccess_helpers_AV3.find_download_and_combine, name='av3-download')
cli.add_command(ea_assist.earthaccess_helpers_EMIT.find_download_and_combine_EMIT, name='emit-download')