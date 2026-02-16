#!/usr/bin/env python3
"""Centralized CLI interface for SpectralUtil."""

import click
from spectral_util import common
from spectral_util.mosaic import mosaic
from spectral_util.ea_assist import earthaccess_helpers_AV3, earthaccess_helpers_EMIT
from spectral_util import ea_assist

@click.group()
def cli():
    pass

cli.add_command(ea_assist.cli, name='download')
cli.add_command(mosaic.cli, name='mosaic')
cli.add_command(common.cli_quicklook, name='quicklooks')
cli.add_command(common.cli_plot, name='plot')

if __name__ == '__main__':
    cli()