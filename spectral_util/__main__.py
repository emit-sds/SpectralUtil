#!/usr/bin/env python3
"""Centralized CLI interface for SpectralUtil."""

import click
from spectral_util.common import quicklooks
from spectral_util.mosaic import mosaic
from spectral_util.ea_assist import earthaccess_helpers_AV3, earthaccess_helpers_EMIT

@click.group()
def cli():
    pass

cli.add_command(earthaccess_helpers_AV3.find_download_and_combine, name='av3_download')
cli.add_command(earthaccess_helpers_EMIT.find_download_and_combine_EMIT, name='emit_download')
cli.add_command(mosaic.cli, name='mosaic')
cli.add_command(quicklooks.cli, name='quicklooks')

if __name__ == '__main__':
    cli()