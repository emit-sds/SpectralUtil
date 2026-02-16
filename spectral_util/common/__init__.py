# common module initialization

import click
from .quicklooks import ndvi, rgb, nbr
from .plotting import plot_basic_overview

@click.group()
def cli_quicklook():
    pass

@click.group()
def cli_plot():
    pass

cli_quicklook.add_command(rgb)
cli_quicklook.add_command(nbr)
cli_quicklook.add_command(ndvi)
cli_plot.add_command(plot_basic_overview)
