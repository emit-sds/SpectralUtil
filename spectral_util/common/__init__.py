# common module initialization

import click
from .quicklooks import ndvi, rgb, nbr
from .plotting import plot_basic_overview, plot_pcs
from .extract import extract_polygon_data, extract_polygon_stats, extract_polygon_to_geojson

@click.group()
def cli_quicklook():
    pass

@click.group()
def cli_plot():
    pass

@click.group()
def cli_extract():
    pass

cli_quicklook.add_command(rgb)
cli_quicklook.add_command(nbr)
cli_quicklook.add_command(ndvi)
cli_plot.add_command(plot_basic_overview)
cli_plot.add_command(plot_pcs)
# Extract commands are functions, not Click commands, so we don't add them here
# They are available as module-level functions via spectral_util.common.extract
