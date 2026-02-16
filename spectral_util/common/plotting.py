#!/usr/bin/env python3
"""Simple plotting routines."""

import click
import numpy as np
from spectral_util.spec_io import load_data
from spectral_util.common.quicklooks import get_rgb
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans

def plot_spectra(input_file, n_points=10, method='even', seed=13):
    """
    Plots an RGB image on the left and selected spectra on the right.

    Args:
        input_file (str): Path to the input spectral file.
        n_points (int): Number of spectra to plot.
        method (str): 'random' or 'kmeans'.
    """
    # Load data
    meta, data = load_data(input_file)
    rgb = get_rgb(data, meta) 

    # Select Spectra
    spectra_to_plot = []
    labels = []
    
    # Remove nodata/NaNs for clustering/sampling if possible
    # Simple valid mask:
    valid_mask = np.logical_not(np.all(rgb == 0, axis=-1))
    if np.sum(valid_mask) == 0:
        print("No valid data found.")
        return
        
    np.random.seed(seed)
    indices = None 
    if method == 'kmeans':
        kmeans = MiniBatchKMeans(n_clusters=n_points, n_init=3, batch_size=1024).fit(np.array(data)[valid_mask,:])
        spectra_to_plot = kmeans.cluster_centers_
        labels = [f"Cluster {i}" for i in range(n_points)]
    elif method == 'random':
        # Random
        indices = np.where(valid_mask)
        perm = np.random.permutation(len(indices[0]))[:n_points]
        indices = tuple(idx[perm] for idx in indices)

        # inefficient to cast this, but the netcdf subsetting is odd, should re-examine
        spectra_to_plot = np.array(data)[indices[0],indices[1],:]
        labels = [f"Point {i}" for i in range(n_points)]
    elif method == 'even':
        indices = [np.linspace(0, data.shape[0], n_points+2, dtype=int)[1:-1],
                   np.linspace(0, data.shape[0], n_points+2, dtype=int)[1:-1]]
        spectra_to_plot = np.array(data)[indices[0],indices[1],:]
        labels = [f"Point {i}" for i in range(n_points)]

    # Plotting
    cmap = plt.get_cmap('Dark2')
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: RGB
    axes[0].imshow(rgb)
    axes[0].set_title(f"RGB Preview")
    if indices is not None:
        for _i, (y, x) in enumerate(zip(indices[0], indices[1])):
            axes[0].plot(x, y, 'o', markerfacecolor='none', markeredgecolor=cmap(_i % cmap.N), 
                         markersize=10, markeredgewidth=2, label='Selected Points')
    axes[0].axis('off')

    # Right: Spectra
    axes[1].set_xlabel("Wavelength")

    wl_nan = meta.wavelengths.copy()
    print(spectra_to_plot[0,:])
    wl_nan[np.isclose(spectra_to_plot[0,:], -0.01, atol=1e-5)] = np.nan
    for i, spec in enumerate(spectra_to_plot):
        axes[1].plot(wl_nan, spec, label=labels[i], c=cmap(i % cmap.N))
    
    axes[1].set_title(f"{method.capitalize()} Spectra")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

@click.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--output_file', '-o', default=None, help='Path to save the plot (if not provided, will display instead)')
@click.option('--n_points', '-n', default=5, help='Number of spectra to plot')
@click.option('--method', '-m', type=click.Choice(['random', 'kmeans', 'even']), default='even', help='Selection method')
def plot_basic_overview(input_file, output_file, n_points, method):
    """
    Visualizes an image and consistent spectra.
    """
    # Click passes None if bands is not provided, handled in function
    plot_spectra(input_file, n_points, method)
    if output_file:
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        click.echo(f"Plot saved to {output_file}")

if __name__ == '__main__':
    plot_basic_overview()

