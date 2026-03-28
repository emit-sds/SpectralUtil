#!/usr/bin/env python3
"""Simple plotting routines."""

import click
import numpy as np
from spectral_util.spec_io import load_data
from spectral_util.common.quicklooks import get_rgb
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
import scipy

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


@click.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--output_file', '-o', default=None, help='Path to save the plot (if not provided, will display instead)')
@click.option('--first_pc', default=0, help='Index of the first principal component to plot')
@click.option('--last_pc', default=19, help='Index of the last principal component to plot')
@click.option('--seed', default=13, help='Random seed for PCA sampling')
@click.option('--n_points', default=10_000, help='Number of points to use for PCA')
def plot_pcs(input_file, output_file, first_pc, last_pc, seed, n_points):
    """
    Visualizes an image and consistent spectra.
    """
    # Click passes None if bands is not provided, handled in function
    pc_figure(input_file, start=first_pc, stop=last_pc, seed=seed, n_points=n_points, show=output_file is None)
    if output_file:
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        click.echo(f"Plot saved to {output_file}")


def build_pca(x):
    mu = x.mean(axis=0)
    C = np.cov(x-mu, rowvar=False)
    [v, d] = scipy.linalg.eigh(C)

    return d, mu


def do_pca(x, npca, d, mu, pca_offset=0):
    # Project, redimension as an image with "npca" channels, and segment
    if pca_offset == 0:
        pca = (x - mu) @ d[:, -npca:]
    else:
        pca = (x - mu) @ d[:, -(npca+pca_offset):-1*pca_offset]
    return pca[...,::-1]



def pc_figure(input_file, start=0, stop=99, seed=13, n_points=10_000, show=True):
    """
    Plots single pand pcs 

    Args:
        input_file (str): Path to the input spectral file.
        start (int): Starting index of the principal components to plot.
        stop (int): Ending index of the principal components to plot.
        seed (int): Random seed for sampling.
        n_points (int): Number of points to use for PCA.
        show (bool): Whether to display the plot immediately.
    """
    np.random.seed(seed)
    # Load data
    meta, data = load_data(input_file)
    build_data = data.copy().reshape((data.shape[0] * data.shape[1], data.shape[2]))
    build_data = build_data[np.random.permutation(build_data.shape[0])[:n_points],:]
    d, mu = build_pca(build_data)
    pca_out = np.zeros((data.shape[0], data.shape[1], stop-start+1))
    n_pcs = stop - start + 1
    for _line in range(data.shape[0]):
        pca_out[_line,...] = do_pca(data[_line,...], n_pcs, d, mu, pca_offset=start)
    
    plt.figure(figsize=(10, 10))
    rows = int(np.ceil(np.sqrt(n_pcs)))
    cols = int(np.ceil(n_pcs / rows))
    grid = plt.GridSpec(rows, cols, wspace=0.13, hspace=0.13)
    for i in range(n_pcs):
        plt.subplot(grid[i])
        plt.imshow(pca_out[...,i], vmin=np.percentile(pca_out[...,i], 2), vmax=np.percentile(pca_out[...,i], 98))
        plt.title(f"PC {start+i}", fontsize=8)
        plt.axis('off')
    plt.tight_layout()
    if show:
        plt.show()




@click.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--output_file', '-o', default=None, help='Path to save the plot (if not provided, will display instead)')
@click.option('--n_mnf', default=20, help='Number of MNF components to plot')
@click.option('--seed', default=13, help='Random seed for MNF sampling')
@click.option('--n_points', default=10_000, help='Number of points to use for MNF')
@click.option('--diff_dim', default=1, help='Dimension along which to calculate noise differences (0 for rows, 1 for cols)')
def plot_mnf(input_file, output_file, n_mnf, seed, n_points, diff_dim):
    """
    Visualizes an image and consistent spectra.
    """
    # Click passes None if bands is not provided, handled in function
    mnf_figure(input_file, n_mnf, seed=seed, n_points=n_points, show=output_file is None, diff_dim=diff_dim)
    if output_file:
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        click.echo(f"Plot saved to {output_file}")

def mnf_figure(input_file, n_mnf=20, seed=13, n_points=10_000, show=True, diff_dim=1):
    """
    Plots single band mnfs

    Args:
        input_file (str): Path to the input spectral file.
        n_mnf (int): Number of MNF components to plot.
        seed (int): Random seed for sampling.
        n_points (int): Number of points to use for MNF.
        show (bool): Whether to display the plot immediately.
        diff_dim (int): Dimension along which to calculate noise differences (0 for rows, 1 for cols).
    """
    np.random.seed(seed)
    # Load data
    meta, data = load_data(input_file)
    mnf_out = calculate_mnf(data, n_mnf=n_mnf, seed=seed, n_points=n_points, diff_dim=diff_dim)
    n_mnfs = mnf_out.shape[2]
    
    plt.figure(figsize=(10, 10))
    rows = int(np.ceil(np.sqrt(n_mnfs)))
    cols = int(np.ceil(n_mnfs / rows))
    grid = plt.GridSpec(rows, cols, wspace=0.13, hspace=0.13)
    for i in range(n_mnfs):
        plt.subplot(grid[i])
        plt.imshow(mnf_out[..., i], vmin=np.percentile(mnf_out[..., i], 2), vmax=np.percentile(mnf_out[..., i], 98))
        plt.title(f"MNF {i}", fontsize=8)
        plt.axis('off')
    plt.tight_layout()
    if show:
        plt.show()


def calculate_mnf(data, n_mnf=99, seed=13, n_points=10_000, diff_dim=1):
    """
    Calculates MNF components for a hyperspectral cube.

    Args:
        data (ndarray): Spectral cube with shape (rows, cols, bands).
        n_mnf (int): Number of MNF components to keep.
        seed (int): Random seed for sampling.
        n_points (int): Number of pixels used to estimate covariance terms.
        diff_dim (int): Dimension for shift-difference noise estimate (0 rows, 1 cols).
    """

    np.random.seed(seed)
    perm_subset = np.random.permutation(data.shape[0] * data.shape[1])[:n_points]
    
    build_data_raw = data.copy().reshape((data.shape[0] * data.shape[1], data.shape[2]))
    build_data = build_data_raw[perm_subset, :]
    build_data -= np.mean(build_data, axis=0)  # mean center

    # Estimate noise covariance with a shift-difference method, using either row
    # or column as specified.
    if diff_dim == 0:
        noise_diff = data[:-1, :, :] - data[1:, :, :]
        if noise_diff.shape[0] < data.shape[0]:
            noise_diff = np.pad(noise_diff, ((0, 1), (0, 0), (0, 0)), mode='edge')
    elif diff_dim == 1:
        noise_diff = data[:, :-1, :] - data[:, 1:, :]
        if noise_diff.shape[1] < data.shape[1]:
            noise_diff = np.pad(noise_diff, ((0, 0), (0, 1), (0, 0)), mode='edge')
    else:
        raise ValueError("diff_dim must be 0 (rows) or 1 (cols)")
    
    noise_diff = noise_diff.reshape((noise_diff.shape[0] * noise_diff.shape[1], noise_diff.shape[2]))
    noise_diff = noise_diff[perm_subset, :]
    
    # Multiply by 0.5 because Var(A-B) = 2*Var(Noise)
    C_N = np.cov(noise_diff, rowvar=False) * 0.5
    
    # Calculate data covariance.
    C_D = np.cov(build_data, rowvar=False)
    
    # Noise whitening transformation.
    evals_N, evecs_N = np.linalg.eigh(C_N)
    
    # Filter out tiny or negative eigenvalues to avoid division by zero
    tol = 1e-8
    valid_idx = evals_N > tol
    evals_N = evals_N[valid_idx]
    evecs_N = evecs_N[:, valid_idx]
    
    # Whitening matrix: W = V_N * Lambda_N^(-1/2)
    W = evecs_N @ np.diag(1.0 / np.sqrt(evals_N))
    
    # Transform the data covariance matrix: C_adj = W^T * C_D * W
    C_adj = W.T @ C_D @ W
    
    # Eigendecomposition of the whitened data covariance.
    evals_adj, evecs_adj = np.linalg.eigh(C_adj)
    
    # Sort descending to place highest SNR components first.
    sort_idx = np.argsort(evals_adj)[::-1]
    evals_adj = evals_adj[sort_idx]
    evecs_adj = evecs_adj[:, sort_idx]
    
    # Final transformation matrix.
    T = W @ evecs_adj
    
    # Apply the transformation.
    mnf_data_flat = build_data_raw @ T
    
    # Reshape back to the original image dimensions using retained MNF components.
    mnf_image = mnf_data_flat.reshape((data.shape[0], data.shape[1], T.shape[1]))

    # filter to top pieces
    mnf_image = mnf_image[..., :n_mnf] 

    
    return mnf_image


if __name__ == '__main__':
    plot_basic_overview()

