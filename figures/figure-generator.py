import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from correlation import compute_2d_correlation
from rmse import compute_rmse
from blur import blur_matrix

def compute_max_error(matrix1, matrix2):
    """Compute the maximum absolute error between two matrices."""
    return np.max(np.abs(matrix1 - matrix2))


# Update Matplotlib parameters for LaTeX rendering
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "text.latex.preamble": r"\usepackage{amsfonts}"
})

titles = [
    'Ground truth, ',
    'Tikhonov Regularization, $m=1920$ (reg), ',
    'Tikhonov Regularization, $m=120$ (reg), ',
    'Sparcity Regularization, $m=120$ (reg), ',
    'Sparcity Regularization, $m=120$ (ran), ',
    'ML with sparcity regularization, $m=120$ (reg), '
]

frequencies = [78, 402, 1483, 3297]
cmaps = ['viridis', 'cividis', 'plasma', 'inferno']

for i in range(1, 5):
    matrix_groundtruth = np.loadtxt(f'images-data/matrix-figure-{i}-1.csv', delimiter=',')

    for j in range(1, 7):
        matrix_regularized = np.loadtxt(f'images-data/matrix-figure-{i}-{j}.csv', delimiter=',')

        # Compute the correlation between the matrices
        correlation = compute_2d_correlation(matrix_groundtruth, matrix_regularized)
        rmse = compute_rmse(matrix_groundtruth, matrix_regularized)
        max_error = compute_max_error(matrix_groundtruth, matrix_regularized)


        # Compute the blurred versions and their RMSE
        blur_groundtruth = blur_matrix(matrix_groundtruth)
        blur_regularized = blur_matrix(matrix_regularized)
        rmse_neighbors = compute_rmse(blur_groundtruth, blur_regularized)

        # Create a figure with a grid layout
        fig = plt.figure(figsize=(10, 6))
        gs = fig.add_gridspec(1, 2, width_ratios=[4, 1.5], wspace=0.1)

        # Display the matrix as an image with the specified colormap in the left grid cell
        ax_img = fig.add_subplot(gs[0, 0])
        im = ax_img.imshow(matrix_regularized, cmap=cmaps[i-1], aspect='auto')
        cbar = fig.colorbar(im, ax=ax_img)

        # Set the color bar ticks to scientific notation
        cbar.ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        cbar.ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

        ax_img.set_title(titles[j-1] + f'$\\nu = {frequencies[i-1]}$' + ' Hz')
        ax_img.set_xlabel(r'$x$ [cm]')
        ax_img.set_ylabel(r'$y$ [cm]')

        # Add text annotations in the right grid cell
        ax_text = fig.add_subplot(gs[0, 1])
        ax_text.axis('off')  # Hide the axes for the text area
        textstr = '\n'.join((
                    r'\textbf{Model Evaluation}',
                    f'$C = {correlation:.4f}$',
                    f'$\\varepsilon_1 = {rmse:.4f}$',
                    f'$\\varepsilon_2 = {rmse_neighbors:.4f}$',
                    f'$M = {max_error:.4f}$'
                ))
        ax_text.text(0.25, 0.5, textstr, transform=ax_text.transAxes, fontsize=12,
                     verticalalignment='center', horizontalalignment='center',
                     bbox=dict(facecolor='white', alpha=0.5))

        plt.tight_layout()

        plt.savefig(f'generated-images/fig-{i}-{j}.pdf', format='pdf')
