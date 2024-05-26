import numpy as np
from scipy.signal import correlate2d

def compute_2d_correlation(matrix1, matrix2):
    """
    Compute the normalized 2D correlation between two matrices.

    Parameters:
    matrix1 (ndarray): First input matrix.
    matrix2 (ndarray): Second input matrix.

    Returns:
    float: Normalized 2D correlation coefficient.
    """
    # Ensure the matrices have the same shape
    if matrix1.shape != matrix2.shape:
        raise ValueError("Input matrices must have the same shape")

    # Compute the 2D correlation
    correlation = correlate2d(matrix1, matrix2, mode='valid')

    # Normalize the correlation
    norm_correlation = correlation / np.sqrt(np.sum(matrix1**2) * np.sum(matrix2**2))

    return norm_correlation[0, 0]
