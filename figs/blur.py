import numpy as np
from scipy.ndimage import convolve

def blur_matrix(matrix):
    """
    Apply a blur effect to the input matrix by averaging each pixel with its 8 neighbors.

    Parameters:
    matrix (np.ndarray): The input matrix to be blurred.

    Returns:
    np.ndarray: The blurred matrix.
    """
    # Define the blurring kernel
    kernel = np.ones((3, 3)) / 9.0

    # Apply the convolution to blur the matrix
    blurred_matrix = convolve(matrix, kernel, mode='constant', cval=0.0)

    return blurred_matrix
