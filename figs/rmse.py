import numpy as np

def compute_rmse(matrix1, matrix2):
    """
    Compute the Root Mean Squared Error (RMSE) between two matrices.

    Parameters:
    matrix1 (np.ndarray): First matrix.
    matrix2 (np.ndarray): Second matrix.

    Returns:
    float: The RMSE between the two matrices.
    """
    mse = np.mean((matrix1 - matrix2) ** 2)
    rmse = np.sqrt(mse)
    return rmse