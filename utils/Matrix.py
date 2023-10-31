from typing import Union
import numpy as np

class Matrix:
    """Represents a square matrix for Hamiltonian calculations."""
    
    def __init__(self, size: int) -> None:
        """
        Initialize a square matrix of the given size.
        
        Parameters:
            size: int - The number of rows (and columns) in the matrix.
        """
        if size <= 0:
            raise ValueError("Matrix size should be a positive integer.")
        self.size = size
        self.matrix = np.zeros((size, size))

    def add_off_element(self, i: int, j: int, const: Union[int, float]) -> None:
        """
        Adds an off-diagonal element to the matrix.
        
        Parameters:
            i: int - The row index.
            j: int - The column index.
            const: int or float - The constant to be added.
        """
        row, col = (np.abs(i), np.abs(j)) if np.abs(i) >= np.abs(j) else (np.abs(j), np.abs(i))
        self.matrix[row, col] += np.sign(i) * np.sign(j) * const

    def add_diag_element(self, i: int, const: Union[int, float]) -> None:
        """
        Adds a diagonal element to the matrix.
        
        Parameters:
            i: int - The index of the diagonal element.
            const: int or float - The constant to be added.
        """
        self.matrix[np.abs(i), np.abs(i)] += np.sign(i) * const