import numpy as np
import pennylane as qml

class Problem:
    """
    General problem generator base class.
    """

    def __init__(self, seed: int) -> None:
        """
        Initialize the problem.
        """
        self.random_seed = seed

    def calc_energy(
        self,
        assignment: np.ndarray,
        single_energy_vector: np.ndarray,
        coupl_energy_matrix: np.ndarray,
    ) -> float:
        """
        Calculates the energy of a given assignment.

        Parameters:
            assignment: np.ndarray - The variable assignment.
            single_energy_vector: np.ndarray - The single-variable energy contributions; i.e., the diagonal entries of the matrix (h_i). 
            correl_energy_matrix: np.ndarray - The correlation energy matrix; i.e., the coupling coef. J_ij.

        Returns:
            float: The calculated energy.
        """
        E_calc = np.sign(assignment).T @ coupl_energy_matrix @ np.sign(
            assignment
        ) + single_energy_vector.T @ np.sign(assignment)
        return E_calc

    def matrix_to_pennylane_hamiltonian(self, matrix: np.ndarray) -> qml.Hamiltonian:
        """
        Translates a matrix to a Pennylane Hamiltonian.

        Parameters:
            matrix: np.ndarray - The matrix to be translated.

        Returns:
            qml.Hamiltonian: The translated Hamiltonian.
        """
        # initialize the Hamiltonian
        hamiltonian = qml.Hamiltonian([], [])

        # loop over the matrix elements
        for i in range(1, matrix.shape[0]):
            for j in range(1, matrix.shape[1]):
                if matrix[i, j] != 0:
                    if i == j:
                        hamiltonian += matrix[i, j] * qml.PauliZ(i - 1)
                    else:
                        hamiltonian += matrix[i, j] * qml.PauliZ(i - 1) @ qml.PauliZ(j - 1)

        return hamiltonian
     