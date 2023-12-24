import pennylane as qml

from expval_calculation.ExpVal import ExpectationValues
from pennylane import numpy as np


class StateVecQAOAExpectationValues(ExpectationValues):
    
    # initialization
    def __init__(self, problem, p, device="default.qubit", num_opts=1, num_opt_steps=50):
        super().__init__(problem)

        self.num_qubits = self.problem.matrix.shape[0] - 1
        self.wires = range(self.num_qubits)
        self.device_name = device
        self.dev = qml.device(self.device_name, wires=self.wires)

        self.mixer_h = qml.qaoa.x_mixer(self.wires)
        self.cost_h = self.problem.matrix_to_pennylane_hamiltonian(self.problem.matrix)
        self.p = p
        self.qaoa_parameters = np.pi * np.array(np.reshape(np.random.rand(2 * p), (p, 2)), requires_grad=True)

        self.num_opts = num_opts
        self.num_opt_steps = num_opt_steps

        self.type = "StateVecQAOAExpectationValues"

    def optimize(self, verbose=False):
        opt = qml.GradientDescentOptimizer()
    
        for retry in range(self.num_opts):
            parameters = self.qaoa_parameters
            if retry > 0:
                parameters = np.array(np.pi * np.reshape(np.random.rand(2 * self.p), (self.p, 2)), requires_grad=True)
            for i in range(self.num_opt_steps):
                parameters = opt.step(self._cost_function, parameters)
                if (i + 1) % 5 == 0 and verbose:
                    print("Cost after step {:5d}: {: .7f}".format(i + 1, self._cost_function(parameters)))
        
            # if verbose:
            print(f"Optimization run {retry + 1} finished. Energy {self._cost_function(parameters)}")
            if retry == 0:
                self.qaoa_parameters = parameters
            else:
                if self._cost_function(parameters) < self._cost_function(self.qaoa_parameters):
                    self.qaoa_parameters = parameters

        return self.calc_expect_val()

    def calc_expect_val(self) -> (list, int, float):

        operator_dict = self._get_expval_operator_dict()
        indices_list = list(operator_dict.keys())
        operators_list = list(operator_dict.values())

        @qml.qnode(self.dev)
        def _calc_exp_val():
            self._circuit(self.qaoa_parameters)
            return [qml.expval(op) for op in operators_list]
        
        exp_val = np.array(_calc_exp_val())
        # find max correlation, break ties at random
        max_correlation = np.max(np.abs(exp_val))
        max_correlation_indices = np.where(np.abs(exp_val) == max_correlation)[0]
        max_correlation_index = np.random.choice(max_correlation_indices)
        
        translated_mci = [self.problem.position_translater[i] for i in indices_list[max_correlation_index]]

        self.expect_val_dict = {}
        # compute energy
        self.energy = self._cost_function(self.qaoa_parameters)
    
        for i, key in enumerate(operator_dict.keys()):
            self.expect_val_dict[key] = float(exp_val[i])
    
        return translated_mci, int(np.sign(exp_val[max_correlation_index])), float(max_correlation)
        

    def _qaoa_layer(self, gamma, beta):
        qml.qaoa.cost_layer(gamma, self.cost_h)
        qml.qaoa.mixer_layer(beta, self.mixer_h)

    def _circuit(self, parameters):
        for w in self.wires:
            qml.Hadamard(wires=w)
        qml.layer(self._qaoa_layer, self.p, parameters[:, 0], parameters[:, 1])

    
    def _cost_function(self, parameters):

        @qml.qnode(self.dev)
        def __cost_function():
            self._circuit(parameters)
            return qml.expval(self.cost_h)
        
        return __cost_function()

    def _get_expval_operator_dict(self):
        """
        Returns a list of PauliZ operators for each qubit.
        """
        expval_operator_dict = {}
        for i in range(1, self.problem.matrix.shape[0]):
            for j in range(1, self.problem.matrix.shape[1]):
                if self.problem.matrix[i, j] != 0:
                    if i == j:
                        expval_operator_dict[frozenset({i})] = qml.PauliZ(i - 1) 
                    else:   
                        expval_operator_dict[frozenset({i, j})] = qml.PauliZ(i - 1) @ qml.PauliZ(j - 1)
        return expval_operator_dict