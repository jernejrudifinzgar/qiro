import timeit
import numpy as np
import itertools as it
import copy
from scipy.optimize import fsolve
import qtensor
import json
import torch
import random

class ExpectationValues():
    """General expectation value calculator"""
    def __init__(self, problem):
        self.problem = problem
        self.energy = None
        self.best_energy = None
        self.fixed_correl = []
        self.expect_val_dict = {}

class SingleLayerQAOAExpectationValues(ExpectationValues):
    """
    :param problem: input problem
    this class is responsible for the whole RQAOA procedure
    """
    def __init__(self, problem, gamma=[0], beta=[0]):
        ExpectationValues.__init__(self, problem)
        self.a = None
        self.b = None
        self.c = None
        self.gamma = gamma[0]*np.pi
        self.beta = beta[0]*np.pi
        self.losses = None
        self.type = 'SingleLayerQAOAExpectationValue'

    """the single_cos und coupling_cos functions are sub-functions which are called in the calculation of the
    expectation values"""
    def single_cos(self, i, gamma):
        """sub-function"""
        a = 1
        vec_i = self.problem.matrix[i, 1:i]
        vec_i = np.append(vec_i, self.problem.matrix[i+1:, i])
        vec_i = vec_i[vec_i != 0]
        vec = np.cos(2 * gamma * (vec_i))
        a = np.prod(vec)
        return a

    def coupling_cos_0(self, i, j, gamma):
        """sub-function, careful it's not symmetric, the first index tells us the row of the matrix"""
        a = 1
        index_small = np.min((i, j))
        index_large = np.max((i, j))

        vec_i = self.problem.matrix[i, 1:index_small]
        if index_small == i:
            vec_i = np.append(vec_i, self.problem.matrix[index_small + 1:index_large, i])
        else:
            vec_i = np.append(vec_i, self.problem.matrix[i, index_small + 1:index_large])
        vec_i = np.append(vec_i, self.problem.matrix[index_large + 1:, i])

        vec_zeros = vec_i
        vec_non_zeros = vec_zeros[vec_zeros != 0]
        vec = np.cos(2 * gamma * (vec_non_zeros))
        a = np.prod(vec)

        return a

    def coupling_cos_plus(self, i, j, gamma):
        """sub-function"""
        index_small = np.min((i, j))
        index_large = np.max((i, j))

        vec_i = self.problem.matrix[index_small, 1:index_small]
        vec_i = np.append(vec_i, self.problem.matrix[index_small + 1:index_large, index_small])
        vec_i = np.append(vec_i, self.problem.matrix[index_large + 1:, index_small])

        vec_j = self.problem.matrix[index_large, 1:index_small]
        vec_j = np.append(vec_j, self.problem.matrix[index_large, index_small + 1:index_large])
        vec_j = np.append(vec_j, self.problem.matrix[index_large + 1:, index_large])
        vec_zeros = vec_i + vec_j
        vec_non_zeros = vec_zeros[vec_zeros != 0]

        vec = np.cos(2 * gamma * (vec_non_zeros))
        a = np.prod(vec)

        return a

    def coupling_cos_minus(self, index_large, index_small, gamma):
        """sub-function"""
        vec_i = self.problem.matrix[index_small, 1:index_small]
        vec_i = np.append(vec_i, self.problem.matrix[index_small + 1:index_large, index_small])
        vec_i = np.append(vec_i, self.problem.matrix[index_large + 1:, index_small])

        vec_j = self.problem.matrix[index_large, 1:index_small]
        vec_j = np.append(vec_j, self.problem.matrix[index_large, index_small + 1:index_large])
        vec_j = np.append(vec_j, self.problem.matrix[index_large + 1:, index_large])
        vec_zeros = vec_i - vec_j
        vec_non_zeros = vec_zeros[vec_zeros != 0]

        vec = np.cos(2 * gamma * (vec_non_zeros))
        a = np.prod(vec)
        return a

    def calc_single_terms(self, gamma, index):
        """This is just a help function to compute lengthy terms of the two-point expectation values and stores them as
        constants"""
        a_part_term = np.sin(2 * gamma * self.problem.matrix[index, index]) * self.single_cos(index, gamma)
        return a_part_term

    def calc_coupling_terms(self, gamma, index_large, index_small):
        """This is just a help function to compute lengthy terms of the two-point expectation values and stores them as
        constants"""

        b_part_term = (1/2) * np.sin(2 * gamma * self.problem.matrix[index_large, index_small]) * \
                    (
                            np.cos(2 * gamma * self.problem.matrix[index_large, index_large]) * self.coupling_cos_0(index_large, index_small, gamma)
                                + \
                            np.cos(2 * gamma * self.problem.matrix[index_small, index_small]) * self.coupling_cos_0(index_small, index_large, gamma)
                    )
        c_0 = 1/2
        c_1 = np.cos(2 * gamma * (self.problem.matrix[index_large, index_large] + self.problem.matrix[index_small, index_small])) * self.coupling_cos_plus(index_large, index_small, gamma)
        c_2 = np.cos(2 * gamma * (self.problem.matrix[index_large, index_large] - self.problem.matrix[index_small, index_small])) * self.coupling_cos_minus(index_large, index_small, gamma)
        c_part_term = c_0 * (c_1 - c_2)

        return b_part_term, c_part_term

    def calc_const(self, gamma):
        """Calculates the constant terms for the step in which optimal beta is being calculated"""
        a = 0
        b = 0
        c = 0

        for index in range(1, len(self.problem.matrix)):
            a_term = self.problem.matrix[index, index] * self.calc_single_terms(gamma, index)
            a = a + a_term

        timi = 0
        count = 0
        for index_large in range(1, len(self.problem.matrix)):
            for index_small in range(1, index_large):

                if self.problem.matrix[index_large, index_small] != 0:
                    start = timeit.default_timer()
                    b_part_term, c_part_term = self.calc_coupling_terms(gamma, index_large, index_small)
                    stop = timeit.default_timer()
                    timi += stop - start
                    b_term = self.problem.matrix[index_large, index_small] * b_part_term
                    c_term = self.problem.matrix[index_large, index_small] * c_part_term
                    b = b + b_term
                    c = c + c_term
                    count += 1
        self.a = a
        self.b = b
        self.c = c

    def calc_expect_val(self):
        """Calculate all one- and two-point correlation expectation values and return the one with highest absolute value."""
        self.expect_val_dict = {}
        Z = np.sin(2 * self.beta) * self.calc_single_terms(gamma=self.gamma, index=1)
        if np.abs(Z) > 0:
            rounding_list = [[[self.problem.position_translater[1]], np.sign(Z), np.abs(Z)]]
            max_expect_val = np.abs(Z)
        else:
            rounding_list = [[[self.problem.position_translater[1]], 1, 0]]
            max_expect_val = 0
        
        self.expect_val_dict[frozenset({1})] = Z

        for index in range(2, len(self.problem.matrix)):
            Z = np.sin(2 * self.beta) * self.calc_single_terms(gamma=self.gamma, index=index)
            self.expect_val_dict[frozenset({index})] = Z
            if np.abs(Z) > max_expect_val:
                rounding_list = [[[self.problem.position_translater[index]], np.sign(Z), np.abs(Z)]]
                max_expect_val = np.abs(Z)
            elif np.abs(Z) == max_expect_val:
                rounding_list.append([[self.problem.position_translater[index]], np.sign(Z), np.abs(Z)])

        for index_large in range(1, len(self.problem.matrix)):
            for index_small in range(1, index_large):
                if self.problem.matrix[index_large, index_small] != 0:
                    b_part_term, c_part_term = self.calc_coupling_terms(gamma=self.gamma, index_large=index_large, index_small=index_small)
                    ZZ = np.sin(4 * self.beta) * b_part_term - ((np.sin(2 * self.beta)) ** 2) * c_part_term
                    self.expect_val_dict[frozenset({index_large, index_small})] = ZZ
                    if np.abs(ZZ) > max_expect_val:
                        rounding_list = [[[self.problem.position_translater[index_large], self.problem.position_translater[index_small]], np.sign(ZZ), np.abs(ZZ)]]
                        max_expect_val = np.abs(ZZ)
                    elif np.abs(ZZ) == max_expect_val:
                        rounding_list.append([[self.problem.position_translater[index_large], self.problem.position_translater[index_small]], np.sign(ZZ), np.abs(ZZ)])

        # random tie-breaking:
        random_index = np.random.randint(len(rounding_list))
        rounding_element = rounding_list[random_index]
        max_expect_val_location = rounding_element[0]
        max_expect_val_sign = rounding_element[1]
        max_expect_val = rounding_element[2]
        return max_expect_val_location, max_expect_val_sign, max_expect_val

    def calc_beta_energy(self, gamma):
        """Calculate the optimal value of beta regarding the energy, dependent on the input gamma"""
        self.calc_const(gamma)

        def f(x):
            """energy derivative"""
            return 2 * self.a * np.cos(2 * x) + 4 * self.b * np.cos(4 * x) - 4 * self.c * np.sin(2 * x) * np.cos(2 * x)

        beta = float(fsolve(f, 0.01, xtol=0.000001))
        energy = self.a * np.sin(2 * beta) + self.b * np.sin(4 * beta) - self.c * ((np.sin(2 * beta)) ** 2)

        # running solver for calculating the root of the derivative of the energy function
        for i in range(10):
            start_point = (np.pi / 10) * (i + 1)
            beta_try = float(fsolve(f, start_point, xtol=0.000001))
            energy_try = self.a * np.sin(2 * beta_try) + self.b * np.sin(4 * beta_try) - self.c * ((np.sin(2 * beta_try)) ** 2)
            if energy_try < energy:
                energy = energy_try
                beta = beta_try

        # this is basically a backup code paragraph which does a rough grid search for the beta parameters in case
        # our solver gets stuck in a local minima then this value of beta will be used
        beta_grid = 0
        energy_grid = self.a * np.sin(2 * beta_grid) + self.b * np.sin(4 * beta_grid) - self.c * ((np.sin(2 * beta_grid)) ** 2)
        for beta_try in np.linspace(0, np.pi, 50):
            energy_try = self.a * np.sin(2 * beta_try) + self.b * np.sin(4 * beta_try) - self.c * ((np.sin(2 * beta_try)) ** 2)
            if energy_try < energy_grid:
                energy_grid = energy_try
                beta_grid = beta_try

        if energy > energy_grid:
            energy = energy_grid
            beta = beta_grid

        return beta, energy

    def calc_best_gamma(self, lb=0, ub=np.pi, steps=30):
        """Calculates best angles in a 30 points grid between a lower and upper bound"""
        for gamma in np.linspace(lb, ub, steps):
            beta, energy = self.calc_beta_energy(gamma)

            if energy < self.energy:
                self.gamma = gamma
                self.beta = beta
                self.energy = energy

    def optimize(self):
        self.gamma = 0
        self.beta, self.energy = self.calc_beta_energy(self.gamma)

        # rough grid search
        steps = 30
        self.calc_best_gamma(lb=0, ub=np.pi, steps=steps)

        # refined grid search
        lb = self.gamma - (np.pi / (steps - 1))
        ub = self.gamma + (np.pi / (steps - 1))
        self.calc_best_gamma(lb=lb, ub=ub, steps=steps)

        max_expect_val_location, max_expect_val_sign, max_expect_val = self.calc_expect_val()
        self.fixed_correl.append([max_expect_val_location, max_expect_val_sign, max_expect_val])
        
        return max_expect_val_location, max_expect_val_sign, max_expect_val

    def brute_force(self):
        """calculate optimal solution of the remaining variables (according to the remaining
        optimization problem) brute force"""
        x_in_dict = {}
        brute_forced_solution = {}
        count = 0
        single_energy_vector = copy.deepcopy(self.problem.matrix.diagonal())
        correl_energy_matrix = copy.deepcopy(self.problem.matrix)
        np.fill_diagonal(correl_energy_matrix, 0)
        
        for iter_var_list in it.product([-1, 1], repeat=(len(self.problem.position_translater)-1)):
            vec = np.array([0])
            vec = np.append(vec, iter_var_list)
            E_current = self.problem.calc_energy(vec, single_energy_vector, correl_energy_matrix)

            for i in range(1, len(vec)):
                x_in_dict[self.problem.position_translater[i]] = iter_var_list[i-1]
            if count == 0:
                E_best = copy.deepcopy(E_current)
                brute_forced_solution = copy.deepcopy(x_in_dict)
                count += 1
            if float(E_current) < float(E_best):
                brute_forced_solution = copy.deepcopy(x_in_dict)
                E_best = copy.deepcopy(E_current)
        return brute_forced_solution
    

# TODO Generalise to arbitrary QUBO by introducing matrix 
class QtensorQAOAExpectationValuesMIS(ExpectationValues):
    """Calculation of expectation values via tensor network contraction using Qtensor"""

    def __init__(self, problem, p, pbar=True, gamma=None, beta=None, backend=qtensor.contraction_backends.TorchBackend(), ordering_algo='greedy'):
        super().__init__(problem)
        random.seed()

        if gamma == None:
            #gamma=[0.1] * p
            gamma=[random.uniform(0, 0.5)]*p

        if beta == None:
            #beta=[0.1] * p
            beta=[random.uniform(0, 0.5)]*p

        self.pbar=pbar
        self.p = p
        self.backend = backend
        self.alpha = self.problem.alpha
        self.graph = self.problem.graph
        self.type = 'QtensorQAOAExpectationValuesMIS'
        #if self.backend == qtensor.contraction_backends.TorchBackend():
    
        self.alpha = torch.tensor(self.alpha, requires_grad=False)
        self.gamma, self.beta = torch.tensor(gamma, requires_grad=True), torch.tensor(beta, requires_grad=True)
        #else:
        #    self.gamma = gamma
        #    self.beta = beta
        self.p = len(self.gamma)
        self.loss = None
        self.ordering_algo = ordering_algo
        self.peos = self.energy_peo()
        self.E_nodes = None
        self.E_edges = None

    def energy_loss(self):
        sim = qtensor.QtreeSimulator(backend=self.backend)
        composer = qtensor.TorchQAOAComposer_MIS(self.graph, alpha=self.alpha, gamma=self.gamma, beta = self.beta)
        self.loss = torch.tensor([0.])
        self.E_nodes = {}
        self.E_edges ={}

        #if peos is None:
        #    peos = [None] *(self.graph.number_of_edges()+self.graph.number_of_nodes())

        #peos_nodes = self.peos[:self.graph.number_of_nodes()]
        #peos_edges = self.peos[self.graph.number_of_nodes():]

        for node in self.graph.nodes():
            peo = self.peos[node]
            #print(node)
            composer.energy_expectation_lightcone_node(node)
            E = torch.real(sim.simulate_batch(composer.circuit, peo=peo))
            composer.builder.reset()
            self.loss -= 0.5 * E
            self.E_nodes[node]=E 
            #self.E_nodes.append(E)

        for edge in self.graph.edges():
            #print(edge)
            peo = self.peos[edge]
            composer.energy_expectation_lightcone(edge)
            E = torch.real(sim.simulate_batch(composer.circuit, peo=peo))
            composer.builder.reset()
            self.E_edges[edge]= E
            self.loss += self.alpha * 0.25 * (E + self.E_nodes[edge[0]] + self.E_nodes[edge[1]])

    #@lru_cache
    def energy_peo(self):
        opt = qtensor.toolbox.get_ordering_algo(self.ordering_algo)
        peos = {}
        for node in self.graph.nodes():
            composer = qtensor.TorchQAOAComposer_MIS(self.graph, alpha = self.alpha, gamma = self.gamma, beta = self.beta)
            composer.energy_expectation_lightcone_node(node)
            tn = qtensor.optimisation.TensorNet.QtreeTensorNet.from_qtree_gates(composer.circuit)
            peo, _ = opt.optimize(tn)
            peos[node] = peo 

        for edge in self.graph.edges():
            composer = qtensor.TorchQAOAComposer_MIS(self.graph, alpha = self.alpha, gamma = self.gamma, beta = self.beta)
            composer.energy_expectation_lightcone(edge)
            tn = qtensor.optimisation.TensorNet.QtreeTensorNet.from_qtree_gates(composer.circuit)
            peo, _ = opt.optimize(tn)
            peos[edge] = peo

        return peos
    
    def calc_expect_val(self):
        self.energy_loss()
        max_expect_val = 0

        for node in self.graph.nodes():
            #+1 introduced to be comparable with SingleLayerQAOAExpectationValues:
            self.expect_val_dict[frozenset({self.problem.position_translater.index(node+1)})]=float(self.E_nodes[node])
            #self.expect_val_dict[frozenset({node+1})]=float(self.E_nodes[node])
            if abs(float(self.E_nodes[node])) > max_expect_val:
                max_expect_val = abs(float(self.E_nodes[node]))
                max_expect_val_sign = np.sign(float(self.E_nodes[node]))
                max_expect_val_location = node
                
        for edge in self.graph.edges():
            self.expect_val_dict[frozenset({self.problem.position_translater.index(max(edge)+1), self.problem.position_translater.index(min(edge)+1)})]=float(self.E_edges[edge])
            #self.expect_val_dict[frozenset({edge[0]+1, edge[1]+1})]=float(self.E_edges[edge])
            if abs(float(self.E_edges[edge])) > max_expect_val:
                max_expect_val = abs(float(self.E_edges[edge]))
                max_expect_val_sign = np.sign(float(self.E_edges[edge]))
                max_expect_val_location = edge 

        return max_expect_val_location, max_expect_val_sign, max_expect_val
    
    def optimize(self, steps=50, Opt = torch.optim.RMSprop, opt_kwargs=dict(), **kwargs):
        random.seed()
        opt = Opt(params=(self.gamma, self.beta) , **opt_kwargs)
        self.peos = self.energy_peo()
        #print(self.graph)
        self.expect_val_dict = {}
        max_expect_val_location = None
        max_expect_val_sign = None
        max_expect_val = None
        self.losses = []
        self.steps = steps
        self.param_history = []
        #ExpectationValues.__init__(self, self.problem)

        self.param_history.append([x.detach().numpy().copy() for x in (self.gamma, self.beta)])
        
        if self.pbar:
            from tqdm.auto import tqdm
            _pbar = tqdm(total=self.steps)
        else:
            _pbar = None

        for i in range(self.steps):
            self.energy_loss()

            opt.zero_grad()
            self.loss.backward()
            opt.step()

            self.losses.append(self.loss.detach().numpy().data)
            self.param_history.append([x.detach().numpy().copy() for x in (self.gamma, self.beta)])
            if self.pbar:
                _pbar.update(1)
        
        max_expect_val = 0

        for node in self.graph.nodes():
            #+1 introduced to be comparable with SingleLayerQAOAExpectationValues:
            self.expect_val_dict[frozenset({self.problem.position_translater.index(node+1)})]=float(self.E_nodes[node])
            #self.expect_val_dict[frozenset({node+1})]=float(self.E_nodes[node])
            if abs(float(self.E_nodes[node])) > max_expect_val:
                max_expect_val = abs(float(self.E_nodes[node]))
                max_expect_val_sign = np.sign(float(self.E_nodes[node]))
                max_expect_val_location = node
                
        for edge in self.graph.edges():
            self.expect_val_dict[frozenset({self.problem.position_translater.index(max(edge)+1), self.problem.position_translater.index(min(edge)+1)})]=float(self.E_edges[edge])
            #self.expect_val_dict[frozenset({edge[0]+1, edge[1]+1})]=float(self.E_edges[edge])
            if abs(float(self.E_edges[edge])) > max_expect_val:
                max_expect_val = abs(float(self.E_edges[edge]))
                max_expect_val_location = edge 

        self.energy = self.loss

        #does the same thing as above but more complicated
        """ for i in self.E_nodes.values():
            if abs(float(i)) > max_expect_val:
                max_expect_val=abs(float(i))
                max_expect_val_sign=np.sign(float(i))
                for key, value in self.expect_val_dict.items():
                    if value == float(i):
                        max_expect_val_location = key
                    
        for i in self.E_edges:
            if abs(float(i)) > max_expect_val:
                max_expect_val=abs(float(i))
                max_expect_val_sign=np.sign(float(i))
                for key, value in self.expect_val_dict.items():
                    if value == float(i):
                        max_expect_val_location = key """

        return max_expect_val_location, max_expect_val_sign, max_expect_val
    


##############################################################################



class QtensorQAOAExpectationValuesMAXCUT(ExpectationValues):
    """Calculation of expectation values via tensor network contraction using Qtensor"""

    def __init__(self, problem, p, pbar=True, gamma=None, beta=None, backend=qtensor.contraction_backends.TorchBackend(), ordering_algo='greedy'):
        super().__init__(problem)
        random.seed()

        if gamma == None:
            #gamma=[0.1] * p
            gamma=[random.uniform(0, 0.5)]*p

        if beta == None:
            #beta=[0.1] * p
            beta=[random.uniform(0, 0.5)]*p

        self.pbar=pbar
        self.p = p
        self.backend = backend
        self.graph = self.problem.graph
        self.losses =[]
        self.type = 'QtensorQAOAExpectationValuesMAXCUT'
        #if self.backend == qtensor.contraction_backends.TorchBackend():
    
        self.gamma, self.beta = torch.tensor(gamma, requires_grad=True), torch.tensor(beta, requires_grad=True)
        #else:
        #    self.gamma = gamma
        #    self.beta = beta
        self.p = len(self.gamma)
        self.loss = None
        self.ordering_algo = ordering_algo
        self.peos = self.energy_peo()
        self.E_nodes = None
        self.E_edges = None

    def energy_loss(self):
        sim = qtensor.QtreeSimulator(backend=self.backend)
        composer = qtensor.TorchQAOAComposer_MAXCUT(self.problem.graph, gamma=self.gamma, beta = self.beta)
        self.loss = torch.tensor([0.])
        self.E_nodes = {}
        self.E_edges ={}

        #if peos is None:
        #    peos = [None] *(self.graph.number_of_edges()+self.graph.number_of_nodes())

        #peos_nodes = self.peos[:self.graph.number_of_nodes()]
        #peos_edges = self.peos[self.graph.number_of_nodes():]

        for edge in self.problem.graph.edges():
            #print(edge)
            peo = self.peos[edge]
            composer.energy_expectation_lightcone(edge)
            E = torch.real(sim.simulate_batch(composer.circuit, peo=peo))
            composer.builder.reset()
            self.E_edges[edge]= E
            self.loss += E 

    #@lru_cache
    def energy_peo(self):
        opt = qtensor.toolbox.get_ordering_algo(self.ordering_algo)
        peos = {}
        
        for edge in self.problem.graph.edges():
            composer = qtensor.TorchQAOAComposer_MAXCUT(self.problem.graph, gamma = self.gamma, beta = self.beta)
            composer.energy_expectation_lightcone(edge)
            tn = qtensor.optimisation.TensorNet.QtreeTensorNet.from_qtree_gates(composer.circuit)
            peo, _ = opt.optimize(tn)
            peos[edge] = peo

        return peos
    
    def calc_expect_val(self):
        self.energy_loss()
        max_expect_val = 0
                
        for edge in self.problem.graph.edges():
            self.expect_val_dict[frozenset({self.problem.position_translater.index(max(edge)+1), self.problem.position_translater.index(min(edge)+1)})]=float(self.E_edges[edge])
            #self.expect_val_dict[frozenset({edge[0]+1, edge[1]+1})]=float(self.E_edges[edge])
            if abs(float(self.E_edges[edge])) > max_expect_val:
                max_expect_val = abs(float(self.E_edges[edge]))
                max_expect_val_sign = np.sign(float(self.E_edges[edge]))
                max_expect_val_location = [self.problem.position_translater.index(max(edge)+1), self.problem.position_translater.index(min(edge)+1)]

        return max_expect_val_location, max_expect_val_sign, max_expect_val
    
    def optimize(self, steps=50, Opt = torch.optim.RMSprop, opt_kwargs=dict(), **kwargs):
        random.seed()
        opt = Opt(params=(self.gamma, self.beta) , **kwargs)
        self.peos = self.energy_peo()
        #print(self.graph)
        self.expect_val_dict = {}
        max_expect_val_location = None
        max_expect_val_sign = None
        max_expect_val = None
        self.losses = []
        self.steps = steps
        self.param_history = []
        #ExpectationValues.__init__(self, self.problem)

        self.param_history.append([x.detach().numpy().copy() for x in (self.gamma, self.beta)])
        
        if self.pbar:
            from tqdm.auto import tqdm
            _pbar = tqdm(total=self.steps)
        else:
            _pbar = None

        counter = 0
        for i in range(self.steps):
            self.energy_loss()

            opt.zero_grad()
            self.loss.backward()
            opt.step()

            #self.losses.append(self.loss.detach().numpy().data)
            self.losses.append(float(self.loss))
            self.param_history.append([x.detach().numpy().copy() for x in (self.gamma, self.beta)])
            
            # if i > 1:
            #     if abs((self.losses[-1]-self.losses[-2])/self.losses[-1])<0.001:
            #         counter += 1
            #         if counter == 5: 
            #             break
            #     else: 
            #         counter = 0


            if self.pbar:
                _pbar.update(1)
        
        max_expect_val = 0
        print(self.problem.graph.nodes())
        print(self.problem.graph.edges())
        for edge in self.problem.graph.edges():
            self.expect_val_dict[frozenset({self.problem.position_translater.index(max(edge)+1), self.problem.position_translater.index(min(edge)+1)})]=float(self.E_edges[edge])
            #self.expect_val_dict[frozenset({edge[0]+1, edge[1]+1})]=float(self.E_edges[edge])
            if abs(float(self.E_edges[edge])) > max_expect_val:
                max_expect_val = abs(float(self.E_edges[edge]))
                max_expect_val_sign = np.sign(float(self.E_edges[edge]))
                max_expect_val_location = (edge[0]+1, edge[1]+1) 

        self.energy = self.loss

        #does the same thing as above but more complicated
        """ for i in self.E_nodes.values():
            if abs(float(i)) > max_expect_val:
                max_expect_val=abs(float(i))
                max_expect_val_sign=np.sign(float(i))
                for key, value in self.expect_val_dict.items():
                    if value == float(i):
                        max_expect_val_location = key
                    
        for i in self.E_edges:
            if abs(float(i)) > max_expect_val:
                max_expect_val=abs(float(i))
                max_expect_val_sign=np.sign(float(i))
                for key, value in self.expect_val_dict.items():
                    if value == float(i):
                        max_expect_val_location = key """

        return max_expect_val_location, max_expect_val_sign, max_expect_val
    

###################################################################################

class QtensorQAOAExpectationValuesQUBO(ExpectationValues):
    """Calculation of expectation values via tensor network contraction using Qtensor"""

    def __init__(self, problem, p, pbar=True, gamma=None, beta=None, initialization='random', regularity=3, opt=torch.optim.RMSprop, opt_kwargs=dict(lr=0.001), backend=qtensor.contraction_backends.TorchBackend(), ordering_algo='greedy'):
        super().__init__(problem)
        random.seed()
        #TODO check fixed angles parameters for non-regular graphs
        if initialization=='fixed_angles_optimization':
            with open('angles_regular_graphs.json', 'r') as file:
                data = json.load(file)

            gamma, beta = data[f"{regularity}"][f"{p}"]["gamma"], data[f"{regularity}"][f"{p}"]["beta"]
            gamma, beta = [value/(-2*np.pi) for value in gamma], [value/(2*np.pi) for value in beta]
            print('is working')

        else:
            if gamma == None:
                #gamma=[0.1] * p
                gamma=[random.uniform(0, 0.5)]*p

            if beta == None:
                #beta=[0.1] * p
                beta=[random.uniform(0, 0.5)]*p

        self.opt = opt
        self.opt_kwargs = opt_kwargs
        self.pbar=pbar
        self.p = p
        self.backend = backend
        self.initialization = initialization
        self.type = 'QtensorQAOAExpectationValuesQUBO'
        #if self.backend == qtensor.contraction_backends.TorchBackend():
    
        self.gamma, self.beta = torch.tensor(gamma, requires_grad=True), torch.tensor(beta, requires_grad=True)
        self.loss = None
        self.ordering_algo = ordering_algo
        #Why should peos be calculated already at initialization??:
        #self.peos = self.energy_peo()
        self.E_nodes = None

    #TODO Move to optimization part and check which correlations are necessary  
    #@lru_cache
    def energy_peo(self):
        opt = qtensor.toolbox.get_ordering_algo(self.ordering_algo)
        peos = {}
        for i in range(1, len(self.problem.matrix)):
            for j in range(1, i+1):
                #if self.problem.matrix[i, j] != 0:
                
                composer = qtensor.TorchQAOAComposer_QUBO(self.problem.graph, self.problem.matrix, self.problem.position_translater, gamma = self.gamma, beta = self.beta)
                if i == j:
                    composer.energy_expectation_lightcone([self.problem.position_translater[i]-1])

                else:
                    composer.energy_expectation_lightcone([self.problem.position_translater[i]-1, self.problem.position_translater[j]-1])

                tn = qtensor.optimisation.TensorNet.QtreeTensorNet.from_qtree_gates(composer.circuit)
                peo, _ = opt.optimize(tn)
                peos[(i, j)] = peo
        return peos

    def energy_loss(self):
        sim = qtensor.QtreeSimulator(backend=self.backend)
        composer = qtensor.TorchQAOAComposer_QUBO(self.problem.graph, self.problem.matrix, self.problem.position_translater, gamma=self.gamma, beta=self.beta)
        self.loss = torch.tensor([0.])
        self.E_nodes = {}

        for i in range(1, len(self.problem.matrix)):
            for j in range(1, i+1):
                if self.problem.matrix[i, j] != 0:
                    peo = self.peos[(i, j)]
                    if i == j:
                        composer.energy_expectation_lightcone([self.problem.position_translater[i]-1])
                    else:
                        composer.energy_expectation_lightcone([self.problem.position_translater[i]-1, self.problem.position_translater[j]-1])
                    E = torch.real(sim.simulate_batch(composer.circuit, peo=peo))
                    composer.builder.reset()
                    matrix_entry = self.problem.matrix[i, j]
                    self.loss += E * matrix_entry
                    self.E_nodes [(i, j)] = E


    def create_expect_val_dict(self):
        self.expect_val_dict={}
        max_expect_val = 0
        max_expect_val_list = []
        max_expect_val_location_list = []
        max_expect_val_sign_list = []
        for i in range(1, len(self.problem.matrix)):
            for j in range(1, i+1):
                if self.problem.matrix[i, j] != 0:
                    energy = float(self.E_nodes[(i, j)])
                    if i == j:
                        self.expect_val_dict[frozenset({i})] = energy
                    else:
                        self.expect_val_dict[frozenset({j, i})] = energy
                        
                
                    if abs(energy) > max_expect_val:
                        max_expect_val_list = []
                        max_expect_val_location_list = []
                        max_expect_val_sign_list = []

                        max_expect_val = abs(energy)
                        max_expect_val_sign = np.sign(energy)
                        if i == j:
                            max_expect_val_location = ([self.problem.position_translater[i]])
                        else:
                            max_expect_val_location = ([self.problem.position_translater[i], self.problem.position_translater[j]])
                        
                        max_expect_val_location_list.append(max_expect_val_location)
                        max_expect_val_list.append(max_expect_val)
                        max_expect_val_sign_list.append(max_expect_val_sign)

                    # elif abs(energy) == max_expect_val:
                    #     if i == j:
                    #         max_expect_val_location_help = ([self.problem.position_translater[i]])
                    #     else:
                    #         max_expect_val_location_help = ([self.problem.position_translater[i], self.problem.position_translater[j]])

                    #     max_expect_val_location_list.append(max_expect_val_location_help)
                    #     max_expect_val_list.append(abs(energy))
                    #     max_expect_val_sign_list.append(np.sign(energy))

        if len(max_expect_val_list) > 1:
            print(max_expect_val_list)
            index = random.randrange(len(max_expect_val_list))
            max_expect_val = max_expect_val_list[index]
            max_expect_val_location = max_expect_val_location_list[index]
            max_expect_val_sign = max_expect_val_sign_list[index]

        return max_expect_val_location, max_expect_val_sign, max_expect_val

    def calc_expect_val(self):
        self.energy_loss()
        max_expect_val_location, max_expect_val_sign, max_expect_val = self.create_expect_val_dict()
        
        return max_expect_val_location, max_expect_val_sign, max_expect_val
    
    def optimize(self, steps=50, **kwargs):
        if self.initialization == 'transition_states' and self.p!=1:
            print('working')
            return self.optimize_transition_states(steps=steps, **kwargs)
        else:
            return self.optimize_general(steps=steps, **kwargs)

    def optimize_transition_states(self, steps, **kwargs):
        expectation_value_single = SingleLayerQAOAExpectationValues(self.problem)
        expectation_value_single.optimize()
        gamma_old = [expectation_value_single.gamma/np.pi]
        beta_old = [expectation_value_single.beta/np.pi]

        for step in range(2, self.p+1):
            print('stufe von transition', step)
            for j in range(step):
                print('stufe in transition', j)
                gamma_ts = gamma_old.copy()
                beta_ts = beta_old.copy()
                gamma_ts.insert(j, 0)
                beta_ts.insert(j, 0)
                expectation_values_qtensor_transition = QtensorQAOAExpectationValuesQUBO(self.problem, step, gamma=gamma_ts, beta=beta_ts, pbar=True, opt = self.opt, opt_kwargs=dict(**self.opt_kwargs))
                max_expect_val_location, max_expect_val_sign, max_expect_val = expectation_values_qtensor_transition.optimize(steps=steps, **kwargs)
                energy_qtensor_transition = float(expectation_values_qtensor_transition.energy)

                if j==0:
                    energy_min = energy_qtensor_transition
                    gamma_min = [float(i) for i in expectation_values_qtensor_transition.gamma]
                    beta_min = [float(i) for i in expectation_values_qtensor_transition.beta]
                    correlations_min = expectation_values_qtensor_transition.expect_val_dict.copy()
                    losses_min = expectation_values_qtensor_transition.losses.copy()
                    max_expect_val_location_min , max_expect_val_sign_min , max_expect_val_min = max_expect_val_location, max_expect_val_sign, max_expect_val
                    param_history_min = expectation_values_qtensor_transition.param_history

                if energy_qtensor_transition < energy_min:
                    energy_min = energy_qtensor_transition
                    gamma_min = [float(i) for i in expectation_values_qtensor_transition.gamma]
                    beta_min = [float(i) for i in expectation_values_qtensor_transition.beta]
                    correlations_min = expectation_values_qtensor_transition.expect_val_dict.copy()
                    losses_min = expectation_values_qtensor_transition.losses.copy()
                    max_expect_val_location_min , max_expect_val_sign_min , max_expect_val_min = max_expect_val_location, max_expect_val_sign, max_expect_val
                    param_history_min = expectation_values_qtensor_transition.param_history

            gamma_old = gamma_min.copy()
            beta_old = beta_min.copy()

        self.expect_val_dict = correlations_min
        self.energy = energy_min
        self.losses = losses_min
        self.param_history = param_history_min

        return max_expect_val_location_min , max_expect_val_sign_min , max_expect_val_min 

    def optimize_general(self, steps, **kwargs):
        random.seed()
        self.peos = self.energy_peo()
        opt = self.opt(params=(self.gamma, self.beta) , **self.opt_kwargs)
        self.expect_val_dict = {}
        max_expect_val_location = None
        max_expect_val_sign = None
        max_expect_val = None
        self.losses = []
        self.steps = steps
        self.param_history = []

        self.param_history.append([x.detach().numpy().copy() for x in (self.gamma, self.beta)])
        
        if self.pbar:
            from tqdm.auto import tqdm
            _pbar = tqdm(total=self.steps)
        else:
            _pbar = None

        counter = 0
        for i in range(self.steps):
            self.energy_loss()

            opt.zero_grad()
            self.loss.backward()
            opt.step()
            
            self.losses.append(float(self.loss))
            #self.losses.append(self.loss.detach().numpy().data)
            self.param_history.append([x.detach().numpy().copy() for x in (self.gamma, self.beta)])
            if self.pbar:
                _pbar.update(1)

            if i>1:
                if abs((self.losses[-1]-self.losses[-2])/self.losses[-1]) < 0.0000001:
                    counter += 1
                    if counter == 5:
                        break
                else:
                    counter == 0
        
        max_expect_val_location, max_expect_val_sign, max_expect_val = self.create_expect_val_dict()
        self.energy = self.loss

        return max_expect_val_location, max_expect_val_sign, max_expect_val
    




















    

class QtensorFixedAnglesQAOAExpectationValuesMIS(ExpectationValues):
    """Calculation of expectation values via tensor network contraction using Qtensor"""

    def __init__(self, problem):
        super().__init__(problem)
        self.type = 'QtensorFixedAnglesQAOAExpectationValuesMIS'

class ExampleClass(ExpectationValues):

    def __init__(self, problem):
        super().__init__(problem)
        self.type = 'ExampleClass'
        self.graph = self.problem.graph
        self.expect_val_dict = {}


    def optimize(self, *params):

        #Optimization process:
            #fill self.expect_val_dict in the following way for all correlations: 
            #self.expect_val_dict[frozenset({self.problem.position_translater.index(node+1)})]=float(node_correlation)
            #self.expect_val_dict[frozenset({self.problem.position_translater.index(node1+1), self.problem.position_translater.index(node2+1)})]=float(edge_correlation)


        max_expect_val_location = None # name/names of node/edge with highest absolute correlation 
        max_expect_val_sign = None # sign of the correlation with the highest absolute value
        max_expect_val = None # correlation value with the highest absolute value
        

        return max_expect_val_location, max_expect_val_sign, max_expect_val



    