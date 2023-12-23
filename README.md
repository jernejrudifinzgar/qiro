# qiro
Quantum-Informed Recursive Optimization Algorithms. 

Contains most of the code of the correspodning publication: https://arxiv.org/abs/2308.13607

### Installation

Set up a new conda environment with the following command:

```
conda create -n qiro python=3.9.13
```
Activate the environment:
```
conda activate qiro
```
Then install dependencies:
```
pip install -r requirements.txt
```


### Module descriptions:

#### QIRO_Execution.ipynb

Jupyter Notebook where you can play around with different problems and execute RQAOA and QIRO for MIS and MAX-2-SAT. Should probably be the entry point for anyone who wants to use this code.

#### problem_generation

Contains generation of MIS and MAX-2-SAT problems, and transforming them in the correct shape for QIRO and RQAOA.

#### expval_calculation

Contains functions for calculating the expectation values of the correlations from p=1 QAOA for MIS and MAX-2-SAT problems. Should in principle work for any quadratic Hamiltonian. 

#### QIRO_MAX_2_SAT.py

Contains the QIRO algorithm for MAX-2-SAT problems.

#### QIRO_MIS.py

Contains the QIRO algorithm for MIS problems.

#### misc/RQAOA.py

Contains the RQAOA algorithm for any quadratic problem (among others, MIS and MAX-2-SAT).

#### misc/aws_quera.py

Contains the code for running the QIRO algorithm to solve MIS on QuEra Aquila (AWS Braket).

### Classical Benchmarks
#### greedy_mis.py

Contains the code for the greedy algorithm for MIS.

#### Parallel_Tempering.py

Contains the code for the parallel tempering algorithm.

#### Simulated_Annealing.py

Contains the code for the simulated annealing algorithm.

### Utils

Some additional class definitions.
