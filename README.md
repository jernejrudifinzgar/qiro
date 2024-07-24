# qiro branch tensor networks
Quantum-Informed Recursive Optimization Algorithms. 

Contains most of the code; stuff to be added: parallel tempering, simulated annealing. 

### Installation

Set up a new conda environment with the following command:

```
conda create -n qiro python=3.9.13
```
Activate the environment:
```
conda activate qiro
```
Clone repository with submodules:
```
git clone --recurse-submodules https://github.com/jernejrudifinzgar/qiro.git
```
Go into repository:
```
cd qiro
```
Go into tensor_networks branch:
```
git checkout tensor_networks
```
Then install dependencies:
```
pip install -r requirements.txt
```
Then go into Qtensor folder and install corresponding requirements:
```
git submodule update --recursive --remote
cd Qtensor/qtree_git
pip install .
cd ..
pip install . 
```
Then go back to main repository and you are ready to go:
cd ..


### File descriptions:

#### QIRO_Execution_Tensor_Networks.ipynb

Jupyter Notebook where you can play around with different problems and execute higher depth RQAOA and QIRO with tensor networks for MIS, MaxCut and SetCover. Should probably be the entry point for anyone who wants to use this code.

#### QIRO_Execution.ipynb

Jupyter Notebook where you can play around with different problems and execute p=1 RQAOA and QIRO for MIS and MAX-2-SAT. Should probably be the entry point for anyone who wants to use this code.

#### Generating_Problems.py

Contains generation of MIS and MAX-2-SAT problems, and transforming them in the correct shape for QIRO and RQAOA.

#### Calculating_Expectation_Values.py

Contains functions for calculating the expectation values of the correlations from QAOA for MIS, MaxCut, SetCover and MAX-2-SAT problems. Should in principle work for any quadratic Hamiltonian.

#### QIRO.py

Contains the QIRO algorithm for MAX-2-SAT problems.
Contains multiple QIRO algorithms for MIS problems.
Contains multiple QIRO algorithms for SetCover problems.

#### RQAOA.py

Contains the RQAOA algorithm (with and without recalculation intervals) for any quadratic problem (among others, MIS, MaxCut and MAX-2-SAT).

#### Experiments

Contains all scripts, data and results of the experiments of Max' Master thesis. You can use this to see how tensor networks are used. 

#### Parameter_optimizarion

Contains all scripts, data and results of the QAOA parameter optimization analysis of Max' Master thesis. You can consult this to see how well which optimizers are working for the different problem classes. 

#### Qtensor

Contains the Qtensor submodule for tensor networks construction and contraction. The submodule is in Max' git and different from the original Qtensor repository. 

#### Helping_tools

Contains different scripts that are useful for some tasks, such as Gurobi solver or instance generators. 

### Classical Benchmarks

Contains the code for classical benchmark algorithms for the different problems.


