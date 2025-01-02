# active-phase-mapping

Code for active phase mapping. Paper can be found here: https://arxiv.org/abs/2402.15582

## Setup notes

Currently uses jax and
[gpjax](https://gpjax.readthedocs.io/en/latest/installation.html), which relies
on an updated version of python:

```
conda create --name apm python=3.10
conda activate apm
conda install -y scipy
pip install jax jaxlib
pip install gpjax
```

## Code notes

The active learning algorithms are in the active_learning directory.
All data generated in the performance testing can be found in the performance_test directory. 
cal.py  -- General active learning procedure used in the paper. 
base.py -- Baseline active learning procedure--not hull aware. 
utils.py -- Lots of miscellaneous functions.
gp_model.py -- script detailing Gaussian Process regression model used in both policies. 
base_policy.py -- functions used in implementing active learning for baseline policy.
mpi.py -- Cal is written to be parallelized across cores using mpi. mpi.py provides function for parallelization of EIG calculation.
fps.py -- script for farthest point sampling.

## Running code
The main code is in '''cal.py''' and can be run as a module, passing arguments using options. If using SLURM on an HPC to run on multiple cores in parallel, '''srun''' can be used as follows:
```
srun python cal.py -num_y 10 -num_samples 200 -num_curves 200 -entropy_type joint -cores 6 -nodes 1 -directory ../data/ternary -method CHAASE
```