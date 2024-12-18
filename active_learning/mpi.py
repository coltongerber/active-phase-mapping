### Adapted from Mines HPC workshop Fall 2022 led by Nicholas Danes
from mpi4py import MPI
import time
from utils import EIG_chaase_meanfield, EIG_chaase_meanfield_multi
import numpy as np

t0 = time.time()
comm = MPI.COMM_WORLD
rank = comm.rank
nprocs = comm.Get_size()
print("rank=", rank, " nprocs=", nprocs)

def do_parallel(designs=None, dataset=None, seed=None, index_dict=None,  
pred_mean=None, pred_cov=None, design_space=None, num_y=None, 
initial_entropy=None, pts=None, endpoint_indices=None, knot_N=None, 
num_curves=None, num_samples=None, poly_dict=None, polymorph_index=None, 
num_polymorphs=None, dimensions=None, true_classifications=None, entropy_type=None):

    # Divide up tasks to roughly equal number per rank (except final)
    num_paths = len(designs)
    paths_per_proc = int(np.floor(num_paths / nprocs))
    start = rank * paths_per_proc
    rem = int(np.mod(num_paths, nprocs))
    print(f'{num_paths} items to process\n{paths_per_proc} items per core (rank)\n{paths_per_proc+rem} items for final rank')

    # Add remaining tasks to final rank's tasks
    if rem != 0 and rank == nprocs - 1:
        paths_per_proc += rem
    end = start + paths_per_proc

    # Actually do the computations on each item in assigned section of designs
    data_dict = {}
    print(f'Starting at entry {start} and ending at {end-1}')

    #for a given composition
    if num_polymorphs == 1:
        for composition in designs[start:end]:
            EIG=EIG_chaase_meanfield(
                composition=composition, dataset=dataset, seed=seed, index_dict=index_dict,
                pred_mean=pred_mean, pred_cov=pred_cov, design_space=design_space,
                num_y=num_y, initial_entropy=initial_entropy, pts=pts, endpoint_indices=endpoint_indices,
                knot_N=knot_N, num_samples=num_samples, dimensions=dimensions,
                true_classifications=true_classifications, entropy_type=entropy_type)
            data_dict[float(EIG)] = composition
    else:
        for composition in designs[start:end]:
            EIG=EIG_chaase_meanfield_multi(
                composition=composition, dataset=dataset, seed=seed, index_dict=index_dict,  
                pred_mean=pred_mean, pred_cov=pred_cov, design_space=design_space, 
                num_y=num_y, initial_entropy=initial_entropy, pts=pts, endpoint_indices=endpoint_indices, 
                knot_N=knot_N, num_curves=num_curves, num_samples=num_samples, poly_dict=poly_dict, 
                polymorph_index=polymorph_index, num_polymorphs=num_polymorphs, dimensions=dimensions, 
                true_classifications=true_classifications, entropy_type=entropy_type)
            data_dict[float(EIG)] = composition

    master_dict=comm.gather(data_dict, root=0)
    master_dict = comm.bcast(master_dict, root = 0)
    
    return master_dict

if __name__ == '__main__':
    do_parallel()
