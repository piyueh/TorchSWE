TorchSWE: GPU shallow-water equation solver
===========================================

TorchSWE is a simple parallel (MPI & GPU) SWE solver supporting several
different backends: CuPy, PyTorch, and
[Legate NumPy](https://github.com/nv-legate/legate.numpy)†.

The MPI support is done through [mpi4py](https://github.com/mpi4py/mpi4py) and a
simple domain decomposition algorithm. For multi-GPU settings (either multiple
GPUs on a single node or across a cluster), only MPI + CuPy have been tested.
For regular CPU clusters, use MPI + NumPy. For single GPU, both CuPy and PyTorch
work fine. Also, PyTorch provides a shared-memory parallelization for a single
CPU computing node.

**Note**  
† Legate NumPy backend has been removed from the master branch due to
incompatibility with MPI. Also, Legate NumPy lacks some required features at its
current stage, making it non-trivial to maintain Legate-compatible code.
Therefore, the last version supporting Legate NumPy has been archived
to release [v0.1](https://github.com/piyueh/TorchSWE/releases/tag/v0.1).

### Installation
----------------

Dependencies can be installed using Anaconda. For example, to create a new
Anaconda environment that is called `torchswe` and has all backends (assuming
now we are under the top-level directory of this repository):
```
$ conda env create -n torchswe -f conda/torchswe.yml
```
Next, source into the environment:
```
$ conda activate torchswe
```
or
```
$ source ${CONDA_PREFIX}/bin/activate torchswe
```
Then install TorchSWE with `pip`:
```
$ pip install .
```
It installs an executable, `TorchSWE.py`, to the `bin` directory of this
Anaconda environment.

Following the above workflow, the MPI backend will be OpenMPI and is
CUDA-aware. If a user wants to use MPICH and multiple GPUs, the user may have
to build MPICH from scratch. (The MPICH package from Anaconda's
`conda-forge` channel does not support CUDA.) Also, to use MPICH, it's necessary
to use MPICH-compatible `netcd4`. (For example, if using Anaconda, do
`$ conda install -c conda-forge "netcdf4=*=mpi_mpich*"`.)

The Anaconda environment created using `torchswe.yml` does not have dependencies
for post-processing/visualizing the results of example cases. These dependencies
include `matplotlib` and `pyvista`. Users can install them separately or,
alternatively, create the Anaconda environment with `development.yml`.

### Example cases
-----------------
Example cases are under the folder `cases`.

### Usage
---------

To see help

```
$ TorchSWE.py --help
```

To run a case:

- using MPI + NumPy (assuming already in a case folder)
  ```
  $ mpiexec -n <number of processes> TorchSWE.py ./
  ```
- using MPI + CuPy (assuming already in a case folder)
  ```
  $ USE_CUPY=1 mpiexec \
        -n <number of processes> \
        --mca opal_cuda_support 1 \
        TorchSWE.py ./
  ```
  Note that using `--mca opal_cuda_support 1` is required if OpenMPI is installed
  through Anaconda. The OpenMPI from Anaconda is built with CUDA but does not
  enable the CUDA support by default.
  
  When multiple GPUs are available on a compute node, the code assigns GPUs based
  on local ranks (local to the compute node), not the global rank. The number of
  processes (i.e., ranks) does not have to be the same as the number of
  available GPUs. If the number of processes is higher than that of GPUs,
  multiple ranks will share GPUs. Performance penalty, however, may apply in
  this case.

- using PyTorch (assuming already in a case folder)
  ```
  $ USE_TORCH=1 TorchSWE.py ./
  ```
  The MPI support of PyTorch has not been tested at all. So currently, it's
  better to use only one GPU when using PyTorch backend.

- using PyTorch's shared-memory CPU backend 
  ```
  $ USE_TORCH=1 TORCH_USE_CPU=1 TorchSWE.py ./
  ```
  This runs the solver with shared-memory parallelization from PyTorch and is
  hence only available when using one computing node.
