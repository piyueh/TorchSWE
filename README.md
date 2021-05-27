TorchSWE: GPU shallow-water equation solver
===========================================

A simple SWE solver on GPU using several different backends, including CuPy,
PyTorch, and [Legate NumPy](https://github.com/nv-legate/legate.numpy). It can
also run on CPU through PyTorch, vanilla NumPy, or Legate NumPy.

A naive implementation for distributed-memory system is also available through
*mpi4py*. I didn't use any special algorithm that is tailored to
distributed-memory system. I just tried to do bare minimum modifications.
Basically, it's built on top of the non-MPI version and adds communication to
exchange data in overlapped cells (i.e., ghost cells) between processes.
Currently, for distributed-memory system, only MPI + NumPy and MPI + CuPy have
been tested.

### Installation
----------------

Everything is WIP, including documentation. But basically, to install:

```
$ pip install .
```

It installs two executables, `TorchSWE.py` and `TorchSWEMPI.py` to your `bin`
path. Which `bin` path it installs to depends on your `pip`.

After installing through `pip`, only NumPy backend is available. To use other
backends, you may need to install them manually. Both PyTorch and CuPy can be
found from PyPI and Anaconda. Legate NumPy has to be installed manually
currently.

MPI (OpenMPI or MPICH) and *mpi4py* are available in Anaconda.


### Usage
---------

To see help

```
$ TorchSWE.py --help
```

#### Non-MPI version
--------------------

- To run a case with vanilla NumPy, go to a case folder and execute:
  ```
  $ TorchSWE.py ./
  ```

- To run with CuPy:
  ```
  $ USE_CUPY=1 TorchSWE.py ./
  ```

- To run with PyTorch on a GPU:
  ```
  $ USE_TORCH=1 TorchSWE.py ./
  ```

- To run with PyTorch on CPU:
  ```
  $ USE_TORCH=1 TORCH_USE_CPU=1 TorchSWE.py ./
  ```

- To run with Legate:
  ```
  $ legate <flags> $(which TorchSWE.py) ./
  ```
  
  For Legate, use its flags to control the target hardware. See Legate's
  documentation. Legate does not know where to find the main Python script of
  the solver, so we use `$(which ...)` to provide the full path of `TorchSWE.py`.

#### MPI version
----------------

The MPI version requires either OpenMPI or MPICH and `mpi4py`. Also, the
`netcdf4` must be compiled with MPI support. If using Anaconda, the easiest way
to get it is through `conda install -c conda-forge "netcdf4=*=mpi_openmpi*"`.
This installs both OpenMPI and MPI-enabled NetCDF4.

Didn't test with MPICH, so I'm not sure if MPICH works with CUDA.

- To run a case using MPI + NumPy (assuming already in a case folder)
  ```
  $ mpiexec -n <number of processes> TorchSWEMPI.py ./
  ```
- To run a case using MPI + CuPy (assuming already in a case folder)
  ```
  $ USE_CUPY=1 mpiexec -n <number of processes> TorchSWEMPI.py ./
  ```
  When multiple GPUs are availabe on a compute node, the code assigns GPUs based
  on local ranks (local to the compute node; not the global rank). Note
  that the number of processes (i.e., ranks) does not have to be the same as the
  number of available GPUs. If the number of processes is higher than that of
  GPUs, multiple ranks will share GPUs. Nevertheless, no study has been done
  to understand if this will give any performance penalty or benefit. To see
  which GPU on which node is assigned to which rank, run the simulation with
  `--log-level debug`. The information should appear at the very beginning of
  the output.
  
  You may need to install `mpi4py` from its GitHub's master branch. The
  currently released versions (as of v3.0.3) do not supported CuPy buffer yet.
  
  Also, if using the OpenMPI from Anaconda, you need to add an extra flag
  `--mca opal_cuda_support 1` to the `mpiexec` command because this build
  disables the CUDA suppory by default.

### Note
--------

I have a very specific application in mind for this solver, so it's capability
is somehow limited. And I intend to keep it simple, instead of generalizing it
to general-purpose SWE solver. However, because it's implemented in Python, I
believe it's not difficult to apply modifications for other applications.
