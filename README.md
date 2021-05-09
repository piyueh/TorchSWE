TorchSWE: GPU shallow-water equation solver
===========================================

A simple SWE solver on GPU using several different backends, including CuPy,
PyTorch, and [Legate NumPy](https://github.com/nv-legate/legate.numpy). It can
also run on CPU through PyTorch, vanilla NumPy, or Legate NumPy.

### Installation
----------------

Everything is WIP, including documentation. But basically, to install:

```shell
$ pip install .
```

It installs a binary executable called `TorchSWE.py` to your `bin` path. Which
`bin` path it installs to depends on your `pip`.

After installing through `pip`, only NumPy backend is available. To use other
backends, you may need to install them manually. Both PyTorch and CuPy can be
found from PyPI and Anaconda. Legate NumPy has to be installed manually
currently.

### Usage
---------

To see help

```shell
$ TorchSWE.py --help
```

To run a case with vanilla NumPy, go to a case folder and execute:

```shell
$ TorchSWE.py ./
```

To run with CuPy:

```shell
$ USE_CUPY=1 TorchSWE.py ./
```

To run with PyTorch on a GPU:

```shell
$ USE_TORCH=1 TorchSWE.py ./
```

To run with PyTorch on CPU:

```shell
$ USE_TORCH=1 TORCH_USE_CPU=1 TorchSWE.py ./
```

To run with Legate:

```shell
$ legate <flags> $(which TorchSWE.py) ./
```

For Legate, use its flags to control the target hardware. See Legate's
documentation. Legate does not know where to find the main Python script of the
solver, so we use `$(which ...)` to provide the full path of `TorchSWE.py`.

### Note
--------

I have a very specific application in mind for this solver, so it's capability
is somehow limited. And I intend to keep it simple, instead of generalizing it
to general-purpose SWE solver. However, because it's implemented in Python, I
believe it's not difficult to apply modifications for other applications.
