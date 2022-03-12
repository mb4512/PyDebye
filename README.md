# DebyeScherrer

Tool to compute the Debye-Scherrer line profile for LAMMPS dump and restart files. Supports periodic boundary conditions for orthogonal simulation cells using the minimum image convention. 

## Getting Started

git clone https://github.com/mb4512/DebyeScherrer.git

### Prerequisites

This project was written and tested only with Python 3.5.2

Required libraries:
```
numpy
scipy
numba
mpi4py
```

### Installing

It is highly recommended to run this on a [virtual environment](https://docs.python.org/3/tutorial/venv.html) for python. All required libraries are easily installed with 
```
pip install numpy
pip install etc...
```

Note that on some systems, pip fails to install `numba` because of a missing `llvm` dependecy. In some cases, this can be fixed by running:
```
pip install llvmlite==0.31.0
```
with an up to date `pip` and `setuptools`.


## Running the code 

The scripts support serial mode and parallel MPI mode. Parallel mode comes highly recommended for systems larger than 100,000 atoms. For example, computing the spectrum for 100,000 atoms with no periodic boundary conditions takes about 1 minute on a single thread. The computational time scales as O(N^2) with the number of atoms N if no cutoff is used, and scales as O(N) if a cutoff is used. Note that the maximum resolution of the spectrum depends on the cutoff radius via Nyquist's theorem. In periodic systems, the cutoff cannot be larger than half the smaller box length.

The script is run in serial mode using
```
python test.py
```

and in parallel MPI mode using
```
mpirun -n X python test.py
```

where `X` is the number of MPI threads. The spectrum is saved in the `spectrum.out` file, where the first column is the s value and the second column is the scattering intensity.

## To be implemented

Passing `LAMMPS` file and settings via command line argument. Currently the spectral resolution is user-supplied, though it should follow from Nyquist's theorem. Currently only completely free boundaries or 3D periodic boundaries are supported, mixed boundaries are to be implemented.

## Authors

* **Max Boleininger**, [UKAEA](http://www.ccfe.ac.uk/), max.boleininger@ukaea.uk
* **Andrew Warwick**, [UKAEA](http://www.ccfe.ac.uk/) , andrew.warwick@ukaea.uk

