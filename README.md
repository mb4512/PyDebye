# PyDebye

Tool to compute the Debye-Scherrer line profile for LAMMPS dump and restart files, written in 100% Python. Supports periodic boundary conditions for orthogonal simulation cells using the minimum image convention. 

## Getting Started

git clone https://github.com/mb4512/PyDebye.git

### Prerequisites

This project was written and tested with Python 3.8.2

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

For parallel use in an HPC environment, ensure to first module load the MPI version which will be used in the parallel jobs, and only then install mpi4py, as otherwise mpi4py will be linked to the incorrect MPI version.

Note that on some systems, pip fails to install `numba` because of a missing `llvm` dependecy. In some cases, this can be fixed by running:
```
pip install llvmlite==0.31.0
```
with an up to date `pip` and `setuptools`.


## Running the code 

The scripts support serial mode and parallel MPI mode. Parallel mode comes highly recommended for systems larger than 100,000 atoms. For example, computing the spectrum for 100,000 atoms with no periodic boundary conditions takes about 1 minute on a single thread. The computational time scales as O(N^2) with the number of atoms N if no cutoff is used, and scales as O(N) if a cutoff is used. In periodic systems, the cutoff cannot be larger than half the smaller box length.

For an overview of input parameters and default values, run
```
python debyescherrer.py -h
```

or 

```
python debyescherrer.py --help
```

For an example of serial usage, the below command will build the bond histogram for the given LAMMPS data file `lammpsfiles/vacanneal_slow_9_1800.relax` using 3D PBC with a cutoff radius of 30 Angstroms, and then compute the Debye-Scherrer spectrum with applied continuum correction and damping window function:
```
python3 debyescherrer.py lammpsfiles/vacanneal_slow_9_1800.relax -ft data -pbc -rc 30.0 -damp -ccor
```
This will take around 40 seconds. 

The same command can be run in parallel MPI mode using
```
mpirun -n X python3 debyescherrer.py lammpsfiles/vacanneal_slow_9_1800.relax -ft data -pbc -rc 30.0 -damp -ccor
```
where `X` is the number of MPI threads. For 8 threads, this will take around 5 seconds.

By default, histogram and spectrum files are saved in `histogram.dat` and `spectrum.dat`, respectively. Output files can be changed with the `-hx` and `-sx` flags. In the histogram file, the first column is the bin mid-point and the second column is the number of atoms in the bin.  In the spectrum file, the first column is the s value and the second column is the scattering intensity.

Once a histogram is stored, it can be used as input to quickly recompute the spectrum, here for example with a higher resolution in reciprocal space:
```
mpirun -n X python3 debyescherrer.py histogram.dat -ft hist -damp -ccor -spts 1001
```


## To be implemented

Currently only completely free boundaries or 3D periodic boundaries are supported, mixed boundaries are to be implemented. N*log(N) scaling Structurefactor methods was implemented, but requires documentation,

## Authors

* **Max Boleininger**, [UKAEA](http://www.ccfe.ac.uk/), max.boleininger@ukaea.uk
* **Andrew Warwick**, [UKAEA](http://www.ccfe.ac.uk/) , andrew.warwick@ukaea.uk

