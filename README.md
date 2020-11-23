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
sklearn
```

### Installing

It is highly recommended to run this on a [virtual environment](https://docs.python.org/3/tutorial/venv.html) for python. All required libraries are easily installed with 
```
pip install numpy
pip install etc...
```

## Running the code 

The scripts support serial mode and parallel MPI mode. Parallel mode comes highly recommended systems larger than 100,000 atoms. For example, computing the spectrum for 100,000 atoms takes about 1 minute on a single thread. The computational time scaled as O(N^2) with the number of atoms if no cutoff is used, and scales as O(N) if a cutoff is used. Note that the maximum resolution of the spectrum depends on the cutoff radius via Nyquist's theorem.

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

Passing `LAMMPS` file and settings via command line argument.

## Authors

* **Max Boleininger**, [UKAEA](http://www.ccfe.ac.uk/) 
* **Andrew Warwick**, [UKAEA](http://www.ccfe.ac.uk/) 

