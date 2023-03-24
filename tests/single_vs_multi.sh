#!/bin/sh
mpirun -n 1 python3 ../debyescherrer.py ../lammpsfiles/vacanneal_slow_9_1800.relax -ft data -m fft -fm SCIPY -dx 1.0 -nr 3 -sx generated_output/svm_scipy_single.dat
mpirun -n 4 python3 ../debyescherrer.py ../lammpsfiles/vacanneal_slow_9_1800.relax -ft data -m fft -fm SCIPY -dx 1.0 -nr 3 -sx generated_output/svm_scipy_multi.dat
mpirun -n 1 python3 ../debyescherrer.py ../lammpsfiles/vacanneal_slow_9_1800.relax -ft data -m fft -fm FFTW  -dx 1.0 -nr 3 -sx generated_output/svm_fftw_single.dat
mpirun -n 4 python3 ../debyescherrer.py ../lammpsfiles/vacanneal_slow_9_1800.relax -ft data -m fft -fm FFTW  -dx 1.0 -nr 3 -sx generated_output/svm_fftw_multi.dat
pyscripts/single_vs_multi.py
