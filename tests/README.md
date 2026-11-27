# Pytest suite

Run from the repository root using the interpreter that has the dependencies
installed. The fast tests need only a CPU, NumPy, SciPy, Numba and pytest:

```sh
python -m pip install numpy scipy numba pytest
python -m pytest -v -ra
```

This runs four small tests and reports four optional integration cases as
skipped. To select only the fast tests, use `python -m pytest -m "not integration"`.
The fast subprocesses deliberately block imports of `mpi4py` and `mpi4py_fft`,
even if they are installed, to exercise the real serial fallback:

- Two atoms separated by 1.235 Angstrom: exactly two self pairs and two ordered
  interatomic pairs, the latter in bin 123 for `dr=0.01`; an independently
  evaluated sinc spectrum; export and reload of the histogram in another process.
  The expected spectrum includes the existing histogram convention that self
  pairs occupy the first bin midpoint. Its error relative to the exact two-atom
  Debye law is also bounded analytically.
- A perfect 4-by-4-by-4 BCC crystal: the SciPy FFT and diffraction export must
  reproduce the allowed (110) intensity and extinct (100) intensity at the actual
  vectors, including axis permutations and their negatives. No Sobol shifts are
  used, and a negative export threshold retains zero-intensity rows. The allowed
  intensity is derived from the BCC basis factor and the analytic Fourier
  multiplier of the voxel smoothing kernel. A powder-shell average is not used
  to infer extinction.
- A complete one-atom serial CLI run with known histogram-based intensity and
  default output filenames, followed by an unavailable FFTW request that must
  exit nonzero and explain the dependency and the `-fm SCIPY` alternative.
- The actual compiled shell-binning kernel at all eight sign combinations of
  each of the four diagnosed boundary index families. With the fixture grid
  and shell width, the expected bins are 51, 74, 102 and 149. These deliberately
  preserve the reference rounding in both directions.

## Optional reference regressions

Install an MPI implementation (such as Open MPI or MPICH) and `mpi4py` built for
that implementation. For FFTW cases also install FFTW and `mpi4py-fft` (the
`mpi4py_fft` import; `pyfftw` is not the backend used here):

```sh
python -m pip install mpi4py
# After installing FFTW and its development headers/libraries:
python -m pip install mpi4py-fft
python -m pytest --run-integration -v -ra
```

The four parameterized cases reproduce `single_vs_multi.sh`: SciPy and FFTW,
each on one and four MPI ranks, with the committed 125,568-atom LAMMPS data file,
`-ft data -m fft -dx 1.0 -nr 3`, and the corresponding committed reference.
Each subprocess runs in a fresh pytest temporary directory, has a timeout,
and must exit successfully before its output is read. No reference or legacy
`generated_output` file is written. Tests check these independently:

- Exactly the reference row count and three columns: reciprocal coordinate,
  intensity, shell count; all entries must be finite.
- Increasing reciprocal coordinates matching the reference with `rtol=0` and
  `atol=1e-14` (roundoff allowance only).
- Nonnegative integer shell counts matching the reference exactly.
- Intensities matching at the original `rtol=1e-10`, `atol=1e-12`.

`integration`, `mpi` and `fftw` are registered markers. Examples:

```sh
python -m pytest --run-integration -m "integration and not fftw" -v -ra
python -m pytest --run-integration -m fftw -v -ra
MPIEXEC=mpiexec MPIEXEC_FLAGS="--oversubscribe" python -m pytest --run-integration -v -ra
```

`MPIEXEC` defaults to `mpirun`; `MPIEXEC_FLAGS` defaults to empty. Use flags
appropriate to your MPI implementation and allocate at least four slots (the
oversubscription example is for Open MPI). Run pytest normally, not under MPI;
it starts each MPI job itself using the same Python interpreter as pytest.
Numerical library thread counts are limited to one per rank. Each integration
command has a 600-second timeout and needs enough memory for a roughly
128-cubed complex voxel grid and temporary FFT buffers.

Missing launchers or unimportable optional backends are explicit skips with
reasons. Once those prerequisites are available, a failed launch, nonzero CLI
exit, missing output or numerical mismatch is a failure, never silently skipped.
Use `-ra` to see skip reasons and `-v` to see the individual case results.

See [REFERENCE_DIAGNOSIS.md](REFERENCE_DIAGNOSIS.md) for the investigation of
shell-boundary differences, full-precision point traces, isolated arithmetic
experiments, and the Linux MPI/FFTW results. The diagnostic helper does not
weaken these regression checks; its historical experiments run only in memory.
The prescribed FMA/add/sqrt/reciprocal-multiply order is now the compatibility
contract for shell assignment in every backend. The separately tracked
[final-bin export issue](OUTPUT_COMPATIBILITY.md) remains unchanged.
