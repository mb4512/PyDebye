# Reference-spectrum boundary diagnosis — 2026-11-27

## Adopted compatibility contract

The user explicitly adopted the reference-compatible arithmetic below for all
future backends. Production `jSOAS_spectrum_inplace` now implements it:

```python
reciprocal_width = 1.0 / shell_width  # once per kernel call, before voxel loops
ksquared = fma(kx, kx, ky * ky) + kz * kz
shell = int(sqrt(ksquared) * reciprocal_width)
```

The binary64 FMA is emitted explicitly with a Numba intrinsic. `fastmath=False`
on this binning kernel prevents reassociation, accidental fusion of the final
addition and replacement of the prescribed reciprocal multiplication. Other
numerical kernels keep their existing fastmath settings. Inspecting the
optimized LLVM confirms one reciprocal division before the voxel loops, then
strict `fmul`, `llvm.fma.f64`, separate `fmul`, `fadd`, `llvm.sqrt.f64`, `fmul`
and integer conversion. These instructions carry no fastmath flags, including
after intrinsic lowering/inlining.

`test_reference_boundary_families` calls the actual compiled production kernel
for all 32 points using the committed fixture's grid and shell width. It checks
each point independently, including negative-frequency array offsets, exact
counts and intensity placement. The expected bins are 51, 74, 102 and 149;
rounding both upward and downward is intentional compatibility behavior.
The test was observed failing on the previous implementation before the fix.

The [omitted final occupied bin](OUTPUT_COMPATIBILITY.md) is a separate deferred
output-compatibility issue. Its export slice is unchanged, as are all four
references, the exact shell-count assertions and intensity tolerances.

### Verification after implementation

| Platform actually executed | Scope and result |
| --- | --- |
| Linux aarch64, Debian container on a local ARM Podman VM | Full integration suite: **8 passed, 0 failed, 0 skipped**, 26.87 seconds |
| macOS arm64, native | Fast suite: **4 passed**, 4 optional integration cases skipped |
| macOS x86_64 Python through Rosetta on the ARM host | Production boundary test: **1 passed**, covering all 32 points |

The Linux passes include SciPy and FFTW on both one and four MPI ranks, the
two-atom round trip, BCC reflections, serial/unavailable-backend CLI test and
the new boundary test. The Linux numerical/MPI stack is the one listed in the
historical Linux execution section below (Python 3.12.16, NumPy 2.5.5,
SciPy 1.18.1, Numba 0.68.1, llvmlite 0.50.0, Open MPI 5.0.7, FFTW 3.3.10,
mpi4py 4.1.2 and mpi4py-fft 2.0.6).

The native macOS run uses Python 3.12.0 with the same numerical versions.
The Rosetta check uses Python 3.12.12, NumPy 2.2.6, SciPy 1.18.1, Numba 0.61.2
and llvmlite 0.44.0: the newer llvmlite version had no usable Intel macOS wheel
and its source build required unavailable LLVM development files. This is
verification of an emulated x86_64 compiled target, not native Intel hardware
or an x86 MPI/FFTW integration run.

### Runtime impact on the existing regression case

The same Linux image ran the original 125,568-atom regression command with
`-dx 1.0 -nr 3`, using both the pre-change and updated source. Each combination
had three fresh subprocess runs per version, alternating the before/after
order. These wall-clock medians include import, JIT compilation, input reading,
FFT computation and output writing; numerical-library threads were limited
to one per MPI rank.

| Backend / ranks | Before median (s) | After median (s) | Change |
| --- | ---: | ---: | ---: |
| SciPy / 1 | 4.532 | 4.501 | -0.68% |
| SciPy / 4 | 3.943 | 4.009 | +1.68% |
| FFTW / 1 | 4.122 | 4.085 | -0.90% |
| FFTW / 4 | 4.247 | 4.090 | -3.71% |

Nine alternating warmed-kernel trials on the same `(128,128,126)` grid, shell
width and five shifts gave median binning times of **54.111 ms before** and
**55.932 ms after** (+3.37%, about 1.82 ms per five-sample sweep). This isolates
the binning cost using a populated complex array; it excludes FFTs and JIT
startup. The whole-command differences are small compared with observed
run-to-run variation; three trials do not establish the apparent speedups.

The pre-change and post-change kernel LLVM/assembly were inspected separately.
The updated ARM kernel has exactly one reciprocal division outside the loops,
an explicit FMA, and no `fast`, `reassoc`, `contract` or `arcp` arithmetic flags.
The x86_64 check likewise preserves the explicit LLVM FMA and single reciprocal;
the emulated `westmere` CPU target lowers the FMA to a library `fma` call rather
than requiring a hardware FMA instruction.

### Tracking and preservation

`git check-ignore -v` initially identified `.gitignore:2:*` as the rule hiding
the diagnosis, helper, pytest configuration and tests. The blanket rule was
replaced by the repository's previous generated-file rules plus targeted
ignores for pytest caches and the virtual environment created at the repository
root. Intended test/documentation files are no longer ignored; generated
spectra still match `*.dat`, and environment packages match `/lib/python*/`.
Reference hashes, the comparison test's hash, and the export-slice source text
were checked against their pre-change versions and are unchanged.

The following investigation describes the **pre-fix** behavior and the evidence
for the decision. The diagnostic script reconstructs that historical arithmetic
in memory and also reports the current production result.

## Historical investigation

The differences are caused by floating-point arithmetic in reciprocal-shell
assignment. They are reproducible on both macOS arm64 and Linux aarch64, on one
and four MPI ranks, with both SciPy and FFTW. No production code, committed
reference, exact count assertion, or intensity tolerance was changed during
this investigation. The original `rtol=1e-10`, `atol=1e-12` remains in force.

## Direct comparisons

All compared arrays have shape `(193, 3)`. In each row below, reciprocal grids
and shell counts are bit-for-bit equal and intensities pass the existing
tolerances. Relative errors use the second spectrum as denominator.

| Comparison | Maximum relative intensity difference |
| --- | ---: |
| Newly generated SciPy, one vs four ranks (Linux) | 2.4265352284750067e-14 |
| Committed SciPy single vs multi references | 3.6216424435470704e-14 |
| Newly generated FFTW, one vs four ranks (Linux) | 2.4984325685779686e-14 |
| New FFTW vs SciPy, one rank (Linux) | 1.6921706868556662e-15 |
| New FFTW vs SciPy, four ranks (Linux) | 7.526563138036873e-15 |

The macOS SciPy one/four comparison also passes and has the same maximum
relative difference, 2.4265352284750067e-14. The two committed SciPy references
have maximum absolute intensity difference 2.9802322387695312e-08; their large
intensities make that consistent with the relative tolerance.

## Points, bin assignments and conservation

The voxel grid is `(128, 128, 126)`. Its diagonal reciprocal basis is:

```text
bx = 0.0078290006351825064
by = 0.0078240748569508838
bz = 0.0079681801148522857
```

The production shell width is
`0.0045460763626522117` inverse Angstrom
(`0x1.29ee814d7e652p-8` as binary64 hexadecimal).
It is computed as `np.linalg.norm(bmat[0] + bmat[1] + bmat[2]) / 3`.
It also equals exactly twice the first reciprocal coordinate in the committed
reference. Generated and reference reciprocal grids match exactly.

The following table gives representative **zero-based FFT array indices**.
All are in sample 0, the unshifted sample, with reduced Sobol shift `(0,0,0)`
and Cartesian shift `(0,0,0)`. Values are printed with 17 significant digits.

| Grid index | Calculated norm | Shell width | Calculated norm / width | Current bin | Reference-compatible bin |
| --- | ---: | ---: | ---: | ---: | ---: |
| (17,17,17) | 0.23184989449526278 | 0.0045460763626522117 | 50.999999999999993 | 50 | 51 |
| (25,25,25) | 0.34095572719891587 | 0.0045460763626522117 | 75 | 75 | 74 |
| (34,34,34) | 0.46369978899052555 | 0.0045460763626522117 | 101.99999999999999 | 101 | 102 |
| (50,50,50) | 0.68191145439783174 | 0.0045460763626522117 | 150 | 150 | 149 |

Each row represents eight independent sign combinations of its signed Fourier
index `(±m,±m,±m)`, for 32 points total. Negative signed indices map to array
indices `(128-m,128-m,126-m)` on the corresponding axes; for example
`(-17,-17,-17)` is array index `(111,111,109)`. All eight members of each group
have the same norm and bin assignment.

The four nonzero reduced Sobol shifts, generated with seed 59210939, are:

```text
(0.20658443216234446, 0.56411021761596203, 0.88655841536819935)
(0.72288619261234999, 0.19986549858003855, 0.37310967873781919)
(0.75951585173606873, 0.97647400479763746, 0.58236114494502544)
(0.31057786382734776, 0.35439819470047951, 0.1577909654006362)
```

None contributes a point within `1e-10` of the four affected integer shell
boundaries. This threshold only selects diagnostic rows; it is never used to
change a bin assignment. The trace reproduces the uninstrumented kernel's
complete count array for each of the five samples. Each of the 32 assignments
was also checked using the original kernel on a one-voxel slab.

| Adjacent shells | Generated counts | Reference counts | Generated minus reference |
| --- | --- | --- | --- |
| 50, 51 | 31005, 32043 | 30997, 32051 | +8, -8 |
| 74, 75 | 67504, 68906 | 67512, 68898 | -8, +8 |
| 101, 102 | 124425, 127130 | 124417, 127138 | +8, -8 |
| 149, 150 | 57502, 54035 | 57510, 54027 | -8, +8 |

Each pair conserves its total. All exported spectra and references sum to
**10,321,919** points. The full accumulator sums to
`5 * 128 * 128 * 126 = 10,321,920`: the existing export slice
`self.spectrum[:lastindex]` omits the final occupied bin (index 193, one point).
This omission is common to these results and is separate from the eight
shell-count differences; it was not changed in this investigation.

An isolated forensic replay moved only the 32 recorded point contributions
into the reference-compatible bins. This reproduced every reference count
exactly and all reference intensities within tolerance (maximum relative
error `4.386371724455257e-14`). Both the raw intensity sum and total count were
conserved. These fixture-specific moves were not installed as a fix.

## First divergent calculation and isolated experiments

The production binning function has `@jit(nopython=True, fastmath=True)`.
Its ARM machine code includes fused multiply-add (`fmadd`), square root
(`fsqrt`), multiplication by a reciprocal (`fmul`), and truncation (`fcvtzs`).

For the representative points, the reciprocal components and shell width are
unchanged between arithmetic variants. The first divergent value is the
evaluation of `kx*kx + ky*ky + kz*kz`, before the square root and division.
For `(17,17,17)`, fastmath gives `0.05375437357746448`; strict arithmetic gives
`0.053754373577464487`. The corresponding norms are
`0.23184989449526278` and `0.2318498944952628`, producing bins 50 and 51.

Each experiment below compiles a separate copy of the production function in
memory. No disk edits are needed, and no epsilon or modified integer-binning
rule is introduced.

| Isolated arithmetic experiment | Shells differing from the reference |
| --- | --- |
| Original `fastmath=True` | 50, 51, 74, 75, 101, 102, 149, 150 |
| `fastmath=False` | 74, 75, 149, 150, 170, 171 |
| All fastmath flags except `contract` | 74, 75, 149, 150, 170, 171 |
| All fastmath flags except `arcp` | 50, 51, 74, 75, 101, 102, 149, 150, 170, 171 |
| `contract` only | 38, 39, 74, 75, 77, 78, 149, 150, 155, 156, 170, 171 |
| `arcp` only | 74, 75, 149, 150, 170, 171 |
| Six strict permutations of the three squared terms | None reproduces all counts |
| Explicit single FMA convention below | No differing shells |

Reciprocal multiplication can matter independently: at `(57,57,57)`, using the
same fastmath norm `0.77737905801352813`, reciprocal multiplication produces
`170.99999999999997`, while division with `arcp` disabled produces `171`.

Recomputing the shell width with left- or right-associated sums, `math.fsum`,
or `np.linalg.norm(v/3)` gives the **same binary64 value** on this machine.
Running the production kernel with each of these widths leaves the same eight
mismatches. The shell-width computation is not the divergent step here.

One explicitly controlled convention reproduces the entire historical result:

```python
# Diagnostic only: correctly rounded FMA, other operations strict binary64.
ksquared = fma(kx, kx, ky * ky) + kz * kz
knorm = sqrt(ksquared)
shell = int(knorm * (1.0 / shell_width))
```

| Signed index | Candidate squared norm | Candidate norm | Candidate quotient |
| --- | ---: | ---: | ---: |
| (17,17,17) | 0.053754373577464487 | 0.2318498944952628 | 51 |
| (25,25,25) | 0.11625080790974152 | 0.34095572719891581 | 74.999999999999986 |
| (34,34,34) | 0.21501749430985795 | 0.46369978899052561 | 102 |
| (50,50,50) | 0.46500323163896606 | 0.68191145439783163 | 149.99999999999997 |

This convention matches the complete count histogram, not just these four
examples. Substituting only this candidate binning kernel in a separate serial
SciPy calculation matches the entire committed spectrum at the original
tolerances (maximum relative intensity error `4.386371724455257e-14`).

This establishes a **compatible numerical convention**, not the provenance of
the historical compiler or hardware. The references do not document either.
The convention assigns some mathematically exact diagonal boundaries to the
lower shell and others to the upper shell. There is no scientific reason here
to privilege this particular fusion order over other evaluations of the same
formula. At that stage no production fix or pytest expectation locking in that
convention was added. The subsequent explicit compatibility decision and its
implementation are recorded above.

## Linux execution

Both FFTW cases were executed in an isolated Debian container in a local
Podman Linux VM, with source copied from the working tree. Environment:

```text
Linux aarch64, kernel 6.12.13-200.fc41.aarch64
Python 3.12.16; pytest 9.2.0
NumPy 2.5.5; SciPy 1.18.1; Numba 0.68.1; llvmlite 0.50.0
Open MPI 5.0.7; FFTW 3.3.10; mpi4py 4.1.2; mpi4py-fft 2.0.6
MPIEXEC_FLAGS=--oversubscribe
python -m pytest --run-integration -v -ra --tb=short --basetemp=/work/results
```

The image used `python:3.12-slim`, installing `build-essential`, `openmpi-bin`,
`libopenmpi-dev`, and `libfftw3-dev` with apt, then the Python packages above.
All four MPI calculation subprocesses exited 0 and produced spectra. The
subsequent scientific assertions failed as listed below; these were not launch
or dependency failures. The VM was stopped after results were copied out.

| Case | Linux pytest result |
| --- | --- |
| SciPy, 1 rank | Failed: same eight shell-count differences |
| SciPy, 4 ranks | Failed: same eight shell-count differences |
| FFTW, 1 rank | Failed: same eight shell-count differences |
| FFTW, 4 ranks | Failed: same eight shell-count differences |
| Two-atom histogram round trip | Passed |
| BCC vector reflections | Passed |
| Serial CLI and unavailable backend | Passed |

**Total: 3 passed, 4 failed, 0 skipped. Cases that could not be executed: none.**
The isolated candidate calculation is a diagnostic result, not a passing result
for these unchanged production tests.

## Repeating the numerical diagnosis

The helper runs on a CPU without MPI or FFTW. It outputs package versions,
direct comparisons, width and fastmath experiments, arithmetic permutations,
all 32 array/signed indices and intermediate values, and conservation checks.
Use the spectrum paths from a fresh integration run:

```sh
python tests/pyscripts/diagnose_shell_boundaries.py \
  --one /path/to/one-rank/spectrum.dat \
  --four /path/to/four-rank/spectrum.dat \
  --candidate-replay --output /tmp/pydebye-boundary-diagnosis.json
```

Omit `--one` and `--four` to run only the kernel/reference diagnostics. The
diagnostic replacement is active only inside this process, is restored
afterward, and never writes a spectrum or a reference file. The CLI itself
uses the production implementation of the approved convention.
