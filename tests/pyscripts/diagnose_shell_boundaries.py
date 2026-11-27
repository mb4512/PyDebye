"""Trace the legacy reference case without editing kernels or reference files.

Run with the normal NumPy/SciPy/Numba dependencies. MPI and FFTW are not needed.
The production kernel now implements the explicit FMA compatibility contract.
Variants reconstruct the pre-fix expression in memory to retain the historical
diagnosis; they never replace the production kernel outside this process.
"""

import argparse
import contextlib
import inspect
import io
import itertools
import json
import math
from pathlib import Path
import platform
import sys

import llvmlite
import numba
from numba import njit, types
from numba.extending import intrinsic
import numpy as np
import scipy
from scipy.stats import qmc


ROOT = Path(__file__).resolve().parents[2]


@intrinsic
def fma64(typingctx, x, y, z):
    """Force one correctly rounded FMA, independently of fastmath contraction."""
    def codegen(context, builder, signature, args):
        return builder.fma(*args)
    return types.float64(types.float64, types.float64, types.float64), codegen


def variant(original, *, fastmath, trace=False, candidate=False, order=None):
    source = inspect.getsource(original.py_func)
    source = source[source.index("def "):]
    # Reconstruct the historical arithmetic before applying each isolated
    # experiment. The original dispatcher still calls the production kernel.
    assignment = "ksquared = _fma64(kx, kx, ky*ky) + kz*kz\n                shell = int(math.sqrt(ksquared) * reciprocal_width)"
    assert assignment in source, "update the diagnostic for the changed production source"
    source = source.replace(assignment, "knorm = math.sqrt(kx*kx + ky*ky + kz*kz)")
    source = source.replace("spectrum[shell]", "spectrum[int(knorm/kres)]")
    source = source.replace("counts[shell]", "counts[int(knorm/kres)]")
    if candidate:
        source = source.replace("math.sqrt(kx*kx + ky*ky + kz*kz)",
                                "math.sqrt(fma64(kx, kx, ky*ky) + kz*kz)")
        source = source.replace("knorm/kres", "knorm*(1.0/kres)")
    if order:
        x, y, z = order
        source = source.replace("math.sqrt(kx*kx + ky*ky + kz*kz)",
                                f"math.sqrt(({x}*{x} + {y}*{y}) + {z}*{z})")
    if trace:
        source = source.replace("local_shape = psi_k.shape", """local_shape = psi_k.shape
    trace = np.empty(local_shape + (6,))""")
        source = source.replace("spectrum[int(knorm/kres)] += _int_k", """trace[i,j,k,0] = kx
                trace[i,j,k,1] = ky
                trace[i,j,k,2] = kz
                trace[i,j,k,3] = kx*kx + ky*ky + kz*kz
                trace[i,j,k,4] = knorm
                trace[i,j,k,5] = knorm/kres
                spectrum[int(knorm/kres)] += _int_k""")
        source = source.replace("return 0", "return trace")
    namespace = {"np": np, "math": math, "fma64": fma64}
    exec(source, namespace)
    return njit(fastmath=fastmath)(namespace[original.py_func.__name__])


def compare(a, b):
    return {
        "shape_equal": a.shape == b.shape,
        "grid_equal": bool(np.array_equal(a[:, 0], b[:, 0])),
        "counts_equal": bool(np.array_equal(a[:, 2], b[:, 2])),
        "intensity_allclose": bool(np.allclose(a[:, 1], b[:, 1], rtol=1e-10, atol=1e-12)),
        "max_relative_intensity_difference": float(np.max(abs(a[:, 1]-b[:, 1])/abs(b[:, 1]))),
        "count_totals": [int(a[:, 2].sum()), int(b[:, 2].sum())],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--one", type=Path, help="generated one-rank SciPy spectrum")
    parser.add_argument("--four", type=Path, help="generated four-rank SciPy spectrum")
    parser.add_argument("--output", type=Path, help="JSON output (otherwise stdout)")
    parser.add_argument("--candidate-replay", "--replay", action="store_true",
                        help="also replay the full serial FFT with the approved FMA convention")
    args = parser.parse_args()
    if bool(args.one) != bool(args.four):
        parser.error("supply both --one and --four")

    # Only this diagnostic interpreter is affected. Do not initialize an MPI
    # runtime merely to inspect a numerical kernel or run its serial replay.
    sys.path.insert(0, str(ROOT))
    sys.modules["mpi4py"] = None
    sys.modules["mpi4py_fft"] = None
    from lib.readfile import ReadFile
    import lib.structurefactor as sf

    with contextlib.redirect_stdout(io.StringIO()):
        atoms = ReadFile(ROOT / "lammpsfiles/vacanneal_slow_9_1800.relax")
        atoms.load()
    b = np.linalg.inv(atoms.cmat).T
    shape = 1 + np.linalg.norm(atoms.cmat, axis=1).astype(int)
    shape[shape % 2 == 1] += 1
    v = b[0] + b[1] + b[2]
    widths = {
        "numpy_norm": np.linalg.norm(v)/3,
        "sqrt_left": math.sqrt((v[0]*v[0] + v[1]*v[1]) + v[2]*v[2])/3,
        "sqrt_right": math.sqrt(v[0]*v[0] + (v[1]*v[1] + v[2]*v[2]))/3,
        "sqrt_fsum": math.sqrt(math.fsum(float(x*x) for x in v))/3,
        "norm_predivide": np.linalg.norm(v/3),
    }
    width = widths["numpy_norm"]
    shifts = np.r_[np.zeros((1, 3)), qmc.Sobol(3, seed=59210939).random(4)]
    ref = np.loadtxt(ROOT / "tests/reference_output/svm_scipy_single.dat")
    ref4 = np.loadtxt(ROOT / "tests/reference_output/svm_scipy_multi.dat")
    original = sf.jSOAS_spectrum_inplace
    historical = variant(original, fastmath=True)
    psi = np.zeros(shape, dtype=complex)

    def histogram(func, spacing=width):
        counts, spectrum = np.zeros(500, dtype=int), np.zeros(500)
        for shift in shifts:
            func(spectrum, counts, spacing, b, b@shift, psi, shape, 0, 0, 0)
        ix = np.flatnonzero(counts[:len(ref)] != ref[:, 2])
        return {"mismatch_bins": ix.tolist(), "deltas": (counts[ix]-ref[ix, 2]).tolist(),
                "full_total": int(counts.sum()), "exported_total": int(counts[:len(ref)].sum())}

    result = {
        "environment": {"platform": platform.platform(), "python": platform.python_version(),
                        "numpy": np.__version__, "scipy": scipy.__version__,
                        "numba": numba.__version__, "llvmlite": llvmlite.__version__},
        "shape": shape.tolist(), "box": atoms.box.tolist(), "bmat": b.tolist(),
        "widths": {key: {"decimal_17g": format(value, ".17g"), "hex": float(value).hex()}
                   for key, value in widths.items()},
        "reference_width_equal": bool(width == 2*ref[0, 0]),
        "shifts": shifts.tolist(), "reference_one_vs_four": compare(ref, ref4),
        "production": histogram(original),
        "historical": histogram(historical),
        "width_experiments": {name: histogram(historical, value) for name, value in widths.items()},
    }
    if args.one:
        one, four = np.loadtxt(args.one), np.loadtxt(args.four)
        result.update({"generated_one_vs_four": compare(one, four),
                       "one_vs_reference": compare(one, ref), "four_vs_reference": compare(four, ref4)})

    flags = {"strict": False, "contract_only": {"contract"}, "arcp_only": {"arcp"},
             "no_contract": {"reassoc", "nnan", "ninf", "nsz", "arcp", "afn"},
             "no_arcp": {"reassoc", "nnan", "ninf", "nsz", "contract", "afn"}}
    result["fastmath_experiments"] = {
        name: histogram(variant(original, fastmath=flag)) for name, flag in flags.items()}
    result["order_experiments"] = {
        ",".join(order): histogram(variant(original, fastmath=False, order=order))
        for order in itertools.permutations(("kx", "ky", "kz"))}
    candidate = variant(original, fastmath=False, candidate=True)
    result["explicit_fma_candidate"] = histogram(candidate)

    traced = variant(original, fastmath=True, trace=True)
    result["boundary_points"] = []
    result["trace_counts_equal_historical"] = []
    for shift_index, shift in enumerate(shifts):
        counts, baseline = np.zeros(500, dtype=int), np.zeros(500, dtype=int)
        spectrum = np.zeros(500)
        historical(spectrum, baseline, width, b, b@shift, psi, shape, 0, 0, 0)
        trace = traced(spectrum, counts, width, b, b@shift, psi, shape, 0, 0, 0)
        # Instrumentation must preserve the historical histogram. Also check
        # each reported point with its uninstrumented kernel on a one-voxel slab.
        np.testing.assert_array_equal(counts, baseline)
        result["trace_counts_equal_historical"].append(True)
        near = np.zeros(shape, dtype=bool)
        for boundary in (51, 75, 102, 150):
            # Selection of diagnostic rows only; never changes a bin assignment.
            near |= abs(trace[:, :, :, 5]-boundary) < 1e-10
        for index in np.argwhere(near):
            stages = trace[tuple(index)]
            single = np.zeros(500, dtype=int)
            historical(np.zeros(500), single, width, b, b@shift,
                     np.zeros((1, 1, 1), dtype=complex), shape, *index)
            assigned = int(np.flatnonzero(single)[0])
            assert assigned == int(stages[5])
            result["boundary_points"].append({
                "shift_number": shift_index, "shift": shift.tolist(),
                "grid_index": index.tolist(),
                "signed_index": np.where(index >= shape/2, index-shape, index).tolist(),
                "kx_ky_kz_ksquared_knorm_quotient": [format(x, ".17g") for x in stages],
                "bin": assigned,
            })
        del trace

    if args.candidate_replay:
        # This override exists only in this diagnostic process and is restored.
        # The CLI and pytest keep using the original production dispatcher.
        sf.jSOAS_spectrum_inplace = candidate
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                model = sf.StructureFactor(atoms, dx=1.0, fftmode="SCIPY")
                model.SOAS_build_structurefactor_fftw(nres=3)
            result["candidate_full_pipeline_vs_reference"] = compare(model.spectrum, ref)
        finally:
            sf.jSOAS_spectrum_inplace = original

    output = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(output)
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
