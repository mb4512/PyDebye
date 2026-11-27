"""The four commands from single_vs_multi.sh, without shared output files."""

from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.integration
@pytest.mark.mpi
@pytest.mark.parametrize("backend,ranks,reference", [
    pytest.param("SCIPY", 1, "svm_scipy_single.dat", id="scipy-1-rank"),
    pytest.param("SCIPY", 4, "svm_scipy_multi.dat", id="scipy-4-ranks"),
    pytest.param("FFTW", 1, "svm_fftw_single.dat", id="fftw-1-rank", marks=pytest.mark.fftw),
    pytest.param("FFTW", 4, "svm_fftw_multi.dat", id="fftw-4-ranks", marks=pytest.mark.fftw),
])
def test_reference_spectrum(backend, ranks, reference, run_cli, mpi_launcher, require_backend):
    require_backend("mpi4py.MPI")
    if backend == "FFTW":
        require_backend("mpi4py_fft")
    workdir, _ = run_cli(
        [ROOT / "lammpsfiles/vacanneal_slow_9_1800.relax", "-ft", "data",
         "-m", "fft", "-fm", backend, "-dx", "1.0", "-nr", "3",
         "-sx", "spectrum.dat"],
        launcher=[*mpi_launcher, "-n", str(ranks)], timeout=600,
    )
    expected = np.loadtxt(ROOT / "tests/reference_output" / reference, ndmin=2)
    actual = np.loadtxt(workdir / "spectrum.dat", ndmin=2)

    assert expected.shape[1] == 3, "reference columns must be k, intensity, shell count"
    assert actual.shape == expected.shape, "spectrum row/column layout changed"
    assert np.isfinite(actual).all(), "spectrum contains NaN or infinity"
    assert np.all(np.diff(actual[:, 0]) > 0), "reciprocal grid must increase"
    # Grid arithmetic gets only roundoff allowance, independently of intensity.
    np.testing.assert_allclose(actual[:, 0], expected[:, 0], rtol=0, atol=1e-14,
                               err_msg="reciprocal-space grid changed")
    for counts in (expected[:, 2], actual[:, 2]):
        assert np.all(counts >= 0), "shell counts must be nonnegative"
        np.testing.assert_array_equal(counts, np.rint(counts),
                                      err_msg="shell counts must be integers")
    np.testing.assert_array_equal(actual[:, 2], expected[:, 2],
                                  err_msg="integer shell populations changed")
    np.testing.assert_allclose(actual[:, 1], expected[:, 1], rtol=1e-10, atol=1e-12,
                               err_msg="spectrum intensities changed")
