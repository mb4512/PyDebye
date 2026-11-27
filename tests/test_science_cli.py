"""Small analytical checks; no MPI runtime or FFTW installation is needed."""

import itertools
import subprocess
import sys

import numpy as np


def test_two_atom_histogram_round_trip(write_dump, run_cli, serial_env):
    # The separation sits at the centre of bin 123, far from either bin edge.
    separation, dr = 1.235, 0.01
    source = write_dump([(1.0, 1.0, 1.0), (1.0 + separation, 1.0, 1.0)])
    grid_args = ["-smin", "0.1", "-smax", "0.8", "-spts", "8"]
    workdir, _ = run_cli(
        [source, "-m", "real", "-dr", str(dr), "-hx", "pairs.hist",
         "-sx", "direct.spec", *grid_args], env=serial_env,
    )
    histogram = np.loadtxt(workdir / "pairs.hist", ndmin=2)
    assert histogram.shape[1] == 2
    np.testing.assert_allclose(histogram[:, 0], (np.arange(len(histogram)) + 0.5) * dr,
                               rtol=0, atol=1e-14)
    expected_counts = np.zeros(len(histogram), dtype=int)
    expected_counts[0] = 2  # two self pairs
    expected_counts[123] = 2  # ordered pairs (1, 2) and (2, 1)
    np.testing.assert_array_equal(histogram[:, 1], expected_counts)
    assert histogram[123, 0] == separation

    spectrum = np.loadtxt(workdir / "direct.spec", ndmin=2)
    s = np.linspace(0.1, 0.8, 8)
    assert spectrum.shape == (8, 2)
    np.testing.assert_allclose(spectrum[:, 0], s, rtol=0, atol=1e-15)
    # PyDebye places even the self pairs at the first bin midpoint dr/2.
    # np.sinc(x) = sin(pi*x)/(pi*x); no production routines form this oracle.
    expected = 2 * np.sinc(s * dr) + 2 * np.sinc(2 * s * separation)
    np.testing.assert_allclose(spectrum[:, 1], expected, rtol=1e-10, atol=1e-12)
    # Also bound the histogram approximation against the exact two-atom Debye law.
    exact = 2 + 2 * np.sinc(2 * s * separation)
    assert np.all(np.abs(spectrum[:, 1] - exact) <= (np.pi * s * dr)**2 / 3 + 1e-12)

    reloaded_dir, _ = run_cli(
        [workdir / "pairs.hist", "-ft", "hist", "-sx", "reloaded.spec", *grid_args],
        env=serial_env,
    )
    reloaded = np.loadtxt(reloaded_dir / "reloaded.spec", ndmin=2)
    assert reloaded.shape == spectrum.shape
    np.testing.assert_array_equal(reloaded[:, 0], spectrum[:, 0])
    np.testing.assert_allclose(reloaded[:, 1], expected, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(reloaded[:, 1], spectrum[:, 1], rtol=1e-10, atol=1e-12)


def test_bcc_vector_reflections(write_dump, run_cli, serial_env):
    a, cells = 3.1652, 4
    origins = np.array(list(itertools.product(range(cells), repeat=3)), dtype=float)
    positions = a * np.concatenate([origins, origins + 0.5])
    source = write_dump(positions, box=cells * a)
    workdir, _ = run_cli(
        [source, "-m", "fft", "-fm", "SCIPY", "-dx", str(a / 7.75), "-ns", "0",
         "-ediff", "bcc", "-ekmin", "0.01", "-ekmax", "0.6", "-ith", "-1"],
        env=serial_env,
    )
    # -ns 0 samples reciprocal lattice vectors exactly. -ith -1 retains zero
    # intensities, so an absent row cannot masquerade as an extinct reflection.
    path = workdir / "bcc.0.xyz"
    spots = np.loadtxt(path, skiprows=2, ndmin=2)
    assert spots.shape[1] == 4
    assert int(path.read_text().splitlines()[0]) == len(spots)
    assert np.isfinite(spots).all()

    # Eight voxels per unit cell put every atom exactly on a voxel. The
    # normalized, 3-sigma-truncated Gaussian kernel has weights 1 at its centre,
    # exp(-2) at six axial neighbours, and exp(-4) at twelve face diagonals.
    # Its Fourier multiplier at (110) is the following closed-form expression.
    c = np.cos(2 * np.pi / 8)
    u, v = np.exp(-2), np.exp(-4)
    window_110 = (1 + 2*u*(2*c + 1) + 4*v*(c*c + 2*c)) / (1 + 6*u + 12*v)
    # BCC basis: F(hkl) = 1 + (-1)**(h+k+l). Thus I110 = N^2 before
    # voxel smoothing and I100 = 0, independently of powder-shell averages.
    allowed_intensity = len(positions)**2 * window_110**2
    for hkl in ((1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 0, 0), (0, 1, 0), (0, 0, 1)):
        for sign in (-1, 1):
            target = sign * np.array(hkl) / a
            rows = spots[np.all(np.isclose(spots[:, :3], target, rtol=0, atol=1e-9), axis=1)]
            assert len(rows) == 1, f"expected exactly one row at k={target}"
            if sum(hkl) % 2 == 0:
                # Diffraction export prints nine significant figures.
                np.testing.assert_allclose(rows[0, 3], allowed_intensity, rtol=1e-8, atol=1e-12)
            else:
                assert abs(rows[0, 3]) < len(positions)**2 * 1e-24, f"(100) extinction lost at {target}"


def test_serial_cli_without_optional_backends(write_dump, run_cli, serial_env, tmp_path):
    # Confirm the child interpreter really cannot import MPI, even if installed.
    probe = subprocess.run(
        [sys.executable, "-c", "import mpi4py"], cwd=tmp_path, env=serial_env,
        text=True, capture_output=True, timeout=30,
    )
    assert probe.returncode != 0
    assert "mpi4py disabled by PyDebye serial tests" in probe.stderr

    source = write_dump([(1.0, 1.0, 1.0)])
    workdir, _ = run_cli(
        [source, "-m", "real", "-dr", "0.01", "-smin", "0.1", "-smax", "0.8", "-spts", "8"],
        env=serial_env,
    )
    spectrum = np.loadtxt(workdir / "spectrum.dat", ndmin=2)
    assert spectrum.shape == (8, 2)
    s = np.linspace(0.1, 0.8, 8)
    np.testing.assert_allclose(spectrum[:, 0], s, rtol=0, atol=1e-15)
    np.testing.assert_allclose(spectrum[:, 1], np.sinc(s * 0.01), rtol=1e-10, atol=1e-12)
    assert (workdir / "histogram.dat").is_file()

    failed_dir, failed = run_cli(
        [source, "-m", "fft", "-fm", "FFTW"], env=serial_env, expected_code=2,
    )
    assert "FFTW" in failed.stderr
    assert "mpi4py-fft" in failed.stderr
    assert "install" in failed.stderr.lower()
    assert "-fm SCIPY" in failed.stderr
    assert "Traceback" not in failed.stderr
    assert not (failed_dir / "spectrum.dat").exists()
