"""Subprocess isolation and optional dependency checks for CLI tests."""

import itertools
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import textwrap

import pytest


ROOT = Path(__file__).resolve().parents[1]


def pytest_addoption(parser):
    parser.addoption(
        "--run-integration", action="store_true",
        help="run the optional one/four-rank reference regressions",
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-integration"):
        skip = pytest.mark.skip(reason="use --run-integration to run MPI reference cases")
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip)


@pytest.fixture
def cli_env():
    env = os.environ.copy()
    # Keep each MPI rank from starting its own pool of numerical-library threads.
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMBA_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env[name] = "1"
    env["PYTHONNOUSERSITE"] = "1"
    return env


@pytest.fixture
def serial_env(tmp_path, cli_env):
    """Make optional imports unavailable even on an MPI-equipped developer machine.

    This is an import failure, not a mock MPI communicator: the real serial
    fallback and numerical routines must execute in the child interpreter.
    """
    startup = tmp_path / "without_optional_backends"
    startup.mkdir()
    (startup / "sitecustomize.py").write_text(textwrap.dedent("""\
        import sys
        from importlib.abc import MetaPathFinder

        class UnavailableOptionalBackend(MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in {'mpi4py', 'mpi4py_fft'}:
                    raise ModuleNotFoundError(
                        fullname + ' disabled by PyDebye serial tests', name=fullname
                    )

        sys.meta_path.insert(0, UnavailableOptionalBackend())
    """))
    cli_env["PYTHONPATH"] = str(startup)
    return cli_env


@pytest.fixture
def run_cli(tmp_path, cli_env):
    sequence = itertools.count()

    def run(arguments, *, env=None, launcher=(), timeout=120, expected_code=0):
        # Every invocation, including histogram reloads, gets an empty directory.
        workdir = tmp_path / f"run-{next(sequence)}"
        workdir.mkdir()
        command = [*launcher, sys.executable, str(ROOT / "debyescherrer.py"),
                   *map(str, arguments)]
        result = subprocess.run(
            command, cwd=workdir, env=cli_env if env is None else env,
            text=True, capture_output=True, timeout=timeout,
        )
        assert result.returncode == expected_code, (
            f"Command: {shlex.join(command)}\nExit code: {result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
        return workdir, result

    return run


@pytest.fixture
def write_dump(tmp_path):
    def write(positions, box=8.0):
        path = tmp_path / "atoms.dump"
        header = (
            f"ITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\n{len(positions)}\n"
            f"ITEM: BOX BOUNDS pp pp pp\n0 {box}\n0 {box}\n0 {box}\n"
            "ITEM: ATOMS id type x y z\n"
        )
        atoms = "".join(
            f"{i} 1 {x:.17g} {y:.17g} {z:.17g}\n"
            for i, (x, y, z) in enumerate(positions, start=1)
        )
        path.write_text(header + atoms)
        return path

    return write


@pytest.fixture
def mpi_launcher():
    command = shlex.split(os.environ.get("MPIEXEC", "mpirun"))
    if not command or shutil.which(command[0]) is None:
        pytest.skip("MPI launcher unavailable; install MPI or set MPIEXEC")
    command += shlex.split(os.environ.get("MPIEXEC_FLAGS", ""))
    return command


@pytest.fixture
def require_backend(cli_env, tmp_path):
    def require(module):
        # Probe in a child: importing MPI in the pytest process can interfere
        # with launching a separate MPI job on some implementations.
        result = subprocess.run(
            [sys.executable, "-c", f"import {module}"], cwd=tmp_path,
            env=cli_env, text=True, capture_output=True, timeout=30,
        )
        if result.returncode:
            pytest.skip(f"{module} unavailable: {result.stderr.strip()}")

    return require
