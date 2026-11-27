"""Reference-compatible rounding at reciprocal-shell boundaries."""

from pathlib import Path
import subprocess
import sys
import textwrap


def test_reference_boundary_families(serial_env, tmp_path):
    # Run the actual compiled production kernel without initializing MPI in
    # pytest. The same test works on machines without either optional backend.
    program = textwrap.dedent("""\
        import itertools
        from pathlib import Path
        import sys
        import numpy as np

        root = Path(sys.argv[1])
        sys.path.insert(0, str(root))
        from lib.readfile import ReadFile
        from lib.structurefactor import StructureFactor, jSOAS_spectrum_inplace

        atoms = ReadFile(root / 'lammpsfiles/vacanneal_slow_9_1800.relax')
        atoms.load()
        grid = np.array([128, 128, 126])
        model = StructureFactor(atoms, dx=1.0, fftmode='SCIPY')
        model.construct_basis_sets(atoms.cmat, grid)
        reference = np.loadtxt(root / 'tests/reference_output/svm_scipy_single.dat')
        width = 2 * reference[0, 0]
        assert width == np.linalg.norm(model.bmat.sum(axis=0)) / 3

        # These are a compatibility contract, not an epsilon or a universal
        # round-up rule. The prescribed FMA/add/sqrt/reciprocal-multiply order
        # rounds 3*17 and 3*34 to the upper shell, but 3*25 and 3*50 just below
        # their mathematical boundaries. Every future backend must agree.
        for index, expected_bin in ((17, 51), (25, 74), (34, 102), (50, 149)):
            for signs in itertools.product((-1, 1), repeat=3):
                signed_index = index * np.array(signs)
                offset = signed_index % grid
                counts = np.zeros(len(reference), dtype=np.int64)
                spectrum = np.zeros(len(reference))
                # A one-voxel slab exercises the production offset/wrapping
                # and assignment path for this exact reciprocal-space point.
                jSOAS_spectrum_inplace(
                    spectrum, counts, width, model.bmat, np.zeros(3),
                    np.full((1, 1, 1), 2 + 3j), grid, *offset,
                )
                expected_counts = np.zeros_like(counts)
                expected_counts[expected_bin] = 1
                message = f'index={signed_index}, array offset={offset}'
                np.testing.assert_array_equal(counts, expected_counts, err_msg=message)
                np.testing.assert_array_equal(spectrum, 13 * expected_counts, err_msg=message)
    """)
    result = subprocess.run(
        [sys.executable, "-c", program, str(Path(__file__).resolve().parents[1])],
        cwd=tmp_path, env=serial_env, text=True, capture_output=True, timeout=120,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
