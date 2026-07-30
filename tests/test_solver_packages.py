from pathlib import Path
import subprocess
import sys
import unittest

from lib import solvers
from lib.solvers import cbqm, mis, qubo


class SolverPackageTests(unittest.TestCase):
    def test_exact_qubo_import_does_not_load_optional_dependencies(self):
        project_root = Path(__file__).resolve().parents[1]
        script = (
            'import sys\n'
            'from lib.solvers.qubo import ExactQuboSolver\n'
            'loaded = [name for name in sys.modules '
            "if name in {'numpy', 'pandas'} "
            "or name.startswith(('numpy.', 'pandas.'))]\n"
            'assert not loaded, loaded\n'
            'assert ExactQuboSolver.__name__ == "ExactQuboSolver"\n'
        )

        completed = subprocess.run(
            [sys.executable, '-c', script],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(
            0,
            completed.returncode,
            msg=completed.stdout + completed.stderr,
        )

    def test_exports_representation_specific_subpackages(self):
        self.assertEqual(['cbqm', 'mis', 'qubo'], solvers.__all__)
        self.assertIs(solvers.cbqm, cbqm)
        self.assertIs(solvers.mis, mis)
        self.assertIs(solvers.qubo, qubo)

    def test_subpackages_export_only_implemented_solvers(self):
        self.assertEqual(
            ['BaseCbqmSolver', 'CbqmSolveOutcome'],
            cbqm.__all__,
        )
        self.assertEqual([], mis.__all__)
        self.assertEqual(
            [
                'BaseQuboSolver',
                'ExactQuboSolver',
                'QaoaQuboSolver',
                'QuboSolveOutcome',
                'SimulatedAnnealingQuboSolver',
            ],
            qubo.__all__,
        )


if __name__ == '__main__':
    unittest.main()
