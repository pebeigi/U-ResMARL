"""Run the site adapter regressions in an isolated process."""
from pathlib import Path
import subprocess
import sys
import unittest


class TGSIMIsolationTests(unittest.TestCase):
    def check_site_contracts(self, case):
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run([sys.executable, str(root/case/'run_module.py'), 'verify', '-v'],
                                cwd=root, capture_output=True, text=True, timeout=300)
        self.assertEqual(result.returncode, 0, result.stdout+'\n'+result.stderr)
        self.assertNotIn('Ran 0 tests', result.stderr)
        self.assertIn('test_environment_and_benchmark_share_utility_transitions', result.stderr)

    def test_site_contracts(self):
        self.check_site_contracts('TGSIM Case')

    def test_roundabout_contracts(self):
        self.check_site_contracts('Roundabout Case')
