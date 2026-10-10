"""Independent quick-suite gate: scientific failures must remain visible.

Run directly with theory/.venv/bin/python (stdlib unittest; no installation).
The test is also discoverable by pytest. C15 is deliberately not exempted.
"""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
CHECKS = ROOT / 'theory' / 'independent_checks'
PYTHON = ROOT / 'theory' / '.venv' / 'bin' / 'python'


class TestIndependentTheoryQuick(unittest.TestCase):
    def test_quick_has_no_failed_claims(self):
        # Keep even temporary outputs inside the authorized directory.
        with tempfile.TemporaryDirectory(prefix='quick-test-', dir=CHECKS) as directory:
            output = Path(directory) / 'quick.json'
            env = dict(os.environ)
            env['CUDA_VISIBLE_DEVICES'] = ''
            env['PYTHONDONTWRITEBYTECODE'] = '1'
            completed = subprocess.run(
                [str(PYTHON), str(CHECKS / 'check_v014_independent.py'),
                 '--quick', '--output', str(output)],
                cwd=ROOT, env=env, capture_output=True, text=True, timeout=300,
                check=False,
            )
            self.assertTrue(output.exists(), completed.stdout + completed.stderr)
            report = json.loads(output.read_text())
            self.assertTrue(report['complete'])
            rows = report['claims'] + [report['controlled_neural']]
            self.assertEqual([row['id'] for row in rows], [f'C{i}' for i in range(1, 24)])
            self.assertTrue(all(row['n_instances'] > 0 for row in rows))
            failures = [row['id'] for row in rows if row['status'] == 'FAIL']
            # Never suppress the coverage counterexample to make this test pass.
            self.assertEqual(failures, [], json.dumps(report['counterexamples'], indent=2))
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)


if __name__ == '__main__':
    unittest.main()
