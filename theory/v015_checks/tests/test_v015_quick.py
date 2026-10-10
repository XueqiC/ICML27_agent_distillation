"""Standalone quick integration test; never imports the repository package."""
import json
import os
from pathlib import Path
import subprocess
import sys


def test_v015_quick():
    here=Path(__file__).resolve().parents[1]
    output=here/'quick'/'test_results.json'
    env=os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',
               MPLCONFIGDIR=str(here/'.mplconfig'),MPLBACKEND='Agg',
               PYTEST_DISABLE_PLUGIN_AUTOLOAD='1')
    process=subprocess.run([sys.executable,str(here/'check_v015.py'),'--quick','--output',str(output)],
        env=env,cwd=here.parents[1],capture_output=True,text=True,timeout=300)
    assert output.exists(),process.stdout+process.stderr
    report=json.loads(output.read_text())
    assert {r['id'] for r in report['groups']}=={*(f'V{i}' for i in range(1,8)),*(f'E{i}' for i in range(10))}
    failures=[r['id'] for r in report['groups'] if r['status']=='FAIL']
    assert not failures,json.dumps(report['counterexamples'],indent=2)
    assert process.returncode==0,process.stdout+process.stderr
    assert not report['counterexamples']
