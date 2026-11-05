"""Read-only preflight for the corrected shared pipeline.

Run with the Python environment intended for training. Does not train or import
TensorFlow. A separate explicit smoke-test command exercises tiny real models.
"""
import ast
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def check():
    results = []
    for path in sorted((ROOT / 'code').glob('*.py')):
        try:
            ast.parse(path.read_text(encoding='utf-8-sig'))
            results.append(dict(check='syntax:' + path.name, status='PASS'))
        except SyntaxError as exc:
            results.append(dict(check='syntax:' + path.name, status='FAIL', detail=str(exc)))
    for package in ['tensorflow', 'numpy', 'PIL', 'matplotlib', 'pywt']:
        try:
            present = importlib.util.find_spec(package) is not None
        except (ImportError, ValueError):
            present = False
        results.append(dict(check='dependency:' + package, status='PASS' if present else 'BLOCKED'))
    # Unit discovery skips the expensive runtime smoke test by explicit environment gate.
    proc = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', str(ROOT / 'tests'), '-v'],
                          cwd=ROOT, capture_output=True, text=True, env=dict(os.environ, PD_RUN_TF_SMOKE="0"))
    results.append(dict(check='pipeline-contract-tests', status='PASS' if proc.returncode == 0 else 'FAIL',
                        detail=proc.stdout + proc.stderr))
    bad = any(r['status'] == 'FAIL' for r in results)
    blocked = any(r['status'] == 'BLOCKED' for r in results)
    return dict(python=sys.executable, checks=results,
                verdict='FAILED' if bad else ('STATIC_CHECKS_PASS_RUNTIME_ENVIRONMENT_BLOCKED' if blocked else 'STATIC_CHECKS_PASS_RUN_EXPLICIT_SMOKE_TEST'),
                limitation='No guarantee of full-training runtime correctness or acquisition independence. Run python code/model_training_vgg16.py smoke-test in the configured environment.')


if __name__ == '__main__':
    report = check()
    for result in report['checks']:
        print(result['status'] + ': ' + result['check'])
    print(report['verdict'])
    if '--json' in sys.argv:
        print(json.dumps(report, indent=2))
    raise SystemExit(1 if report['verdict'] in ['FAILED', 'STATIC_CHECKS_PASS_RUNTIME_ENVIRONMENT_BLOCKED'] else 0)
