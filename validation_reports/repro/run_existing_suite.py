"""Run the unmodified repository suite with isolated settings and raw evidence."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from time import monotonic
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'validation_reports' / 'evidence'
OUT.mkdir(parents=True, exist_ok=True)
env = dict(os.environ)
env.update(QT_QPA_PLATFORM='offscreen', PYTHONDONTWRITEBYTECODE='1',
           XDG_CONFIG_HOME='/tmp/nla-audit-cb95a5e/config', XDG_CACHE_HOME='/tmp/nla-audit-cb95a5e/cache',
           MPLCONFIGDIR='/tmp/nla-audit-cb95a5e/mpl', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
env['PYTHONPATH'] = '/tmp/label-assistant-test-deps:' + str(ROOT / 'src')
collection = [sys.executable, '-m', 'pytest', 'tests', '--collect-only', '-q', '-p', 'no:cacheprovider']
with (OUT / 'existing_collection.log').open('w') as stream:
    collect_result = subprocess.run(collection, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
command = [sys.executable, '-m', 'pytest', 'tests', '-q', '-p', 'no:cacheprovider',
           '--basetemp=/tmp/nla-audit-cb95a5e/existing-tests',
           '--junitxml=' + str(OUT / 'existing_suite.xml'), '--durations=20']
started = datetime.now(timezone.utc).isoformat()
clock = monotonic()
with (OUT / 'existing_suite.log').open('w') as stream:
    result = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
summary = dict(started_utc=started, elapsed_seconds=monotonic()-clock,
               collection_command=collection, collection_exit_code=collect_result.returncode,
               command=command, exit_code=result.returncode,
               environment_overrides={k: env[k] for k in ('QT_QPA_PLATFORM', 'PYTHONPATH', 'PYTHONDONTWRITEBYTECODE',
                   'XDG_CONFIG_HOME', 'XDG_CACHE_HOME', 'MPLCONFIGDIR', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')})
if (OUT / 'existing_suite.xml').exists():
    cases = list(ET.parse(OUT / 'existing_suite.xml').getroot().iter('testcase'))
    summary.update(total_cases=len(cases), failed=sum(c.find('failure') is not None for c in cases),
                   errors=sum(c.find('error') is not None for c in cases), skipped=sum(c.find('skipped') is not None for c in cases))
    summary['passed'] = len(cases)-summary['failed']-summary['errors']-summary['skipped']
(OUT / 'existing_suite_summary.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
sys.exit(result.returncode)
