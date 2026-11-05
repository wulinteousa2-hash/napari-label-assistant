"""Record authorized coordinate-fix verification without replacing audit evidence."""
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'validation_reports/evidence/coordinate_patch'
OUT.mkdir(parents=True, exist_ok=True)
results = []
for label, stem in (
    ('full_suite', 'full-final'),
    ('coordinate_regressions', 'coordinates-final'),
    ('original_A01_A02_A04', 'original-probes'),
):
    source = Path(f'/tmp/nla-coordinate-patch-{stem}.xml')
    tree = ET.parse(source)
    cases = tree.findall('.//testcase')
    failed = sum(case.find('failure') is not None for case in cases)
    errors = sum(case.find('error') is not None for case in cases)
    skipped = sum(case.find('skipped') is not None for case in cases)
    results.append(dict(name=label, tests=len(cases), passed=len(cases)-failed-errors-skipped,
        failed=failed, errors=errors, skipped=skipped,
        seconds=sum(float(suite.get('time',0)) for suite in tree.findall('.//testsuite'))))
    for suffix in ('.xml', '.log'):
        shutil.copyfile(source.with_suffix(suffix), OUT/f'{label}{suffix}')
shutil.copyfile('/tmp/nla-coordinate-patch-targeted.log', OUT/'initial_targeted.log')
shutil.copyfile('/tmp/nla-coordinate-patch-run-audit.py', OUT/'run_original_probes.py')
observations = Path('/tmp/nla-coordinate-patch/audit-observations')
for path in observations.glob('*.json'):
    shutil.copyfile(path, OUT/path.name)
changed = subprocess.check_output(['git','diff','--name-only'],cwd=ROOT,text=True).splitlines()
paths = changed+['src/napari_label_assistant_tools/_coordinates.py','tests/test_coordinate_contract.py']
summary = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
    base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
    working_tree_patch=True, suites=results,
    source_file_sha256={name:sha256((ROOT/name).read_bytes()).hexdigest() for name in paths},
    notes=['Original audit logs/observations were preserved.',
           'One old positive-copy fixture was aligned to its source, replacing its formerly misaligned destination.',
           'Full suite followed by focused coordinate tests covers the final singleton-plane guard.',
           'ruff executable unavailable; no linter installation performed. git diff --check and Python parsing passed.'])
(OUT/'summary.json').write_text(json.dumps(summary,indent=2))
assert all(not row['failed'] and not row['errors'] for row in results)
print(json.dumps(results,indent=2))
