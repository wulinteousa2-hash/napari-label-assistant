"""Archive implementation verification separately from the original audit."""
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'validation_reports/evidence/grid_tiles'
OUT.mkdir(parents=True,exist_ok=True)
results=[]
for name,stem in (('full_suite','full-final'),('final_widgets','final-ui'),('library_workspace_integration','integration-final')):
    source=Path(f'/tmp/nla-mask-tiles-{stem}.xml')
    tree=ET.parse(source)
    cases=tree.findall('.//testcase')
    failed=sum(case.find('failure') is not None for case in cases)
    errors=sum(case.find('error') is not None for case in cases)
    skipped=sum(case.find('skipped') is not None for case in cases)
    results.append(dict(name=name,collected=len(cases),passed=len(cases)-failed-errors-skipped,
                        failed=failed,errors=errors,skipped=skipped,
                        seconds=sum(float(s.get('time',0)) for s in tree.findall('.//testsuite'))))
    for suffix in ('.log','.xml'):
        shutil.copyfile(source.with_suffix(suffix),OUT/f'{name}{suffix}')
shutil.copyfile('/tmp/nla-grid-opening-benchmark.log',OUT/'opening_memory.log')
changed=subprocess.check_output(['git','diff','--name-only'],cwd=ROOT,text=True).splitlines()
paths=changed+['src/napari_label_assistant_tools/_mask_tiles.py','tests/test_mask_tiles.py']
summary=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
             base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
             uncommitted_working_tree=True,suites=results,
             changed_file_sha256={name:sha256((ROOT/name).read_bytes()).hexdigest() for name in paths},
             notes=['No production masks were opened or edited during implementation testing.',
                    'Final widget run covers compact-control visibility changes after the full suite.',
                    'Sparse local mask-only opening measurement is not a real low-end desktop benchmark.',
                    'Original audit evidence is preserved in its existing locations.'])
(OUT/'summary.json').write_text(json.dumps(summary,indent=2))
assert all(not item['failed'] and not item['errors'] for item in results)
print(json.dumps(results,indent=2))
