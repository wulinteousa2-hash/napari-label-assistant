# Commands and reproducibility

Reviewed SHA: `cb95a5e97717aca24db658c488fd810647955bcd`. Commands below were executed from the repository root. Paths identify this machine; use the equivalent environment when reproducing elsewhere. Outputs are stored under `validation_reports/evidence/`. No git commit/push, source repair, dependency installation, or production-dataset mutation was performed.

```bash
git rev-parse HEAD
git status --short
git ls-files
git ls-remote --heads origin main
```

Tracked source/docs/tests were inspected using `rg`, `rg --files`, `sed`, AST indexing and SHA256 hashing. README, CHANGELOG, pyproject, plugin manifest, docs and test definitions are included in the inventory. Function-level references are in source_index.md; targeted `sed` reads examined the high-risk branches. Public GitHub landing-page content was also opened, but exact revision validation uses Git rather than a cached web-page version.

Common executed test environment:

```bash
export QT_QPA_PLATFORM=offscreen
export PYTHONPATH=/tmp/label-assistant-test-deps:src
export PYTHONDONTWRITEBYTECODE=1
export XDG_CONFIG_HOME=/tmp/nla-audit-cb95a5e/config
export XDG_CACHE_HOME=/tmp/nla-audit-cb95a5e/cache
export MPLCONFIGDIR=/tmp/nla-audit-cb95a5e/mpl
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
AUDIT_PYTHON=/home/wteox/miniforge3/envs/napari-dev2/bin/python

"$AUDIT_PYTHON" validation_reports/repro/collect_inventory.py
"$AUDIT_PYTHON" validation_reports/repro/run_existing_suite.py

"$AUDIT_PYTHON" -m pytest validation_reports/repro/test_independent_audit.py -q -p no:cacheprovider --basetemp=/tmp/nla-audit-cb95a5e/independent-tests --junitxml=validation_reports/evidence/independent_suite.xml
"$AUDIT_PYTHON" -m pytest validation_reports/repro/test_synthetic_workflow.py -q -p no:cacheprovider --basetemp=/tmp/nla-audit-cb95a5e/synthetic-tests-final --junitxml=validation_reports/evidence/synthetic_suite.xml
"$AUDIT_PYTHON" -m pytest validation_reports/repro/test_supplemental_audit.py -q -p no:cacheprovider --basetemp=/tmp/nla-audit-cb95a5e/supplemental-tests --junitxml=validation_reports/evidence/supplemental_suite.xml
"$AUDIT_PYTHON" -m pytest validation_reports/repro/test_cuda_smoke.py -q -p no:cacheprovider --basetemp=/tmp/nla-audit-cb95a5e/cuda-tests-diagnostics --junitxml=validation_reports/evidence/cuda_suite.xml
"$AUDIT_PYTHON" validation_reports/repro/benchmark.py
"$AUDIT_PYTHON" validation_reports/repro/finalize_reports.py
```

The wrapper also executed `python -m pytest tests --collect-only -q -p no:cacheprovider` and the full existing command recorded in existing_suite_summary.json (including `--durations=20`). stdout/stderr were redirected/captured into the corresponding logs. The independent/supplemental suites are expected to return exit code 1 on the reviewed source; that is audit evidence, not a command-execution failure to hide.

Historical runs: synthetic initial `--basetemp=.../synthetic-tests`; GPU initial `--basetemp=.../cuda-tests`; benchmark initial controller import failure. The audit harness was corrected, application source was not. Preserved logs identify each failure. Performance was repeated after separating import time from operation time; published measurements use the final script/output.

Re-run individual findings with `-k A01`, `-k A03`, etc. Run all added probes with `python -m pytest validation_reports/repro -q -p no:cacheprovider --basetemp=/tmp/nla-audit-rerun`; use a new basetemp if preserving original disposable test files. Existing test selection remains `tests/`, so future default test runs will not automatically include these independent regression proposals.

Array checksums and final tracked-file verification are in repository_integrity.json. No coverage-instrumentation, build/wheel install, real-desktop user study or massive dense allocation command was executed.
