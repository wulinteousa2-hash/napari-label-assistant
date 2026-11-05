# Version 5.3.0 release validation

Date: 2026-11-05. Base SHA: `cb95a5e97717aca24db658c488fd810647955bcd`.
This report records the final release working tree before its release commit.
Package and runtime versions are both 5.3.0, a +1.0.0 increment from 4.3.0.

Executed full suite: **353 passed, 0 failed, 0 skipped**, 353 warnings,
250.88 seconds. Raw [log](evidence/release_5_3_0/nla-release-530.log) and
[JUnit XML](evidence/release_5_3_0/nla-release-530.xml) are retained. Warnings
remain visible in the log; passing tests do not close every original audit finding.

Environment: Python 3.11, napari 0.9.1, PySide6 6.11.2, NumPy 2.4.6,
Zarr 3.1.6, Linux ARM64, Qt offscreen. Command:

```bash
env QT_QPA_PLATFORM=offscreen PYTHONPATH=/tmp/label-assistant-test-deps:src \
 PYTHONDONTWRITEBYTECODE=1 XDG_CONFIG_HOME=/tmp/nla-copy-match/config \
 XDG_CACHE_HOME=/tmp/nla-copy-match/cache MPLCONFIGDIR=/tmp/nla-copy-match/mpl \
 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
 /home/wteox/miniforge3/envs/napari-dev2/bin/python -m pytest \
 -q -p no:cacheprovider --basetemp=/tmp/nla-release-530 \
 --junitxml=/tmp/nla-release-530.xml
git diff --check
```

README and CHANGELOG now describe the released multilabel, managed-grid editing,
coordinate preservation and confirmed/recoverable copy workflows. Historical
release entries and original audit evidence retain their dates and original
results. See reports 08–10 for detailed contracts and remaining limitations.
No user datasets were modified. Offscreen tests are not a human usability study
or an 84k-image workstation benchmark. No new runtime dependency was introduced.
