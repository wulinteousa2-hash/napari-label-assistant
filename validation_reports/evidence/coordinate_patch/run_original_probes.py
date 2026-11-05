"""Re-run three original audit probes without overwriting historical evidence."""
from pathlib import Path
import pytest


class PreserveEvidence:
    def pytest_collection_modifyitems(self, items):
        destination = Path('/tmp/nla-coordinate-patch/audit-observations')
        destination.mkdir(parents=True, exist_ok=True)
        for item in items:
            if hasattr(item.module, 'OUT'):
                item.module.OUT = destination


raise SystemExit(pytest.main([
    'validation_reports/repro/test_independent_audit.py',
    '-k', 'A01 or A02 or A04', '-q', '-p', 'no:cacheprovider',
    '--basetemp=/tmp/nla-coordinate-patch/original-audit-probes',
    '--junitxml=/tmp/nla-coordinate-patch-original-probes.xml',
], plugins=[PreserveEvidence()]))
