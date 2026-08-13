#!/usr/bin/env python3
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
MANIFEST = json.loads((ROOT / 'REPOSITORY_CONTENTS.json').read_text())
errors: list[str] = []

archive = ROOT / MANIFEST['original_v2_archive']
if not archive.is_file():
    errors.append(f'missing original archive: {archive.relative_to(ROOT)}')
else:
    got = hashlib.sha256(archive.read_bytes()).hexdigest()
    if got != MANIFEST['original_v2_archive_sha256']:
        errors.append(f'archive hash mismatch: expected {MANIFEST["original_v2_archive_sha256"]}, got {got}')

expected_paths = {entry['path'] for entry in MANIFEST['files']}
actual_paths = {
    p.relative_to(ROOT / 'v2_baseline').as_posix()
    for p in (ROOT / 'v2_baseline').rglob('*') if p.is_file()
}
for missing in sorted(expected_paths - actual_paths):
    errors.append(f'missing V2 file: v2_baseline/{missing}')
for extra in sorted(actual_paths - expected_paths):
    errors.append(f'unexpected V2 file: v2_baseline/{extra}')

for entry in MANIFEST['files']:
    p = ROOT / 'v2_baseline' / entry['path']
    if not p.is_file():
        continue
    data = p.read_bytes()
    if len(data) != entry['bytes']:
        errors.append(f'byte-count mismatch: v2_baseline/{entry["path"]}')
    got = hashlib.sha256(data).hexdigest()
    if got != entry['sha256']:
        errors.append(f'hash mismatch: v2_baseline/{entry["path"]}')

if errors:
    print('SELF_CONTAINED_HANDOFF_VERIFICATION: FAILED')
    for e in errors:
        print(f' - {e}')
    sys.exit(1)

print('SELF_CONTAINED_HANDOFF_VERIFICATION: PASSED')
print(f'V2 files verified: {MANIFEST["v2_file_count"]}')
print(f'V2 file bytes verified: {MANIFEST["v2_total_file_bytes"]}')
print(f'Original V2 archive SHA-256: {MANIFEST["original_v2_archive_sha256"]}')
