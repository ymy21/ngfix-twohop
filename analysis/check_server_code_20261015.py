#!/usr/bin/env python3
"""Check that the complete local source snapshot reached the server."""
import hashlib
import json
from pathlib import Path
import sys


def digest(path):
    # Allow CRLF/LF conversion by Windows/Linux transfer tools.
    return hashlib.sha256(path.read_bytes().replace(b'\r\n',b'\n')).hexdigest()


def main():
    root=Path(__file__).resolve().parents[1]
    manifest=root/'analysis/server_code_manifest_20261015.json'
    data=json.loads(manifest.read_text(encoding='utf-8'))
    missing=[];different=[]
    for item in data['files']:
        path=root/item['path']
        if not path.is_file():missing.append(item['path'])
        elif digest(path)!=item['sha256_lf']:different.append(item['path'])
    for name in missing:print('MISSING:',name)
    for name in different:print('DIFFERENT:',name)
    if missing or different:
        print('FAIL: copy the listed files from the local NGFix source again.')
        return 1
    print('PASS: all %d source files match the local snapshot (LF-normalized).'%len(data['files']))
    return 0


if __name__=='__main__':sys.exit(main())
