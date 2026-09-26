#!/usr/bin/env python3
"""Temporary: print exact game.js lines (repr, indentation preserved) for anchors."""
import io
import re

GROUPS = {
    'motor': r'motorFactor',
    'rate': r'CHARS_PER_SEC|BASE_TYPING|typingRate',
    'typing': r'typeProgress|typedChars|currentComment|nextComment',
    'posted': r'commentsWritten|postCurrentComment|pickComment',
    'sim': r'createConnectomeSim|neuroSim =|getSummary\(\)|groupRates',
    'camblock': r'CAMERA CONTROLS \(OrbitControls\)|function safeSetRagdoll',
    'query': r'location\.search|URLSearchParams',
}

with io.open('game.js', 'r', encoding='utf-8') as fh:
    lines = fh.read().split('\n')

print('game.js lines: %d' % len(lines))
for name, pattern in GROUPS.items():
    rx = re.compile(pattern)
    hits = [(i + 1, ln) for i, ln in enumerate(lines) if rx.search(ln)]
    print('\n=== %s (%d hits) ===' % (name, len(hits)))
    for num, ln in hits[:40]:
        print('%4d|%s' % (num, ln.encode('unicode_escape').decode('ascii')))
