#!/usr/bin/env python3
"""Temporary: print exact game.js line ranges (indentation preserved)."""
import io

RANGES = [(1284, 1300), (1340, 1440), (1136, 1150), (1300, 1322), (1540, 1570), (86, 96)]

with io.open('game.js', 'r', encoding='utf-8') as fh:
    lines = fh.read().split('\n')

print('game.js lines: %d' % len(lines))
for start, end in RANGES:
    print('\n=== %d..%d ===' % (start, end))
    for num in range(start, min(end, len(lines)) + 1):
        ln = lines[num - 1]
        print('%4d|%s' % (num, ln.encode('unicode_escape').decode('ascii')))
