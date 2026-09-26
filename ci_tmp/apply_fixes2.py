#!/usr/bin/env python3
"""Temporary one-shot patcher.

1. Reindent the camera/debug block inserted earlier from 4/8-space to the
   file's 2/4-space convention.
2. Make typing independent of neural-sim health: missing / zero / NaN motor
   rate yields motorFactor 1, clamped to MOTOR_FACTOR_MIN..MOTOR_FACTOR_MAX.
3. Guard every NaN path in the typing accumulator so a comment always lands.
4. Add a ?nosim=1 URL flag that skips the connectome sim (smoke runs only).

Every anchor must match exactly one line or the script exits non-zero without
writing anything.
"""
import io
import sys

PATH = 'game.js'

with io.open(PATH, 'r', encoding='utf-8') as fh:
    lines = fh.read().split('\n')

print('read %s: %d lines' % (PATH, len(lines)))

# ---------------------------------------------------------------- reindent
HDR = 1139
END = 1317
SAFE = 1319

problems = []


def guard(idx1, pred, label):
    actual = lines[idx1 - 1] if idx1 - 1 < len(lines) else None
    if actual is None or not pred(actual):
        problems.append('line %d (%s): %r' % (idx1, label, actual))


guard(HDR, lambda l: l.strip().startswith('// ====='), 'block header rule')
guard(HDR + 1,
      lambda l: l.strip() == '// CAMERA CONTROLS (OrbitControls) + debug hook',
      'block title')
guard(END, lambda l: l.strip() == '}', 'block end brace')
guard(SAFE, lambda l: l == 'function safeSetRagdoll(weight) {',
      'safeSetRagdoll at column 0')

if problems:
    print('REINDENT GUARDS FAILED:')
    for p in problems:
        print('  ' + p)
    sys.exit(1)

for i in range(HDR - 1, END):
    ln = lines[i]
    if not ln.strip():
        lines[i] = ''
        continue
    n = len(ln) - len(ln.lstrip(' '))
    if n >= 4:
        lines[i] = ' ' * (2 * max(1, int(n / 4.0 + 0.5))) + ln.lstrip(' ')
lines[HDR - 1] = '  ' + lines[HDR - 1].lstrip(' ')
lines[SAFE - 1] = '  ' + lines[SAFE - 1]
print('reindented lines %d..%d to 2-space style' % (HDR, END))

# ------------------------------------------------------------------- edits
EDITS = [
    ('const BASE_TYPING_CHARS_PER_SEC = 3.4;', [
        '// Fast enough that one comment completes in roughly ten seconds at',
        '// normal fatigue, even when the motor factor sits at its floor.',
        'const BASE_TYPING_CHARS_PER_SEC = 8.0;',
        'const MIN_TYPING_CHARS_PER_SEC = 2.5;',
        'const MOTOR_FACTOR_MIN = 0.6;',
        'const MOTOR_FACTOR_MAX = 1.6;',
        '// Smoke-test escape hatch: ?nosim=1 skips the connectome sim, whose',
        '// load blocks the main thread for minutes on a CI runner.',
        "const SKIP_NEURO_SIM = /[?&]nosim=1(?:&|$)/.test(location.search);",
    ]),
    ('const summary = neuroSim.getSummary();', [
        'const summary = neuroSim.getSummary();',
        'if (!summary || !summary.groupRates) {',
        '  game.motorFactor = 1;',
        '  return;',
        '}',
    ]),
    ('const motorRate = (summary.groupRates && summary.groupRates.motor) || 0;', [
        'const rawMotorRate = summary.groupRates.motor;',
        'const motorRate = Number.isFinite(rawMotorRate) ? rawMotorRate : 0;',
    ]),
    ('if (game.motorBaseline === null) {', [
        'if (!Number.isFinite(game.motorBaseline)) {',
    ]),
    ('const w = Math.min(1, MOTOR_BASELINE_SMOOTHING * dtSeconds * 60);', [
        'const rawW = MOTOR_BASELINE_SMOOTHING * dtSeconds * 60;',
        'const w = Number.isFinite(rawW) ? Math.min(1, Math.max(0, rawW)) : 0.05;',
    ]),
    ('const ratio = motorRate / Math.max(1, game.motorBaseline);', [
        'const baseline = Number.isFinite(game.motorBaseline)',
        '  ? Math.max(1, game.motorBaseline)',
        '  : 1;',
        '// Typing must never depend on the sim being healthy. No motor signal',
        '// means neutral speed, not slow speed.',
        'const ratio = motorRate > 0 ? motorRate / baseline : 1;',
    ]),
    ('game.motorFactor = Math.max(0.5, Math.min(2, ratio));', [
        'game.motorFactor = Number.isFinite(ratio)',
        '  ? Math.max(MOTOR_FACTOR_MIN, Math.min(MOTOR_FACTOR_MAX, ratio))',
        '  : 1;',
    ]),
    ('game.anger = Math.max(0, Math.min(100, (angerHz / 2 / 150) * 100));', [
        'const angerPct = (angerHz / 2 / 150) * 100;',
        'game.anger = Number.isFinite(angerPct)',
        '  ? Math.max(0, Math.min(100, angerPct))',
        '  : game.anger;',
    ]),
    ('const rate = BASE_TYPING_CHARS_PER_SEC * game.motorFactor * (1 - game.fatigue * 0.5);', [
        'const motor = Number.isFinite(game.motorFactor)',
        '  ? Math.max(MOTOR_FACTOR_MIN, Math.min(MOTOR_FACTOR_MAX, game.motorFactor))',
        '  : 1;',
        'if (!Number.isFinite(game.fatigue)) game.fatigue = 0;',
        'const rate = BASE_TYPING_CHARS_PER_SEC * motor * (1 - game.fatigue * 0.5);',
    ]),
    ('game.typedCharsAccum += Math.max(0.2, rate) * dtSeconds;', [
        'if (!Number.isFinite(game.typedCharsAccum)) game.typedCharsAccum = 0;',
        'const safeRate = Number.isFinite(rate)',
        '  ? Math.max(MIN_TYPING_CHARS_PER_SEC, rate)',
        '  : MIN_TYPING_CHARS_PER_SEC;',
        'const safeDt = Number.isFinite(dtSeconds) ? Math.max(0, dtSeconds) : 0;',
        'game.typedCharsAccum += safeRate * safeDt;',
    ]),
    ('const targetChars = Math.min(game.currentComment.length, Math.floor(game.typedCharsAccum));', [
        'const typedSoFar = Math.floor(game.typedCharsAccum);',
        'const targetChars = Math.min(',
        '  game.currentComment.length,',
        '  Number.isFinite(typedSoFar) ? Math.max(0, typedSoFar) : 0,',
        ');',
    ]),
    ('? BASE_TYPING_CHARS_PER_SEC * game.motorFactor * (1 - game.fatigue * 0.5)', [
        '? BASE_TYPING_CHARS_PER_SEC',
        '  * (Number.isFinite(game.motorFactor) ? game.motorFactor : 1)',
        '  * (1 - game.fatigue * 0.5)',
    ]),
    ('typingRate: Math.max(0, Math.min(12, typingRate)),', [
        'typingRate: Number.isFinite(typingRate)',
        '  ? Math.max(0, Math.min(12, typingRate))',
        '  : 0,',
    ]),
    ('neuroSim = await loadNeuroSim(displayIndices);', [
        'if (SKIP_NEURO_SIM) {',
        "  console.warn('[flyingFly] nosim=1: connectome sim skipped');",
        '  neuroSim = null;',
        '} else {',
        '  neuroSim = await loadNeuroSim(displayIndices);',
        '}',
    ]),
]

# Pre-validate every anchor before mutating anything.
bad = []
for needle, _ in EDITS:
    hits = [i + 1 for i, ln in enumerate(lines) if needle in ln]
    if len(hits) != 1:
        bad.append('%d hits for %r' % (len(hits), needle))
    else:
        print('anchor ok at line %d: %s' % (hits[0], needle[:64]))

if bad:
    print('ANCHOR CHECK FAILED:')
    for b in bad:
        print('  ' + b)
    sys.exit(1)

for needle, repl in EDITS:
    idx = [i for i, ln in enumerate(lines) if needle in ln][0]
    original = lines[idx]
    indent = original[:len(original) - len(original.lstrip(' '))]
    lines[idx:idx + 1] = [(indent + r) if r else '' for r in repl]

out = '\n'.join(lines)
with io.open(PATH, 'w', encoding='utf-8') as fh:
    fh.write(out)
print('wrote %s: %d lines' % (PATH, len(lines)))
