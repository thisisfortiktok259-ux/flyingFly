#!/usr/bin/env python3
"""Temporary hard-bounded headless smoke harness.

Usage: smoke2.py <watch_seconds> [url_suffix]

The page blocks its main thread for minutes while it loads connectome data,
and page.evaluate is NOT governed by Playwright's default timeout, so a
blocked evaluate simply waits. That is used deliberately here: the first
evaluate returns the instant the main thread frees up. SIGALRM is the only
hard bound.

All output is ASCII-escaped because Actions logs mangle UTF-8 Cyrillic.
"""
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request

WATCH = int(sys.argv[1]) if len(sys.argv) > 1 else 25
SUFFIX = sys.argv[2] if len(sys.argv) > 2 else ''
TICK = 30
BASE = 'http://127.0.0.1:8085'
URL = BASE + '/' + SUFFIX
HARD_LIMIT = WATCH + 120


def out(text):
    sys.stdout.write(text.encode('unicode_escape').decode('ascii') + '\n')
    sys.stdout.flush()


def on_alarm(signum, frame):
    out('FATAL: global alarm fired after %ds' % HARD_LIMIT)
    sys.stdout.flush()
    os._exit(2)


signal.signal(signal.SIGALRM, on_alarm)
signal.alarm(HARD_LIMIT)

server = subprocess.Popen([sys.executable, 'server.py'],
                          stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL)

up = False
for _ in range(50):
    try:
        urllib.request.urlopen(BASE + '/api/health', timeout=2).read()
        up = True
        break
    except Exception:
        time.sleep(0.5)

if not up:
    out('FATAL: server.py never answered /api/health')
    os._exit(3)

out('server up, target URL: %s' % URL)

from playwright.sync_api import sync_playwright  # noqa: E402

console = []
page_errors = []
failed = []


def safe(label, fn, default=None):
    try:
        return fn()
    except Exception as exc:
        out('STALLED/ERR %s: %s' % (label, str(exc).split('\n')[0][:160]))
        return default


with sync_playwright() as pw:
    browser = pw.chromium.launch(args=[
        '--no-sandbox',
        '--disable-dev-shm-usage',
        '--use-gl=swiftshader',
        '--enable-unsafe-swiftshader',
        '--ignore-gpu-blocklist',
        '--enable-webgl',
        '--disable-gpu-sandbox',
    ])
    page = browser.new_page(viewport={'width': 1280, 'height': 800})
    page.set_default_timeout(10000)
    page.on('console', lambda m: console.append((m.type, m.text[:300])))
    page.on('pageerror', lambda e: page_errors.append(str(e)[:300]))
    page.on('requestfailed',
            lambda r: failed.append('%s %s' % (r.url[:120],
                                               r.failure or 'unknown')))

    started = time.time()
    safe('goto', lambda: page.goto(URL, wait_until='domcontentloaded',
                                   timeout=60000))
    out('page loaded at t=%.1fs, watching up to %ds (tick %ds)'
        % (time.time() - started, WATCH, TICK))

    debug = None
    deadline = started + WATCH
    while time.time() < deadline:
        got = safe('evaluate __debug',
                   lambda: page.evaluate('() => window.__debug ? '
                                         'JSON.stringify(window.__debug) : null'))
        elapsed = time.time() - started
        if got:
            debug = json.loads(got)
            out('t=%03.0fs %s' % (elapsed, got[:460]))
            # Once the loop is alive, sample a few more times to prove that
            # the comment counter actually advances.
            if debug.get('comments', 0) > 0:
                break
        else:
            out('t=%03.0fs __debug=null' % elapsed)
        time.sleep(TICK)

    def text_of(sel):
        return safe('text %s' % sel,
                    lambda: page.eval_on_selector(
                        sel, 'el => (el.textContent || "").trim().slice(0,120)'),
                    default='<none>')

    out('')
    out('== DOM ==')
    for sel in ['#status', '#error-banner', '#comments-count', '#forced-count',
                '#backend-label', '#source-label', '#loading-text',
                '#subtitle']:
        out('%-18s %s' % (sel, text_of(sel)))

    out('')
    out('== CAMERA UI ==')
    for sel in ['.camera-hint', '.camera-focus-btn']:
        found = safe('count %s' % sel,
                     lambda s=sel: page.eval_on_selector_all(
                         s, 'els => els.length'), default=-1)
        out('%-20s count=%s' % (sel, found))

    out('')
    out('== GL ==')
    gl = safe('gl info', lambda: page.evaluate(
        '''() => {
            const c = document.createElement('canvas');
            const g = c.getContext('webgl2') || c.getContext('webgl');
            if (!g) return 'no webgl';
            return g.getParameter(g.VERSION) + ' | ' +
                   g.getParameter(g.RENDERER) + ' | gpu=' +
                   (navigator.gpu ? 'yes' : 'no');
        }'''), default='<stalled>')
    out(str(gl))

    os.makedirs('artifacts', exist_ok=True)
    safe('screenshot', lambda: page.screenshot(path='artifacts/page.png',
                                               timeout=20000))

    final = safe('final __debug',
                 lambda: page.evaluate('() => window.__debug ? '
                                       'JSON.stringify(window.__debug) : null'))
    if final:
        debug = json.loads(final)

    out('')
    out('== PAGE ERRORS (%d) ==' % len(page_errors))
    for e in page_errors[:12]:
        out('  ' + e)

    out('== FAILED REQUESTS (%d) ==' % len(failed))
    for f in failed[:12]:
        out('  ' + f)

    errs = [c for c in console if c[0] in ('error', 'warning')]
    out('== CONSOLE error/warning (%d of %d total) ==' % (len(errs),
                                                          len(console)))
    for kind, text in errs[:25]:
        out('  [%s] %s' % (kind, text))

    d = debug or {}
    box = d.get('flyBox') or {}
    out('')
    out('== VERDICT ==')
    out('FLY_LOADED     %s' % d.get('flyLoaded'))
    out('FLY_VISIBLE    %s' % d.get('flyVisible'))
    out('FLY_MESH_COUNT %s' % d.get('meshCount'))
    out('FLY_SCALE      %s' % d.get('flyScale'))
    out('FLY_BOX_SIZE   %s' % (box.get('size'),))
    out('FLY_BOX_CENTER %s' % (box.get('center'),))
    out('PHASE          %s' % d.get('phase'))
    out('COMMENTS       %s' % d.get('comments'))
    out('FATIGUE        %s' % d.get('fatigue'))
    out('MOTOR_FACTOR   %s' % d.get('motorFactor'))
    out('TYPED_CHARS    %s' % d.get('typedChars'))
    out('COMMENT_LEN    %s' % d.get('commentLen'))
    out('FOLLOW         %s' % d.get('follow'))
    out('WORLD_CONTROLS %s' % d.get('hasWorldControls'))
    out('BRAIN_CONTROLS %s' % d.get('hasBrainControls'))
    out('PAGE_ERRORS    %d' % len(page_errors))

    safe('close', lambda: browser.close())

server.kill()
sys.stdout.flush()
os._exit(0)
