#!/usr/bin/env python3
"""Temporary hard-bounded headless smoke check for the flyingFly front end.

Prints ASCII-only diagnostics (Actions logs mangle Cyrillic) and never hangs:
SIGALRM kills the process, every wait has a timeout, and it exits via os._exit.
"""
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request

PORT = 8085
BASE = 'http://127.0.0.1:%d' % PORT
WATCH_SECONDS = int(sys.argv[1]) if len(sys.argv) > 1 else 35
OUT_DIR = 'artifacts'


def die_on_alarm(signum, frame):
    sys.stdout.write('\nFATAL: global alarm fired, aborting\n')
    sys.stdout.flush()
    os._exit(2)


signal.signal(signal.SIGALRM, die_on_alarm)
signal.alarm(WATCH_SECONDS + 180)


def asciify(value):
    """Actions logs corrupt UTF-8, so escape everything to ASCII."""
    if value is None:
        return 'None'
    text = value if isinstance(value, str) else str(value)
    return text.encode('unicode_escape').decode('ascii')


def wait_for_server(timeout=25):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(BASE + '/api/health', timeout=3) as rsp:
                if rsp.status == 200:
                    return True
        except Exception:
            time.sleep(0.5)
    return False


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    server = subprocess.Popen(
        [sys.executable, 'server.py'],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if not wait_for_server():
        print('FATAL: server.py did not answer /api/health')
        server.kill()
        os._exit(3)
    print('server.py is up on %s' % BASE)

    from playwright.sync_api import sync_playwright

    console_lines = []
    page_errors = []
    failed_requests = []

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
        page.set_default_timeout(15000)
        page.on('console', lambda m: console_lines.append(
            '[%s] %s' % (m.type, m.text)))
        page.on('pageerror', lambda e: page_errors.append(str(e)))
        page.on('requestfailed', lambda r: failed_requests.append(
            '%s %s' % (r.url, r.failure)))

        try:
            page.goto(BASE + '/', wait_until='domcontentloaded', timeout=30000)
        except Exception as exc:
            print('FATAL: goto failed: %s' % asciify(exc))
            browser.close()
            server.kill()
            os._exit(4)
        print('page loaded, watching for %ds' % WATCH_SECONDS)

        elapsed = 0
        while elapsed < WATCH_SECONDS:
            page.wait_for_timeout(5000)
            elapsed += 5
            try:
                dbg = page.evaluate('() => window.__debug || null')
            except Exception as exc:
                dbg = {'evaluate_error': asciify(exc)}
            print('t=%02ds __debug=%s' % (elapsed, asciify(json.dumps(dbg))))

        print('\n== DOM ==')
        for el_id in ('status', 'error-banner', 'comments-count',
                      'forced-count', 'backend-label', 'source-label',
                      'loading-text', 'subtitle'):
            try:
                text = page.evaluate(
                    '(id) => { const n = document.getElementById(id);'
                    ' return n ? (n.hidden ? "[hidden] " : "") + '
                    '(n.textContent || "").trim().slice(0, 200) : "[missing]"; }',
                    el_id)
            except Exception as exc:
                text = 'eval error: %s' % exc
            print('  #%-15s %s' % (el_id, asciify(text)))

        print('\n== camera UI present ==')
        try:
            ui = page.evaluate(
                '() => ({ hint: !!document.querySelector(".camera-hint"),'
                ' button: !!document.querySelector(".camera-focus-btn") })')
        except Exception as exc:
            ui = {'error': str(exc)}
        print('  %s' % asciify(json.dumps(ui)))

        print('\n== graphics ==')
        try:
            gfx = page.evaluate(
                '() => { const c = document.createElement("canvas");'
                ' const gl = c.getContext("webgl2") || c.getContext("webgl");'
                ' if (!gl) return { webgl: false };'
                ' const d = gl.getExtension("WEBGL_debug_renderer_info");'
                ' return { webgl: true, version: gl.getParameter(gl.VERSION),'
                ' renderer: d ? gl.getParameter(d.UNMASKED_RENDERER_WEBGL) : null,'
                ' webgpu: !!navigator.gpu }; }')
        except Exception as exc:
            gfx = {'error': str(exc)}
        print('  %s' % asciify(json.dumps(gfx)))

        try:
            page.screenshot(path=os.path.join(OUT_DIR, 'page.png'))
            world = page.query_selector('#world-canvas')
            if world:
                world.screenshot(path=os.path.join(OUT_DIR, 'world.png'))
        except Exception as exc:
            print('screenshot failed: %s' % asciify(exc))

        final = None
        try:
            final = page.evaluate('() => window.__debug || null')
        except Exception:
            pass

        browser.close()

    server.kill()

    print('\n== page errors (%d) ==' % len(page_errors))
    for line in page_errors[:20]:
        print('  ' + asciify(line))
    print('\n== failed requests (%d) ==' % len(failed_requests))
    for line in failed_requests[:20]:
        print('  ' + asciify(line))
    print('\n== console: errors and warnings ==')
    shown = 0
    for line in console_lines:
        if line.startswith('[error]') or line.startswith('[warning]'):
            print('  ' + asciify(line))
            shown += 1
            if shown >= 30:
                break
    print('(console messages total: %d)' % len(console_lines))

    print('\n== VERDICT ==')
    fly_visible = bool(final and final.get('flyVisible'))
    comments = (final or {}).get('comments')
    box = (final or {}).get('flyBox')
    print('FLY_VISIBLE=%s' % fly_visible)
    print('FLY_MESH_COUNT=%s' % ((final or {}).get('meshCount')))
    print('FLY_SCALE=%s' % ((final or {}).get('flyScale')))
    print('FLY_BOX=%s' % asciify(json.dumps(box)))
    print('COMMENTS=%s' % comments)
    print('FATIGUE=%s' % ((final or {}).get('fatigue')))
    print('MOTOR_FACTOR=%s' % ((final or {}).get('motorFactor')))
    print('TYPED_CHARS=%s' % ((final or {}).get('typedChars')))
    print('COMMENT_LEN=%s' % ((final or {}).get('commentLen')))
    print('PAGE_ERRORS=%d' % len(page_errors))
    os._exit(0)


if __name__ == '__main__':
    main()
