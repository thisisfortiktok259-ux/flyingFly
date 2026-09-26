#!/usr/bin/env python3
# Temporary CI investigation toolkit. Removed before the end of the task.
import json
import os
import re
import struct
import subprocess
import sys
import time
import urllib.request

SPACE = 'Xenova/fruit-fly-simulation'
RESOLVE = 'https://huggingface.co/spaces/%s/resolve/main/' % SPACE
API_TREE = ('https://huggingface.co/api/spaces/%s/tree/main?recursive=true'
            % SPACE)


def get(url, binary=False, timeout=180):
    req = urllib.request.Request(url, headers={'User-Agent': 'ci-probe'})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read()
    return data if binary else data.decode('utf-8', 'replace')


def fmt3(v):
    return '[%.4f, %.4f, %.4f]' % (v[0], v[1], v[2])


def stl_bounds(data):
    """Return (kind, tri_count, min[3], max[3]) for a binary or ASCII STL."""
    if len(data) >= 84:
        n = struct.unpack('<I', data[80:84])[0]
        if 84 + n * 50 == len(data):
            mn = [1e30] * 3
            mx = [-1e30] * 3
            off = 84
            for _ in range(n):
                for v in range(3):
                    p = off + 12 + v * 12
                    xyz = struct.unpack('<3f', data[p:p + 12])
                    for k in range(3):
                        if xyz[k] < mn[k]:
                            mn[k] = xyz[k]
                        if xyz[k] > mx[k]:
                            mx[k] = xyz[k]
                off += 50
            return 'binary', n, mn, mx
    txt = data.decode('utf-8', 'replace')
    mn = [1e30] * 3
    mx = [-1e30] * 3
    n = 0
    for m in re.finditer(r'vertex\s+(\S+)\s+(\S+)\s+(\S+)', txt):
        n += 1
        for k in range(3):
            val = float(m.group(k + 1))
            if val < mn[k]:
                mn[k] = val
            if val > mx[k]:
                mx[k] = val
    return 'ascii', n // 3, mn, mx


def walk_positions(node, out, depth=0, name='root'):
    """Collect (name, pos) pairs from an unknown-shaped model.json tree."""
    if isinstance(node, dict):
        nm = node.get('name', name)
        pos = node.get('pos') or node.get('position')
        if isinstance(pos, list) and len(pos) == 3:
            try:
                out.append((str(nm), [float(x) for x in pos],
                            node.get('mesh') or node.get('meshes')))
            except (TypeError, ValueError):
                pass
        for key, val in node.items():
            if isinstance(val, (dict, list)):
                walk_positions(val, out, depth + 1, str(key))
    elif isinstance(node, list):
        for item in node:
            walk_positions(item, out, depth + 1, name)


def cmd_assets():
    print('== model.json ==')
    raw = None
    model_url = None
    for cand in ('public/body/assets/model.json',
                 'public/assets/body/model.json',
                 'public/body/model.json'):
        try:
            raw = get(RESOLVE + cand)
            model_url = RESOLVE + cand
            break
        except Exception as exc:  # noqa: BLE001
            print('miss %s (%s)' % (cand, type(exc).__name__))
    if raw is None:
        print('FATAL: model.json not found')
        return 1
    print('url: %s' % model_url)
    print('bytes: %d' % len(raw))
    model = json.loads(raw)
    print('top-level keys: %s' % sorted(model.keys())[:40])
    for key in ('meshScale', 'scale', 'unit', 'units', 'version'):
        if key in model:
            print('%s = %r' % (key, model[key]))

    positions = []
    walk_positions(model, positions)
    print('parts with pos: %d' % len(positions))
    mn = [1e30] * 3
    mx = [-1e30] * 3
    for _, pos, _m in positions:
        for k in range(3):
            if pos[k] < mn[k]:
                mn[k] = pos[k]
            if pos[k] > mx[k]:
                mx[k] = pos[k]
    if positions:
        print('rest-pos min %s' % fmt3(mn))
        print('rest-pos max %s' % fmt3(mx))
        print('rest-pos span %s' % fmt3([mx[i] - mn[i] for i in range(3)]))
    wanted = ('c_thorax', 'c_head', 'c_abdomen', 'lf_tibia', 'l_wing')
    for nm, pos, mesh in positions:
        if nm in wanted:
            print('part %-10s pos %s mesh %r' % (nm, fmt3(pos), mesh))

    mesh_names = set()

    def collect_meshes(node):
        if isinstance(node, dict):
            for key, val in node.items():
                if key in ('mesh', 'meshes', 'file', 'stl'):
                    if isinstance(val, str):
                        mesh_names.add(val)
                    elif isinstance(val, list):
                        for item in val:
                            if isinstance(item, str):
                                mesh_names.add(item)
                collect_meshes(val)
        elif isinstance(node, list):
            for item in node:
                collect_meshes(item)

    collect_meshes(model)
    print('distinct mesh refs: %d, sample %s'
          % (len(mesh_names), sorted(mesh_names)[:8]))

    print('')
    print('== STL vertex ranges (raw file units) ==')
    base = model_url.rsplit('/', 1)[0]
    targets = ['c_thorax.stl', 'c_head.stl', 'lf_tibia.stl', 'l_wing.stl']
    prefixes = ['meshes/', '', '../meshes/']
    for name in targets:
        got = False
        for pref in prefixes:
            url = '%s/%s%s' % (base, pref, name)
            try:
                data = get(url, binary=True, timeout=120)
            except Exception:  # noqa: BLE001
                continue
            kind, tris, mn, mx = stl_bounds(data)
            span = [mx[i] - mn[i] for i in range(3)]
            print('%-13s %s tris=%-6d bytes=%d' % (name, kind, tris,
                                                   len(data)))
            print('    min %s' % fmt3(mn))
            print('    max %s' % fmt3(mx))
            print('    span %s  url=%s%s' % (fmt3(span), pref, name))
            got = True
            break
        if not got:
            print('%-13s NOT FOUND under %s' % (name, base))
    return 0


def cmd_space_js():
    print('== Space file tree (js/ts only) ==')
    try:
        tree = json.loads(get(API_TREE))
    except Exception as exc:  # noqa: BLE001
        print('tree fetch failed: %r' % (exc,))
        return 0
    paths = [e.get('path', '') for e in tree if isinstance(e, dict)]
    code = [p for p in paths
            if p.endswith(('.js', '.mjs', '.ts', '.jsx', '.tsx', '.html'))]
    print('total entries %d, code files %d' % (len(paths), len(code)))
    for p in code[:60]:
        print('  %s' % p)
    print('')
    print('== grep meshScale in Space code ==')
    hits = 0
    for p in code:
        try:
            txt = get(RESOLVE + p, timeout=90)
        except Exception:  # noqa: BLE001
            continue
        if 'meshScale' not in txt:
            continue
        hits += 1
        print('--- %s ---' % p)
        for m in re.finditer(r'meshScale', txt):
            s = max(0, m.start() - 160)
            e = min(len(txt), m.end() + 160)
            frag = txt[s:e].replace('\n', ' ')
            frag = re.sub(r'\s+', ' ', frag)
            print('   ...%s...' % frag)
            if hits > 6:
                break
        if hits > 6:
            break
    if not hits:
        print('no meshScale references found in Space code')
    return 0


TOPICS = {
    'head': [(r'^\s*import\s', 0), (r'getElementById', 0),
             (r'const dom\s*=', 6)],
    'world': [(r'function createWorldScene', 150)],
    'fly': [(r'function initFlyRig', 90), (r'function safeApplyImpulse', 30),
            (r'function safeSetRagdoll', 20)],
    'frame': [(r'function frame', 110)],
    'game': [(r'function stepGame', 150)],
    'err': [(r'error-banner|errorBanner|showError|loadingText|loading-screen',
             4)],
    'comment': [(r'function pushComment|function emitComment|commentsCount'
                 r'|forcedCount|comments\s*\+\+|game\.comments', 6)],
}


def cmd_dump(topic):
    with open('game.js', 'r', encoding='utf-8') as fh:
        lines = fh.read().split('\n')
    print('game.js lines: %d  topic: %s' % (len(lines), topic))
    shown = set()
    for pattern, window in TOPICS[topic]:
        rx = re.compile(pattern)
        for idx, line in enumerate(lines):
            if not rx.search(line):
                continue
            lo = idx
            hi = min(len(lines), idx + window + 1)
            print('--- match line %d: %s' % (idx + 1, pattern))
            for j in range(lo, hi):
                if j in shown:
                    continue
                shown.add(j)
                print('%d|%s' % (j + 1, lines[j].strip()))
    return 0


BROWSER_ARGS = [
    '--no-sandbox',
    '--disable-dev-shm-usage',
    '--use-gl=swiftshader',
    '--enable-unsafe-swiftshader',
    '--ignore-gpu-blocklist',
    '--enable-webgl',
    '--disable-gpu-sandbox',
]


def cmd_smoke(wait_seconds):
    from playwright.sync_api import sync_playwright

    env = dict(os.environ)
    proc = subprocess.Popen([sys.executable, 'server.py'], env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    url = 'http://127.0.0.1:8085/'
    ready = False
    for _ in range(60):
        try:
            get(url, timeout=3)
            ready = True
            break
        except Exception:  # noqa: BLE001
            time.sleep(0.5)
    print('server ready: %s' % ready)
    if not ready:
        proc.kill()
        print(proc.stdout.read().decode('utf-8', 'replace')[:2000])
        return 1

    console = []
    errors = []
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(args=BROWSER_ARGS)
            page = browser.new_page(viewport={'width': 1440, 'height': 900})
            page.on('console', lambda m: console.append(
                '%s: %s' % (m.type, m.text[:400])))
            page.on('pageerror', lambda e: errors.append(str(e)[:600]))
            page.goto(url, wait_until='load', timeout=60000)
            page.wait_for_timeout(int(wait_seconds) * 1000)

            print('')
            print('== page errors (%d) ==' % len(errors))
            for e in errors[:25]:
                print('  ! %s' % e)

            print('')
            print('== console (%d, warnings/errors first) ==' % len(console))
            bad = [c for c in console
                   if c.startswith('error') or c.startswith('warning')]
            for c in bad[:40]:
                print('  %s' % c)
            rest = [c for c in console if c not in bad]
            for c in rest[:25]:
                print('  %s' % c)

            print('')
            print('== DOM ==')
            for sel in ('#status', '#error-banner', '#comments-count',
                        '#forced-count', '#backend-label', '#source-label',
                        '#loading-text', '#subtitle'):
                try:
                    node = page.query_selector(sel)
                    if node is None:
                        print('%-16s MISSING' % sel)
                        continue
                    vis = node.is_visible()
                    print('%-16s vis=%s text=%r'
                          % (sel, vis, (node.inner_text() or '')[:160]))
                except Exception as exc:  # noqa: BLE001
                    print('%-16s read failed %r' % (sel, exc))
            for sel in ('#event-log', '#comment-log'):
                node = page.query_selector(sel)
                txt = (node.inner_text() if node else '') or ''
                txt = re.sub(r'\n+', ' | ', txt)
                print('%s (%d chars): %s' % (sel, len(txt), txt[:900]))

            print('')
            print('== canvases ==')
            info = page.evaluate(
                """() => Array.from(document.querySelectorAll('canvas'))
                    .map(c => ({id: c.id, w: c.width, h: c.height,
                                cw: c.clientWidth, ch: c.clientHeight}))""")
            for c in info:
                print('  %r' % (c,))

            print('')
            print('== window.__debug ==')
            dbg = page.evaluate(
                "() => { try { return JSON.stringify(window.__debug); }"
                " catch (e) { return 'ERR ' + e.message; } }")
            print('  %s' % (dbg if dbg else 'undefined'))

            print('')
            print('== webgl probe ==')
            gl = page.evaluate(
                """() => { const c = document.createElement('canvas');
                    const g = c.getContext('webgl2') || c.getContext('webgl');
                    if (!g) return 'no webgl';
                    return g.getParameter(g.VERSION) + ' | ' +
                        g.getParameter(g.RENDERER); }""")
            print('  %s' % gl)
            print('  webgpu: %s' % page.evaluate(
                "() => !!(navigator.gpu)"))

            os.makedirs('artifacts', exist_ok=True)
            page.screenshot(path='artifacts/page.png', full_page=False)
            world = page.query_selector('#world-canvas')
            if world:
                world.screenshot(path='artifacts/world.png')
            brain = page.query_selector('#brain-canvas')
            if brain:
                brain.screenshot(path='artifacts/brain.png')
            print('')
            print('screenshots written')
            browser.close()
    finally:
        proc.kill()
    return 0


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'assets'
    if cmd == 'assets':
        return cmd_assets()
    if cmd == 'space-js':
        return cmd_space_js()
    if cmd == 'dump':
        return cmd_dump(sys.argv[2])
    if cmd == 'smoke':
        return cmd_smoke(sys.argv[2] if len(sys.argv) > 2 else '40')
    print('unknown command %r' % cmd)
    return 2


if __name__ == '__main__':
    sys.exit(main())
