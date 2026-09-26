#!/usr/bin/env python3
# Temporary anchored patcher. Fails loudly unless every anchor matches once.
import io
import re
import sys

FAILURES = []


def read(path):
    with io.open(path, 'r', encoding='utf-8', newline='') as fh:
        return fh.read()


def write(path, text):
    with io.open(path, 'w', encoding='utf-8', newline='') as fh:
        fh.write(text)


def plain(text, old, new, why):
    n = text.count(old)
    if n != 1:
        FAILURES.append('%s: %d plain matches, expected 1 (%r)'
                        % (why, n, old[:70]))
        return text
    print('patched: %s' % why)
    return text.replace(old, new, 1)


def rx(text, pattern, repl, why):
    new_text, n = re.subn(pattern, repl, text, count=1)
    if n != 1:
        FAILURES.append('%s: %d regex matches, expected 1 (%s)'
                        % (why, len(re.findall(pattern, text)), pattern))
        return text
    print('patched: %s' % why)
    return new_text


# ---------------------------------------------------------------- fly_rig.js
RIG = read('fly_rig.js')

RIG = plain(
    RIG,
    'const inv = 1 / meshScale;',
    'const inv = meshScale; // metres -> model millimetres',
    'fly_rig: meshScale must MULTIPLY raw STL vertices, not divide',
)
RIG = plain(
    RIG,
    'Scales by 1/meshScale',
    'Scales by meshScale (raw STL metres -> model millimetres)',
    'fly_rig: fix stale doc comment about the scale direction',
)

# ------------------------------------------------------------------- game.js
GAME = read('game.js')

GAME = plain(
    GAME,
    "import { loadFly } from './fly_rig.js';",
    "import { OrbitControls } from 'three/addons/controls/OrbitControls.js';\n"
    "import { loadFly } from './fly_rig.js';",
    'game: import OrbitControls',
)

CAMERA_BLOCK = """    // =====================================================================
    // CAMERA CONTROLS (OrbitControls) + debug hook
    // =====================================================================
    const cameraCtl = {
        world: null,
        brain: null,
        follow: true,
        hint: null,
        button: null,
    };

    function attachOrbit(camera, canvas, opts) {
        const controls = new OrbitControls(camera, canvas);
        controls.enableDamping = true;
        controls.dampingFactor = 0.08;
        controls.rotateSpeed = 0.9;
        controls.zoomSpeed = 0.9;
        controls.panSpeed = 0.8;
        controls.enablePan = true;
        controls.screenSpacePanning = false;
        controls.minDistance = opts.minDistance;
        controls.maxDistance = opts.maxDistance;
        controls.target.copy(opts.target);
        controls.mouseButtons = {
            LEFT: THREE.MOUSE.ROTATE,
            MIDDLE: THREE.MOUSE.DOLLY,
            RIGHT: THREE.MOUSE.PAN,
        };
        controls.touches = {
            ONE: THREE.TOUCH.ROTATE,
            TWO: THREE.TOUCH.DOLLY_PAN,
        };
        canvas.style.touchAction = 'none';
        canvas.addEventListener('contextmenu', (ev) => ev.preventDefault());
        controls.update();
        return controls;
    }

    // Where the camera should look: the fly when it exists, otherwise the
    // spot on the desk where it is about to appear.
    function flyFocusPoint(out) {
        const point = out || new THREE.Vector3();
        if (fly.root) {
            const box = new THREE.Box3().setFromObject(fly.root);
            if (!box.isEmpty()) {
                box.getCenter(point);
                return point;
            }
            fly.root.getWorldPosition(point);
            return point;
        }
        point.set(
            world.laptopPos ? world.laptopPos.x : 0,
            (world.deskTopY || 0.75) + 0.06,
            world.laptopPos ? world.laptopPos.z : 0,
        );
        return point;
    }

    function buildCameraUi() {
        const host = dom.worldCanvas && dom.worldCanvas.parentElement;
        if (!host || cameraCtl.hint) return;
        if (window.getComputedStyle(host).position === 'static') {
            host.style.position = 'relative';
        }
        const hint = document.createElement('p');
        hint.className = 'camera-hint';
        hint.textContent =
            '\\u041b\\u041a\\u041c \\u2014 \\u0432\\u0440\\u0430\\u0449\\u0430\\u0442\\u044c, '
            + '\\u043a\\u043e\\u043b\\u0435\\u0441\\u043e \\u2014 '
            + '\\u043f\\u0440\\u0438\\u0431\\u043b\\u0438\\u0436\\u0430\\u0442\\u044c, '
            + '\\u041f\\u041a\\u041c \\u2014 '
            + '\\u0441\\u0434\\u0432\\u0438\\u0433\\u0430\\u0442\\u044c';
        hint.style.cssText = 'position:absolute;left:10px;bottom:10px;margin:0;'
            + 'padding:4px 8px;border-radius:6px;font-size:11px;line-height:1.35;'
            + 'background:rgba(6,9,16,0.6);color:#cfd6e4;pointer-events:none;'
            + 'z-index:4;';
        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'camera-focus-btn';
        button.textContent =
            '\\u041a\\u0430\\u043c\\u0435\\u0440\\u0430 \\u043d\\u0430 '
            + '\\u043c\\u0443\\u0445\\u0443';
        button.style.cssText = 'position:absolute;right:10px;bottom:10px;'
            + 'padding:5px 10px;border-radius:6px;border:1px solid #47506a;'
            + 'background:rgba(12,16,26,0.82);color:#e6ebf5;font-size:11px;'
            + 'cursor:pointer;z-index:5;';
        button.addEventListener('click', () => {
            cameraCtl.follow = true;
            recenterOnFly(true);
        });
        host.appendChild(hint);
        host.appendChild(button);
        cameraCtl.hint = hint;
        cameraCtl.button = button;
    }

    function recenterOnFly(snapCamera) {
        if (!cameraCtl.world) return;
        const focus = flyFocusPoint();
        cameraCtl.world.target.copy(focus);
        if (snapCamera) {
            cameraCtl.world.object.position.copy(focus)
                .add(new THREE.Vector3(0.34, 0.24, 0.42));
        }
        cameraCtl.world.update();
    }

    function initWorldControls() {
        if (!world.camera || !dom.worldCanvas || cameraCtl.world) return;
        cameraCtl.world = attachOrbit(world.camera, dom.worldCanvas, {
            target: flyFocusPoint(),
            minDistance: 0.04,
            maxDistance: 8,
        });
        cameraCtl.world.maxPolarAngle = Math.PI * 0.495;
        // Scripted follow stops the moment the user touches the camera.
        cameraCtl.world.addEventListener('start', () => {
            cameraCtl.follow = false;
        });
        buildCameraUi();
    }

    function initBrainControls() {
        if (!brain.camera || !dom.brainCanvas || cameraCtl.brain) return;
        cameraCtl.brain = attachOrbit(brain.camera, dom.brainCanvas, {
            target: new THREE.Vector3(0, 0, 0),
            minDistance: 0.4,
            maxDistance: 14,
        });
    }

    function updateCameraControls() {
        if (cameraCtl.world) {
            if (cameraCtl.follow && fly.root) {
                cameraCtl.world.target.lerp(flyFocusPoint(), 0.08);
            }
            cameraCtl.world.update();
        }
        if (cameraCtl.brain) cameraCtl.brain.update();
    }

    // Scene-graph facts for headless verification.
    function publishDebug() {
        const info = {
            flyLoaded: !!fly.rig,
            flyScale: fly.scale || null,
            flyVisible: false,
            meshCount: 0,
            flyBox: null,
            phase: game.phase,
            comments: game.commentsWritten,
            fatigue: game.fatigue,
            motorFactor: game.motorFactor,
            typedChars: game.typedChars,
            commentLen: game.currentComment ? game.currentComment.length : null,
            follow: cameraCtl.follow,
            hasWorldControls: !!cameraCtl.world,
            hasBrainControls: !!cameraCtl.brain,
        };
        if (fly.root) {
            let meshes = 0;
            fly.root.traverse((obj) => { if (obj.isMesh) meshes += 1; });
            info.meshCount = meshes;
            const box = new THREE.Box3().setFromObject(fly.root);
            if (!box.isEmpty()) {
                const size = new THREE.Vector3();
                const center = new THREE.Vector3();
                box.getSize(size);
                box.getCenter(center);
                info.flyBox = {
                    size: [size.x, size.y, size.z],
                    center: [center.x, center.y, center.z],
                };
                info.flyVisible = fly.root.visible && size.length() > 1e-4;
            }
        }
        window.__debug = info;
    }

"""

GAME = plain(
    GAME,
    'function safeSetRagdoll(weight) {',
    CAMERA_BLOCK + 'function safeSetRagdoll(weight) {',
    'game: insert camera-control + debug block',
)

GAME = rx(
    GAME,
    r'(world\.renderer = makeRenderer\([^;]*\);)',
    r'\1\n        initWorldControls();',
    'game: create world OrbitControls',
)

GAME = rx(
    GAME,
    r'(brain\.renderer = makeRenderer\([^;]*\);)',
    r'\1\n        initBrainControls();',
    'game: create brain OrbitControls',
)

GAME = rx(
    GAME,
    r'(world\.renderer\.render\(world\.scene, world\.camera\);)',
    r'updateCameraControls();\n        \1\n        publishDebug();',
    'game: drive controls + debug from the render loop',
)

GAME = plain(
    GAME,
    'world.scene.add(flyRoot);',
    """world.scene.add(flyRoot);
        if (cameraCtl.follow) recenterOnFly(true);
        // Loud guard against a silently invisible fly: a healthy rig measures
        // roughly FLY_TARGET_LENGTH across once scaled.
        const visBox = new THREE.Box3().setFromObject(flyRoot);
        const visSize = new THREE.Vector3();
        visBox.getSize(visSize);
        if (!(visSize.length() > FLY_TARGET_LENGTH * 0.2)) {
            const detail = visSize.length().toExponential(3);
            console.error('[flyingFly] degenerate fly geometry, size=' + detail);
            showErrorBanner(
                '\\u041c\\u0443\\u0445\\u0430 \\u0437\\u0430\\u0433\\u0440\\u0443\\u0436\\u0435\\u043d\\u0430, '
                + '\\u043d\\u043e \\u0435\\u0451 \\u0433\\u0435\\u043e\\u043c\\u0435\\u0442\\u0440\\u0438\\u044f '
                + '\\u0432\\u044b\\u0440\\u043e\\u0436\\u0434\\u0435\\u043d\\u0430 (\\u0440\\u0430\\u0437\\u043c\\u0435\\u0440 '
                + detail + ').', true);
        }""",
    'game: recenter on the fly + degenerate-geometry banner',
)

GAME = rx(
    GAME,
    r'(function showErrorBanner\([^)]*\)\s*\{)',
    r"\1\n        console.error('[flyingFly] banner: ' + arguments[0]);",
    'game: mirror every error banner to the console',
)

if FAILURES:
    sys.stderr.write('ANCHOR FAILURES:\n')
    for item in FAILURES:
        sys.stderr.write('  - %s\n' % item)
    sys.stderr.write('\nContext probes:\n')
    for probe in ('makeRenderer(', 'renderer.render(', 'showErrorBanner',
                  'safeSetRagdoll', 'scene.add(flyRoot'):
        for m in re.finditer(re.escape(probe), read('game.js')):
            s = max(0, m.start() - 70)
            frag = read('game.js')[s:m.end() + 70]
            sys.stderr.write('  %s => %r\n' % (probe, frag))
            break
    sys.exit(1)

write('fly_rig.js', RIG)
write('game.js', GAME)
print('ALL ANCHORS OK')
