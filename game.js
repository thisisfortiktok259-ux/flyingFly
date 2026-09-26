/**
 * game.js
 *
 * ------------------------------------------------------------
 * Flying Fly — autonomous 3D fly simulation front end.
 *
 * This file is the ONLY frontend module this change touches (aside from
 * coordinated markup/CSS tweaks in index.html/style.css needed to match).
 * It coordinates with index.html, which declares an importmap for "three"
 * (three@0.169.0) and the following element ids:
 * brain-canvas, world-canvas, scope-canvas, raster-canvas, equalizer,
 * chapter-name, chapter-list, chapter-prev, chapter-next, chapter-autoplay,
 * caption, source-label, telemetry-rate, motor-walk, motor-turn,
 * motor-escape, status, sound-toggle, event-log, loading-screen,
 * error-banner
 *
 * Important framing, matching the backend README:
 * - The fly is fully autonomous. There are no manual flight controls. The
 *   motor-walk / motor-turn / motor-escape sliders are read-only telemetry
 *   reflecting what the autopilot is doing, not inputs (their pointer/key
 *   interaction is suppressed and aria-readonly is set). The chapter-prev /
 *   chapter-next / chapter-autoplay buttons and the chapter-list only pick
 *   which narrative segment the camera focuses on; they never override the
 *   fly's own movement logic.
 * - GET /api/neurons returns REAL sampled MaleCNS soma coordinates. If that
 *   request fails, this file shows an explicit error (status text, event
 *   log, and error banner) and renders NO point cloud. It never fabricates
 *   or invents connectome data.
 * - The oscilloscope, 96-channel raster, and the point-cloud pulse shader
 *   are procedurally modeled visual flourishes (loosely referencing an
 *   89.3 Hz baseline that ramps into a 125-140 Hz "excited" band, and
 *   PAM-style burst timing). None of this is a recording of real
 *   electrophysiology, none of it is labeled as an actual recording, and
 *   this project does not claim a whole-fly neural emulation anywhere.
 * - The 28-bar equalizer is a set of real DOM elements (.eq-bar divs inside
 *   #equalizer) whose heights are driven by a REAL WebAudio AnalyserNode
 *   listening to a synthesized, original 140 BPM procedural loop built from
 *   oscillators/noise buffers (no samples, no copyrighted audio). Audio
 *   only starts after the user clicks the sound toggle.
 * - The fly body model is loaded from the real STL/model.json assets via
 *   fly_rig.js's loadFly(). If loading fails, the status text and error
 *   banner say so and NO primitive placeholder fly is drawn.
 * - The phone "screen" content is drawn with an original, procedurally
 *   drawn CanvasTexture that simulates a generic vertical short-form video
 *   feed (progress dots, a heart/like counter, an @handle). It contains no
 *   downloaded video, no real footage, and no third-party logos or brand
 *   names — it is a stylized, original UI mockup only.
 * - The skateboard, compact car, and podium are stylized, generic
 *   geometric props with no logos/decals and make no trademark or brand
 *   claims.
 * - Chat uses POST /api/chat with the backend's exact NPC name whitelist
 *   (Cyrillic: Зина, Артем, Григорий, Петрович, Барсик, Даня).
 *   Replies are model-generated text (see server.py / README), shown as
 *   subtitles and, opt-in only after a user gesture, read aloud with the
 *   Web Speech API in ru-RU.
 * - Autonomous chapter cycle order is: phone check (simulated short-form
 *   video break) -> street NPC chat -> skateboard -> compact car, with the
 *   podium kept as an optional fifth chapter at the end of the loop. Users
 *   can step chapters manually (chapter-prev/next) or pause the automatic
 *   cycle (chapter-autoplay); the chapter-list is rendered dynamically from
 *   this same chapter set so its labels can never drift out of sync.
 * ------------------------------------------------------------------------
 */
import * as THREE from 'three';
import { loadFly } from './fly_rig.js';

(() => {
  'use strict';

  // ----------------------------------------------------------------------
  // Config
  // ----------------------------------------------------------------------
  const ENDPOINTS = {
    chat: '/api/chat',
    neurons: '/api/neurons',
  };

  // Exact server whitelist (see server.py NPC_PERSONAS). Never invent names.
  const NPC_NAMES = Object.freeze({
    zina: 'Зина',
    artem: 'Артем',
    grigory: 'Григорий',
    petrovich: 'Петрович',
    barsik: 'Барсик',
    danya: 'Даня',
  });

  const CHAT_MIN_INTERVAL_MS = 5000; // request throttling for /api/chat
  const CHAT_TIMEOUT_MS = 12000;

  // Fallback chapter dwell window; per-chapter overrides live in
  // CHAPTER_DURATIONS below (phone swipes are quick, conversational
  // chapters get more room so a chat reply has time to land).
  const CHAPTER_MIN_MS = 6000;
  const CHAPTER_MAX_MS = 8000;
  const CHAPTER_DURATIONS = {
    phone: [6000, 8000],
    street: [8000, 11000],
    skate: [8000, 11000],
    car: [8000, 11000],
    podium: [8000, 11000],
  };

  const EQUALIZER_BARS = 28;
  const RASTER_ROWS = 48; // sampled rows drawn out of...
  const RASTER_CHANNELS = 96; // ...a modeled 96-channel layout (2 channels/row)
  const MAX_EVENT_LOG_ENTRIES = 30;

  const CHAPTER_LINES = {
    street: ['Что там видно с высоты?', 'Куда лучше свернуть на этой улице?'],
    skate: ['Покажешь трюк на доске?', 'Оранжевые колёса не подводят?'],
    car: ['Куда едет эта зелёная машина?', 'Успеешь обогнать её на повороте?'],
    podium: ['Как ощущения на подиуме?', 'Отражение пола тебя не слепит?'],
    phone: ['Что там на экране телефона?', 'Что ты сейчас свотришь?'],
  };

  // ----------------------------------------------------------------------
  // DOM references (coordinated ids; every lookup is null-safe)
  // ----------------------------------------------------------------------
  const dom = {
    brainCanvas: document.getElementById('brain-canvas'),
    worldCanvas: document.getElementById('world-canvas'),
    scopeCanvas: document.getElementById('scope-canvas'),
    rasterCanvas: document.getElementById('raster-canvas'),
    equalizer: document.getElementById('equalizer'),
    chapterName: document.getElementById('chapter-name'),
    chapterList: document.getElementById('chapter-list'),
    chapterPrev: document.getElementById('chapter-prev'),
    chapterNext: document.getElementById('chapter-next'),
    chapterAutoplay: document.getElementById('chapter-autoplay'),
    caption: document.getElementById('caption'),
    sourceLabel: document.getElementById('source-label'),
    telemetryRate: document.getElementById('telemetry-rate'),
    motorWalk: document.getElementById('motor-walk'),
    motorTurn: document.getElementById('motor-turn'),
    motorEscape: document.getElementById('motor-escape'),
    status: document.getElementById('status'),
    soundToggle: document.getElementById('sound-toggle'),
    eventLog: document.getElementById('event-log'),
    loadingScreen: document.getElementById('loading-screen'),
    errorBanner: document.getElementById('error-banner'),
  };

  // ----------------------------------------------------------------------
  // Status text, event log, error banner, loading screen
  // ----------------------------------------------------------------------
  let fatalErrorActive = false;
  let errorBannerTimeout = null;

  function setStatus(message, state) {
    if (!dom.status) return;
    dom.status.textContent = message;
    dom.status.dataset.state = state || 'loading';
  }

  function logEvent(message) {
    if (!dom.eventLog) return;
    const now = new Date();
    const hh = String(now.getHours()).padStart(2, '0');
    const mm = String(now.getMinutes()).padStart(2, '0');
    const ss = String(now.getSeconds()).padStart(2, '0');
    const li = document.createElement('li');
    li.textContent = `[${hh}:${mm}:${ss}] ${message}`;
    dom.eventLog.appendChild(li);
    while (dom.eventLog.children.length > MAX_EVENT_LOG_ENTRIES) {
      dom.eventLog.removeChild(dom.eventLog.firstChild);
    }
    dom.eventLog.scrollTop = dom.eventLog.scrollHeight;
  }

  function showErrorBanner(message, persist) {
    if (!dom.errorBanner) return;
    dom.errorBanner.textContent = message;
    dom.errorBanner.hidden = false;
    if (errorBannerTimeout) {
      clearTimeout(errorBannerTimeout);
      errorBannerTimeout = null;
    }
    if (!persist) {
      errorBannerTimeout = setTimeout(() => {
        if (dom.errorBanner) dom.errorBanner.hidden = true;
      }, 7000);
    }
  }

  function hideLoadingScreen() {
    if (dom.loadingScreen) dom.loadingScreen.hidden = true;
  }

  function showCaption(npcName, replyText) {
    if (!dom.caption) return;
    dom.caption.textContent = replyText ? `${npcName}: ${replyText}` :
      `${npcName}: …`;
  }

  // ----------------------------------------------------------------------
  // Renderer helpers (two THREE.WebGLRenderer instances, responsive resize)
  // ----------------------------------------------------------------------
  const MAX_PIXEL_RATIO = 2; // rendering budget cap

  function makeRenderer(canvas) {
    if (!canvas) return null;
    const renderer = new THREE.WebGLRenderer({
      canvas,
      antialias: true,
      alpha: true,
      powerPreference: 'high-performance',
    });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1,
      MAX_PIXEL_RATIO));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    return renderer;
  }

  function fitRendererToCanvas(renderer, camera, canvas) {
    if (!renderer || !canvas) return;
    const width = Math.max(1, canvas.clientWidth);
    const height = Math.max(1, canvas.clientHeight);
    const ratio = renderer.getPixelRatio();
    const needResize = canvas.width !== Math.floor(width * ratio) ||
      canvas.height !== Math.floor(height * ratio);
    if (needResize) {
      renderer.setSize(width, height, false);
      if (camera && camera.isPerspectiveCamera) {
        camera.aspect = width / height;
        camera.updateProjectionMatrix();
      }
    }
  }

  function sizeCanvas2d(canvas) {
    if (!canvas) return;
    const dpr = Math.min(window.devicePixelRatio || 1, MAX_PIXEL_RATIO);
    const width = Math.max(1, canvas.clientWidth);
    const height = Math.max(1, canvas.clientHeight);
    const targetW = Math.floor(width * dpr);
    const targetH = Math.floor(height * dpr);
    if (canvas.width !== targetW || canvas.height !== targetH) {
      canvas.width = targetW;
      canvas.height = targetH;
    }
    const ctx = canvas.getContext('2d');
    if (ctx) ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  function setup2dCanvasSizing() {
    const canvases = [dom.scopeCanvas, dom.rasterCanvas];
    canvases.forEach((canvas) => {
      if (!canvas) return;
      sizeCanvas2d(canvas);
      if ('ResizeObserver' in window) {
        const ro = new ResizeObserver(() => sizeCanvas2d(canvas));
        ro.observe(canvas);
      }
    });
    window.addEventListener('resize', () => canvases.forEach(sizeCanvas2d));
  }

  // ----------------------------------------------------------------------
  // Equalizer: #equalizer is a <div role="img">, not a canvas. Build
  // EQUALIZER_BARS real .eq-bar <div> children once and drive their height
  // from a real WebAudio AnalyserNode each frame.
  // ----------------------------------------------------------------------
  let equalizerBarEls = [];

  function buildEqualizerBars() {
    if (!dom.equalizer) return;
    dom.equalizer.innerHTML = '';
    equalizerBarEls = [];
    for (let i = 0; i < EQUALIZER_BARS; i++) {
      const bar = document.createElement('div');
      bar.className = 'eq-bar';
      bar.style.height = '8%';
      dom.equalizer.appendChild(bar);
      equalizerBarEls.push(bar);
    }
  }

  function drawEqualizer(analyser, freqData) {
    if (!equalizerBarEls.length || !analyser) return;
    analyser.getByteFrequencyData(freqData);
    const bucketSize = Math.floor(freqData.length / EQUALIZER_BARS) || 1;
    for (let i = 0; i < EQUALIZER_BARS; i++) {
      let sum = 0;
      const start = i * bucketSize;
      for (let j = 0; j < bucketSize; j++) sum += freqData[start + j] || 0;
      const avg = sum / bucketSize; // 0-255
      const pct = Math.max(8, Math.min(100, (avg / 255) * 100));
      equalizerBarEls[i].style.height = `${pct.toFixed(1)}%`;
    }
  }

  // ----------------------------------------------------------------------
  // Small canvas drawing helpers (original procedural graphics only)
  // ----------------------------------------------------------------------
  function roundedRectPath(ctx, x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
  }

  function wrapText(ctx, text, x, y, maxWidth, lineHeight) {
    if (!text) return;
    const words = String(text).split(' ');
    let line = '';
    let cursorY = y;
    for (const word of words) {
      const test = line ? `${line} ${word}` : word;
      if (ctx.measureText(test).width > maxWidth && line) {
        ctx.fillText(line, x, cursorY);
        line = word;
        cursorY += lineHeight;
      } else {
        line = test;
      }
    }
    if (line) ctx.fillText(line, x, cursorY);
  }

  function easeOutCubic(x) {
    return 1 - Math.pow(1 - x, 3);
  }

  function makeLabelSprite(text) {
    const canvas = document.createElement('canvas');
    canvas.width = 256;
    canvas.height = 64;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = 'rgba(10,10,14,0.65)';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.font = '700 32px sans-serif';
    ctx.fillStyle = '#f4f1e8';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(text, canvas.width / 2, canvas.height / 2);
    const texture = new THREE.CanvasTexture(canvas);
    texture.colorSpace = THREE.SRGBColorSpace;
    const material = new THREE.SpriteMaterial({ map: texture, depthWrite: false });
    const sprite = new THREE.Sprite(material);
    sprite.scale.set(1.6, 0.4, 1);
    return sprite;
  }

  // ----------------------------------------------------------------------
  // World scene: street, NPC places, skateboard, car, podium, phone prop
  // ----------------------------------------------------------------------
  const NPC_PLACES = [
    { key: 'zina', position: new THREE.Vector3(-16, 0, 6), color: 0xff6b8b },
    { key: 'artem', position: new THREE.Vector3(-8, 0, -6), color: 0x6bb6ff },
    { key: 'grigory', position: new THREE.Vector3(2, 0, 8), color: 0x8f6bff },
    { key: 'petrovich', position: new THREE.Vector3(10, 0, -7), color: 0xffb46b },
    { key: 'barsik', position: new THREE.Vector3(18, 0, 4), color: 0x6bffb0 },
    { key: 'danya', position: new THREE.Vector3(-2, 0, -2), color: 0xffe36b },
  ];

  // Chapter narrative metadata. Order below (object key insertion order) is
  // not what drives playback — the explicit CHAPTER_ORDER array does — but
  // is kept aligned with it for readability.
  const CHAPTER_INFO = {
    phone: { label: 'Phone check', target: new THREE.Vector3(4, 1.8, 5) },
    street: { label: 'Street cruise', target: new THREE.Vector3(0, 1.4, 0) },
    skate: { label: 'Skate spot', target: new THREE.Vector3(-2, 1.2, -1) },
    car: { label: 'Compact car', target: new THREE.Vector3(6, 1.6, -3) },
    podium: { label: 'Podium moment', target: new THREE.Vector3(-6, 1.8, 8) },
  };

  // Required order: phone -> street NPC -> skateboard -> car. Podium is an
  // optional fifth chapter kept at the end of the loop rather than removed.
  const CHAPTER_ORDER = ['phone', 'street', 'skate', 'car', 'podium'];

  // Close "hero" camera offsets relative to the fly's own position, so the
  // fly is reliably framed at a readable size no matter where in the (much
  // larger) street scene it currently is. Distances are in world units.
  const CHAPTER_CAMERA_OFFSET = {
    phone: new THREE.Vector3(0.6, 0.4, 1.1),
    street: new THREE.Vector3(1.0, 0.65, 1.8),
    skate: new THREE.Vector3(0.85, 0.55, 1.5),
    car: new THREE.Vector3(1.1, 0.65, 1.9),
    podium: new THREE.Vector3(0.9, 0.6, 1.6),
  };

  function buildStreet(scene) {
    const group = new THREE.Group();
    group.name = 'street';

    const roadMat = new THREE.MeshStandardMaterial({ color: 0x2b2b2f, roughness:
      0.95, metalness: 0.02 });
    const road = new THREE.Mesh(new THREE.PlaneGeometry(60, 20), roadMat);
    road.rotation.x = -Math.PI / 2;
    group.add(road);

    const sidewalkMat = new THREE.MeshStandardMaterial({ color: 0x9a9a92, roughness:
      0.9 });
    [-11, 11].forEach((z) => {
      const walk = new THREE.Mesh(new THREE.BoxGeometry(60, 0.2, 6), sidewalkMat);
      walk.position.set(0, 0.1, z);
      group.add(walk);
    });

    const buildingMat = new THREE.MeshStandardMaterial({ color: 0x51525e, roughness:
      0.8 });
    for (let i = -3; i <= 3; i++) {
      if (i === 0) continue;
      const height = 3 + Math.abs(i) * 1.4;
      const building = new THREE.Mesh(new THREE.BoxGeometry(4, height, 4),
        buildingMat);
      building.position.set(i * 8, height / 2, i % 2 === 0 ? 14 : -14);
      group.add(building);
    }

    const lampMat = new THREE.MeshStandardMaterial({ color: 0x2f2f33, metalness: 0.6,
      roughness: 0.35 });
    const lampHeadMat = new THREE.MeshStandardMaterial({
      color: 0xfff2c4,
      emissive: 0xffdd88,
      emissiveIntensity: 0.8,
    });
    for (let i = -2; i <= 2; i++) {
      const pole = new THREE.Mesh(new THREE.CylinderGeometry(0.05, 0.07, 3.2, 8),
        lampMat);
      pole.position.set(i * 12, 1.6, 9.5);
      group.add(pole);
      const head = new THREE.Mesh(new THREE.SphereGeometry(0.18, 10, 10), lampHeadMat);
      head.position.set(i * 12, 3.2, 9.5);
      group.add(head);
    }

    scene.add(group);
    return group;
  }

  function buildNpcMarkers(scene) {
    const group = new THREE.Group();
    group.name = 'npc-markers';
    NPC_PLACES.forEach((place) => {
      const marker = new THREE.Mesh(
        new THREE.CylinderGeometry(0.35, 0.45, 1.5, 12),
        new THREE.MeshStandardMaterial({ color: place.color, roughness: 0.5, metalness:
          0.1 }),
      );
      marker.position.copy(place.position).setY(0.75);
      group.add(marker);
      const label = makeLabelSprite(NPC_NAMES[place.key]);
      label.position.copy(place.position).setY(2.1);
      group.add(label);
    });
    scene.add(group);
    return group;
  }

  // Stylized skate deck with orange wheels. Plain colors only, no logos.
  function buildSkateboard() {
    const group = new THREE.Group();
    group.name = 'skateboard';
    const deckMat = new THREE.MeshStandardMaterial({ color: 0x2a2a2a, roughness: 0.7
    });
    const deck = new THREE.Mesh(new THREE.BoxGeometry(1.9, 0.06, 0.5), deckMat);
    deck.position.y = 0.22;
    group.add(deck);

    const truckMat = new THREE.MeshStandardMaterial({ color: 0xb8b8bc, metalness:
      0.7, roughness: 0.3 });
    const wheelMat = new THREE.MeshStandardMaterial({ color: 0xff7a1a, roughness:
      0.4, metalness: 0.05 });
    const wheelGeo = new THREE.CylinderGeometry(0.09, 0.09, 0.06, 16);

    const wheels = [];
    [[-0.7, -0.18], [-0.7, 0.18], [0.7, -0.18], [0.7, 0.18]].forEach(([x,
    z]) => {
      const truck = new THREE.Mesh(new THREE.BoxGeometry(0.32, 0.05, 0.05), truckMat);
      truck.position.set(x, 0.16, z);
      group.add(truck);
      const wheel = new THREE.Mesh(wheelGeo, wheelMat);
      wheel.rotation.z = Math.PI / 2;
      wheel.position.set(x, 0.09, z);
      group.add(wheel);
      wheels.push(wheel);
    });

    group.userData.wheels = wheels;
    return group;
  }

  // Stylized green compact car. Generic boxy shape, no badges/logos, no
  // trademark or real-brand claim of any kind.
  function buildCar() {
    const group = new THREE.Group();
    group.name = 'compact-car';
    const bodyMat = new THREE.MeshStandardMaterial({ color: 0x2e8b57, roughness:
      0.45, metalness: 0.35 });
    const glassMat = new THREE.MeshPhysicalMaterial({
      color: 0xbfe3ff,
      roughness: 0.05,
      metalness: 0,
      transmission: 0.7,
      thickness: 0.05,
      transparent: true,
      opacity: 0.55,
    });
    const wheelMat = new THREE.MeshStandardMaterial({ color: 0x161616, roughness: 0.8
    });

    const lower = new THREE.Mesh(new THREE.BoxGeometry(2.2, 0.55, 1.1), bodyMat);
    lower.position.y = 0.45;
    group.add(lower);

    const cabin = new THREE.Mesh(new THREE.BoxGeometry(1.2, 0.5, 1.02), glassMat);
    cabin.position.set(-0.1, 0.95, 0);
    group.add(cabin);

    const wheelGeo = new THREE.CylinderGeometry(0.28, 0.28, 0.22, 18);
    const wheels = [];
    [[-0.75, -0.55], [-0.75, 0.55], [0.75, -0.55], [0.75,
    0.55]].forEach(([x, z]) => {
      const wheel = new THREE.Mesh(wheelGeo, wheelMat);
      wheel.rotation.x = Math.PI / 2;
      wheel.position.set(x, 0.28, z);
      group.add(wheel);
      wheels.push(wheel);
    });

    group.userData.wheels = wheels;
    group.position.set(6, 0, -3);
    return group;
  }

  // Podium with an approximated reflective floor: MeshPhysicalMaterial with
  // a small procedural gradient used ONLY as an environment map (no
  // real-time mirror render target, no ray tracing).
  function buildPodium() {
    const group = new THREE.Group();
    group.name = 'podium';

    const envCanvas = document.createElement('canvas');
    envCanvas.width = 64;
    envCanvas.height = 32;
    const envCtx = envCanvas.getContext('2d');
    const grad = envCtx.createLinearGradient(0, 0, 0, 32);
    grad.addColorStop(0, '#3b4b63');
    grad.addColorStop(0.5, '#0d0f14');
    grad.addColorStop(1, '#1c1c1c');
    envCtx.fillStyle = grad;
    envCtx.fillRect(0, 0, 64, 32);
    const envTexture = new THREE.CanvasTexture(envCanvas);
    envTexture.mapping = THREE.EquirectangularReflectionMapping;
    envTexture.colorSpace = THREE.SRGBColorSpace;

    const floorMat = new THREE.MeshPhysicalMaterial({
      color: 0x14161c,
      roughness: 0.18,
      metalness: 0.6,
      envMap: envTexture,
      envMapIntensity: 1.1,
      clearcoat: 0.4,
      clearcoatRoughness: 0.25,
    });
    const floor = new THREE.Mesh(new THREE.CylinderGeometry(2.4, 2.6, 0.25, 32),
      floorMat);
    floor.position.y = 0.12;
    group.add(floor);

    const rimMat = new THREE.MeshStandardMaterial({ color: 0xd8b04a, metalness: 0.7,
      roughness: 0.3 });
    const rim = new THREE.Mesh(new THREE.TorusGeometry(2.5, 0.06, 12, 48), rimMat);
    rim.rotation.x = Math.PI / 2;
    rim.position.y = 0.25;
    group.add(rim);

    group.position.set(-6, 0, 8);
    return group;
  }

  // Phone prop whose screen is an original, procedurally drawn CanvasTexture.
  function buildPhoneProp() {
    const canvas = document.createElement('canvas');
    canvas.width = 360;
    canvas.height = 720;
    const ctx = canvas.getContext('2d');
    const texture = new THREE.CanvasTexture(canvas);
    texture.colorSpace = THREE.SRGBColorSpace;

    const bodyMat = new THREE.MeshStandardMaterial({ color: 0x141414, roughness: 0.4,
      metalness: 0.5 });
    const screenMat = new THREE.MeshBasicMaterial({ map: texture });

    const group = new THREE.Group();
    group.name = 'phone';
    const body = new THREE.Mesh(new THREE.BoxGeometry(0.42, 0.86, 0.04), bodyMat);
    group.add(body);
    const screen = new THREE.Mesh(new THREE.PlaneGeometry(0.38, 0.8), screenMat);
    screen.position.z = 0.021;
    group.add(screen);

    group.position.set(4, 1.6, 5);
    group.userData = { canvas, ctx, texture };
    return group;
  }

  function chapterColor(chapterKey, stop) {
    const palette = {
      street: ['#33475b', '#101820'],
      skate: ['#7a3b12', '#1a0f08'],
      car: ['#1f6b45', '#0c1a12'],
      podium: ['#4a3a1a', '#141008'],
      phone: ['#2a2a4a', '#0e0e1a'],
    };
    const colors = palette[chapterKey] || palette.street;
    return colors[stop];
  }

  function drawChapterGlyph(ctx, chapterKey, cx, cy, tSeconds) {
    ctx.save();
    ctx.translate(cx, cy);
    ctx.strokeStyle = 'rgba(255,255,255,0.85)';
    ctx.lineWidth = 4;
    const wobble = Math.sin(tSeconds * 2) * 4;
    switch (chapterKey) {
      case 'skate':
        roundedRectPath(ctx, -60, 10 + wobble, 120, 14, 8);
        ctx.stroke();
        ctx.beginPath();
        ctx.arc(-40, 30 + wobble, 10, 0, Math.PI * 2);
        ctx.arc(40, 30 + wobble, 10, 0, Math.PI * 2);
        ctx.stroke();
        break;
      case 'car':
        roundedRectPath(ctx, -70, -10 + wobble, 140, 40, 12);
        ctx.stroke();
        ctx.beginPath();
        ctx.arc(-40, 32 + wobble, 12, 0, Math.PI * 2);
        ctx.arc(40, 32 + wobble, 12, 0, Math.PI * 2);
        ctx.stroke();
        break;
      case 'podium':
        ctx.beginPath();
        ctx.ellipse(0, 20 + wobble, 55, 16, 0, 0, Math.PI * 2);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(0, -30 + wobble);
        ctx.lineTo(0, 4 + wobble);
        ctx.stroke();
        break;
      default:
        ctx.beginPath();
        ctx.moveTo(-60, 20 + wobble);
        ctx.lineTo(60, 20 + wobble);
        ctx.stroke();
        ctx.beginPath();
        ctx.arc(0, -10 + wobble, 22, 0, Math.PI * 2);
        ctx.stroke();
    }
    ctx.restore();
  }

  // Original, procedural short-form-video-style chrome for the phone chapter
  // only: drifting abstract shapes (not footage), simulated feed-position
  // dots, an @handle, and a beating heart/like counter. Purely decorative
  // canvas drawing — no downloaded video, no logos, no real engagement data.
  function drawPhoneFeedChrome(ctx, width, height, tSeconds, npcKey, captionText) {
    const hue = (tSeconds * 12) % 360;
    const gradient = ctx.createLinearGradient(0, 0, 0, height);
    gradient.addColorStop(0, `hsl(${hue}, 45%, 16%)`);
    gradient.addColorStop(1, `hsl(${(hue + 40) % 360}, 40%, 8%)`);
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, width, height);

    for (let i = 0; i < 5; i++) {
      const px = width * (0.2 + 0.6 * ((Math.sin(tSeconds * 0.4 + i * 1.7) + 1) / 2));
      const py = height * (0.15 + 0.6 * ((Math.cos(tSeconds * 0.33 + i * 2.1) + 1) / 2));
      const radius = 40 + 18 * Math.sin(tSeconds * 0.6 + i);
      ctx.beginPath();
      ctx.fillStyle = `hsla(${(hue + i * 40) % 360}, 70%, 60%, 0.18)`;
      ctx.arc(px, py, radius, 0, Math.PI * 2);
      ctx.fill();
    }

    const dotCount = 5;
    for (let i = 0; i < dotCount; i++) {
      const active = i === Math.floor(tSeconds / 2) % dotCount;
      ctx.fillStyle = active ? 'rgba(255,255,255,0.95)' : 'rgba(255,255,255,0.35)';
      ctx.fillRect(width - 10, 24 + i * 14, 4, active ? 18 : 10);
    }

    ctx.fillStyle = 'rgba(0,0,0,0.35)';
    ctx.fillRect(0, height - 140, width, 140);
    ctx.fillStyle = '#f4f1e8';
    ctx.font = '700 24px sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText(`@${npcKey}`, 20, height - 96);
    ctx.font = '400 18px sans-serif';
    ctx.fillStyle = 'rgba(255,255,255,0.85)';
    wrapText(ctx, captionText || '', 20, height - 66, width - 90, 24);

    const beat = 1 + 0.15 * Math.max(0, Math.sin(tSeconds * 3));
    ctx.save();
    ctx.translate(width - 40, height - 90);
    ctx.scale(beat, beat);
    ctx.fillStyle = '#ff5c7a';
    ctx.beginPath();
    ctx.moveTo(0, 6);
    ctx.bezierCurveTo(-14, -10, -2, -22, 0, -8);
    ctx.bezierCurveTo(2, -22, 14, -10, 0, 6);
    ctx.fill();
    ctx.restore();

    // Decorative UI chrome only — NOT a measurement of real engagement.
    ctx.fillStyle = 'rgba(255,255,255,0.85)';
    ctx.font = '400 14px sans-serif';
    ctx.textAlign = 'center';
    const likeCount = Math.floor(120 + tSeconds * 3) % 999;
    ctx.fillText(String(likeCount), width - 40, height - 55);
    ctx.textAlign = 'left';
  }

  function drawGenericChapterPanel(ctx, chapterKey, npcKey, tSeconds, width, height) {
    const gradient = ctx.createLinearGradient(0, 0, 0, height);
    gradient.addColorStop(0, chapterColor(chapterKey, 0));
    gradient.addColorStop(1, chapterColor(chapterKey, 1));
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, width, height);

    drawChapterGlyph(ctx, chapterKey, width / 2, height * 0.4, tSeconds);

    ctx.fillStyle = 'rgba(0,0,0,0.4)';
    ctx.fillRect(0, height - 120, width, 120);
    ctx.fillStyle = '#f4f1e8';
    ctx.font = '700 22px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(CHAPTER_INFO[chapterKey] ? CHAPTER_INFO[chapterKey].label : '',
      width / 2, height - 84);
    ctx.font = '400 16px sans-serif';
    ctx.fillStyle = 'rgba(255,255,255,0.8)';
    wrapText(ctx, `with ${NPC_NAMES[npcKey] || ''}`, width / 2, height - 54, width - 40,
      22);
    ctx.textAlign = 'left';
  }

  function drawPhoneScreen(phone, chapterKey, npcKey, tSeconds, swipeStart) {
    if (!phone || !phone.userData) return;
    const { ctx, canvas, texture } = phone.userData;
    const width = canvas.width;
    const height = canvas.height;
    const swipeElapsed = performance.now() - swipeStart;
    const swipeProgress = Math.min(1, swipeElapsed / 550);
    const slide = (1 - easeOutCubic(swipeProgress)) * width * 0.15;

    ctx.fillStyle = '#0b0d12';
    ctx.fillRect(0, 0, width, height);

    ctx.save();
    ctx.translate(0, slide);
    if (chapterKey === 'phone') {
      const captionText = dom.caption ? dom.caption.textContent : '';
      drawPhoneFeedChrome(ctx, width, height, tSeconds, npcKey, captionText);
    } else {
      drawGenericChapterPanel(ctx, chapterKey, npcKey, tSeconds, width, height);
    }
    ctx.restore();

    texture.needsUpdate = true;
  }

  // ----------------------------------------------------------------------
  // Brain point cloud (REAL MaleCNS soma coordinates; cosmetic pulse only)
  // ----------------------------------------------------------------------
  async function loadNeurons() {
    const response = await fetch(ENDPOINTS.neurons);
    if (!response.ok) {
      throw new Error(`GET ${ENDPOINTS.neurons} failed with status ${response.status}`);
    }
    const data = await response.json();
    if (!data || !Array.isArray(data.points) || data.points.length === 0) {
      throw new Error('Neuron response contained no points.');
    }
    return data;
  }

  const BRAIN_VERTEX_SHADER = `
    attribute float pulseSeed;
    uniform float uTime;
    uniform float uMood;
    varying float vPulse;
    void main() {
      float pulse = 0.5 + 0.5 * sin(uTime * (1.5 + uMood * 2.0) + pulseSeed * 6.2831);
      vPulse = pulse;
      vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
      gl_PointSize = (1.6 + pulse * 1.4) * (200.0 / -mvPosition.z);
      gl_Position = projectionMatrix * mvPosition;
    }
  `;

  const BRAIN_FRAGMENT_SHADER = `
    varying float vPulse;
    void main() {
      vec3 gold = vec3(1.0, 0.72, 0.13);
      vec3 cyan = vec3(0.0, 0.95, 1.0);
      vec3 color = mix(cyan, gold, vPulse);
      float d = length(gl_PointCoord - vec2(0.5));
      float alpha = smoothstep(0.5, 0.1, d);
      gl_FragColor = vec4(color, alpha * (0.5 + vPulse * 0.5));
    }
  `;

  function buildBrainPoints(neuronData) {
    const points = neuronData.points;
    const positions = new Float32Array(points.length * 3);
    const seeds = new Float32Array(points.length);
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity, minZ =
      Infinity, maxZ = -Infinity;
    points.forEach((p) => {
      minX = Math.min(minX, p.x); maxX = Math.max(maxX, p.x);
      minY = Math.min(minY, p.y); maxY = Math.max(maxY, p.y);
      minZ = Math.min(minZ, p.z); maxZ = Math.max(maxZ, p.z);
    });
    const spanX = Math.max(1e-6, maxX - minX);
    const spanY = Math.max(1e-6, maxY - minY);
    const spanZ = Math.max(1e-6, maxZ - minZ);
    const targetSpan = 6;
    const scale = targetSpan / Math.max(spanX, spanY, spanZ);
    const cx = (minX + maxX) / 2;
    const cy = (minY + maxY) / 2;
    const cz = (minZ + maxZ) / 2;

    points.forEach((p, i) => {
      positions[i * 3] = (p.x - cx) * scale;
      positions[i * 3 + 1] = (p.y - cy) * scale;
      positions[i * 3 + 2] = (p.z - cz) * scale;
      seeds[i] = Math.random();
    });

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('pulseSeed', new THREE.BufferAttribute(seeds, 1));

    const material = new THREE.ShaderMaterial({
      vertexShader: BRAIN_VERTEX_SHADER,
      fragmentShader: BRAIN_FRAGMENT_SHADER,
      uniforms: {
        uTime: { value: 0 },
        uMood: { value: 0.2 },
      },
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    });

    return new THREE.Points(geometry, material);
  }

  // ----------------------------------------------------------------------
  // Oscilloscope (modeled, not measured) 89.3 -> 125-140 Hz excited band
  // ----------------------------------------------------------------------
  const scopeState = { exciteUntil: 0 };

  function triggerScopeExcite() {
    scopeState.exciteUntil = performance.now() + 2400;
  }

  function drawOscilloscope(ctx, width, height, tSeconds) {
    const excited = performance.now() < scopeState.exciteUntil;
    const baseHz = 89.3;
    const excitedHz = 125 + 15 * Math.sin(tSeconds * 0.7);
    const freq = excited ? excitedHz : baseHz;

    ctx.fillStyle = 'rgba(4,10,14,0.9)';
    ctx.fillRect(0, 0, width, height);

    ctx.strokeStyle = 'rgba(0,243,255,0.15)';
    ctx.lineWidth = 1;
    for (let i = 1; i < 4; i++) {
      const y = (height / 4) * i;
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }

    ctx.strokeStyle = excited ? '#ffb703' : '#00f3ff';
    ctx.lineWidth = 2;
    ctx.beginPath();
    const amplitude = excited ? height * 0.32 : height * 0.2;
    for (let x = 0; x <= width; x += 2) {
      const phase = (x / width) * Math.PI * 2 * (freq / 20) + tSeconds * 4;
      const noise = excited ? (Math.random() - 0.5) * 6 : (Math.random() - 0.5) * 2;
      const y = height / 2 + Math.sin(phase) * amplitude + noise;
      if (x === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();

    if (dom.telemetryRate) {
      dom.telemetryRate.textContent = `${freq.toFixed(1)} Hz (modeled)`;
    }
    return freq;
  }

  // ----------------------------------------------------------------------
  // Raster (modeled 96-channel layout, 48 sampled rows drawn)
  // ----------------------------------------------------------------------
  const rasterState = { burstUntil: 0 };

  function triggerRasterBurst() {
    rasterState.burstUntil = performance.now() + 2000;
  }

  function drawRaster(ctx, width, height, tSeconds) {
    ctx.fillStyle = 'rgba(4,10,14,0.9)';
    ctx.fillRect(0, 0, width, height);
    const bursting = performance.now() < rasterState.burstUntil;
    const rowHeight = height / RASTER_ROWS;
    const density = bursting ? 0.22 : 0.06;
    ctx.fillStyle = bursting ? 'rgba(255,183,3,0.85)' : 'rgba(0,243,255,0.7)';
    for (let row = 0; row < RASTER_ROWS; row++) {
      for (let x = 0; x < width; x += 3) {
        const seed = Math.sin(row * 12.9898 + x * 78.233 + Math.floor(tSeconds * 8)) *
          43758.5453;
        const rnd = seed - Math.floor(seed);
        if (rnd < density) {
          ctx.fillRect(x, row * rowHeight, 2, Math.max(1, rowHeight - 1));
        }
      }
    }
  }

  // ----------------------------------------------------------------------
  // Audio: real WebAudio 140 BPM procedural loop feeding the equalizer
  // ----------------------------------------------------------------------
  const audioState = {
    context: null,
    analyser: null,
    freqData: null,
    started: false,
    stepIndex: 0,
    nextStepTime: 0,
  };

  function ensureAudioContext() {
    if (audioState.context) return audioState.context;
    const Ctx = window.AudioContext || window.webkitAudioContext;
    if (!Ctx) return null;
    const context = new Ctx();
    const analyser = context.createAnalyser();
    analyser.fftSize = 128;
    analyser.connect(context.destination);
    audioState.context = context;
    audioState.analyser = analyser;
    audioState.freqData = new Uint8Array(analyser.frequencyBinCount);
    return context;
  }

  function scheduleDrumStep(context, analyser, stepIndex, time) {
    const isKick = stepIndex % 4 === 0;
    const isHat = stepIndex % 2 === 1;
    if (isKick) {
      const osc = context.createOscillator();
      const gain = context.createGain();
      osc.type = 'sine';
      osc.frequency.setValueAtTime(120, time);
      osc.frequency.exponentialRampToValueAtTime(40, time + 0.12);
      gain.gain.setValueAtTime(0.9, time);
      gain.gain.exponentialRampToValueAtTime(0.001, time + 0.18);
      osc.connect(gain).connect(analyser);
      osc.start(time);
      osc.stop(time + 0.2);
    }
    if (isHat) {
      const bufferSize = context.sampleRate * 0.05;
      const buffer = context.createBuffer(1, bufferSize, context.sampleRate);
      const data = buffer.getChannelData(0);
      for (let i = 0; i < bufferSize; i++) data[i] = (Math.random() * 2 - 1) * 0.3;
      const noise = context.createBufferSource();
      noise.buffer = buffer;
      const gain = context.createGain();
      gain.gain.setValueAtTime(0.4, time);
      gain.gain.exponentialRampToValueAtTime(0.001, time + 0.05);
      noise.connect(gain).connect(analyser);
      noise.start(time);
    }
    if (stepIndex % 8 === 0) {
      const osc = context.createOscillator();
      const gain = context.createGain();
      osc.type = 'triangle';
      const notes = [220, 246.94, 261.63, 293.66];
      osc.frequency.setValueAtTime(notes[(stepIndex / 8) % notes.length], time);
      gain.gain.setValueAtTime(0.25, time);
      gain.gain.exponentialRampToValueAtTime(0.001, time + 0.4);
      osc.connect(gain).connect(analyser);
      osc.start(time);
      osc.stop(time + 0.45);
    }
  }

  function startSequencer() {
    const context = ensureAudioContext();
    if (!context || audioState.started) return;
    audioState.started = true;
    const stepDuration = 60 / 140 / 2; // 140 BPM, eighth notes
    audioState.nextStepTime = context.currentTime + 0.05;

    function tick() {
      if (!audioState.started) return;
      while (audioState.nextStepTime < context.currentTime + 0.2) {
        scheduleDrumStep(context, audioState.analyser, audioState.stepIndex,
          audioState.nextStepTime);
        audioState.stepIndex += 1;
        audioState.nextStepTime += stepDuration;
      }
      requestAnimationFrame(tick);
    }
    tick();
  }

  function stopSequencer() {
    audioState.started = false;
  }

  function wireSoundToggle() {
    if (!dom.soundToggle) return;
    dom.soundToggle.addEventListener('click', () => {
      const context = ensureAudioContext();
      if (!context) {
        setStatus('Web Audio is not available in this browser.', 'error');
        showErrorBanner('Web Audio is not available in this browser.', false);
        logEvent('Sound toggle failed: Web Audio unavailable');
        return;
      }
      const willEnable = dom.soundToggle.getAttribute('aria-pressed') !== 'true';
      if (willEnable) {
        if (context.state === 'suspended') context.resume();
        startSequencer();
        dom.soundToggle.setAttribute('aria-pressed', 'true');
        logEvent('Sound enabled (original 140 BPM procedural loop)');
      } else {
        stopSequencer();
        dom.soundToggle.setAttribute('aria-pressed', 'false');
        logEvent('Sound disabled');
      }
    });
  }

  // ----------------------------------------------------------------------
  // Web Speech (opt-in, ru-RU)
  // ----------------------------------------------------------------------
  function speak(text) {
    if (!('speechSynthesis' in window)) return;
    if (!audioState.started && audioState.context && audioState.context.state !==
      'running') {
      // Only speak after a user gesture has already unlocked audio via the
      // sound toggle; this keeps speech strictly opt-in.
      return;
    }
    try {
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.lang = 'ru-RU';
      window.speechSynthesis.speak(utterance);
    } catch (err) {
      // Speech is a non-critical enhancement; failures should not surface as
      // a fatal error.
      logEvent(`Speech synthesis unavailable: ${err.message}`);
    }
  }

  // ----------------------------------------------------------------------
  // Chat: POST /api/chat with throttling, timeout, exact NPC whitelist
  // ----------------------------------------------------------------------
  const chatState = { lastRequestAt: 0, abortController: null };

  function pickChapterLine(chapterKey) {
    const lines = CHAPTER_LINES[chapterKey] || CHAPTER_LINES.street;
    return lines[Math.floor(Math.random() * lines.length)];
  }

  async function requestNpcReply(npcKey, chapterKey) {
    const npcName = NPC_NAMES[npcKey];
    if (!npcName) return;
    const now = performance.now();
    if (now - chatState.lastRequestAt < CHAT_MIN_INTERVAL_MS) return;
    chatState.lastRequestAt = now;

    if (chatState.abortController) chatState.abortController.abort();
    const controller = new AbortController();
    chatState.abortController = controller;
    const timeoutId = setTimeout(() => controller.abort(), CHAT_TIMEOUT_MS);

    const message = pickChapterLine(chapterKey);
    showCaption(npcName, null);
    logEvent(`Chat request sent to ${npcName}`);

    try {
      const response = await fetch(ENDPOINTS.chat, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: npcName, message }),
        signal: controller.signal,
      });
      if (!response.ok) {
        throw new Error(`POST ${ENDPOINTS.chat} failed with status ${response.status}`);
      }
      const data = await response.json();
      const reply = data && (data.reply || data.text || data.message);
      if (!reply) throw new Error('Chat response contained no reply text.');
      showCaption(npcName, reply);
      logEvent(`${npcName} replied: “${String(reply).slice(0, 60)}${reply.length > 60 ?
        '…' : ''}”`);
      speak(reply);
      if (!fatalErrorActive) setStatus('Simulation running.', 'ready');
    } catch (err) {
      if (err && err.name === 'AbortError') return;
      const msg = `Chat with ${npcName} unavailable: ${err.message}`;
      showCaption(npcName, null);
      if (!fatalErrorActive) setStatus(msg, 'error');
      showErrorBanner(msg, false);
      logEvent(msg);
    } finally {
      clearTimeout(timeoutId);
    }
  }

  // ----------------------------------------------------------------------
  // Autopilot: fly movement, chapter cycling, motor telemetry
  // ----------------------------------------------------------------------
  const autopilot = {
    flyPos: new THREE.Vector3(0, 1.6, 0),
    flyHeading: 0,
    chapterIndex: 0,
    currentChapter: CHAPTER_ORDER[0],
    currentNpcKey: 'zina',
    mood: 'neutral',
    walk: 0.4,
    turn: 0,
    escape: 0,
    nextChangeAt: 0,
    swipeStart: 0,
    autoplayEnabled: true,
  };

  function pickNpcForChapter(chapterKey) {
    const info = CHAPTER_INFO[chapterKey];
    if (!info) return NPC_PLACES[0].key;
    let nearest = NPC_PLACES[0];
    let nearestDist = Infinity;
    NPC_PLACES.forEach((place) => {
      const dist = place.position.distanceTo(info.target);
      if (dist < nearestDist) {
        nearestDist = dist;
        nearest = place;
      }
    });
    return nearest.key;
  }

  function moodToScalar(mood) {
    const scale = {
      calm: 0.05,
      neutral: 0.2,
      curious: 0.35,
      excited: 0.6,
      agitated: 0.8,
      aggressive: 1.0,
    };
    return scale[mood] !== undefined ? scale[mood] : 0.2;
  }

  function renderChapterList() {
    if (!dom.chapterList) return;
    dom.chapterList.innerHTML = '';
    CHAPTER_ORDER.forEach((key, idx) => {
      const info = CHAPTER_INFO[key];
      const li = document.createElement('li');
      const button = document.createElement('button');
      button.type = 'button';
      button.className = 'chapter-item';
      button.textContent = `${idx + 1}. ${info.label}`;
      button.dataset.chapterKey = key;
      button.setAttribute('aria-current', 'false');
      button.addEventListener('click', () => {
        enterChapter(idx, { manual: true });
      });
      li.appendChild(button);
      dom.chapterList.appendChild(li);
    });
    updateChapterListUI();
  }

  function updateChapterListUI() {
    if (!dom.chapterList) return;
    const buttons = dom.chapterList.querySelectorAll('.chapter-item');
    buttons.forEach((button, idx) => {
      const isActive = idx === autopilot.chapterIndex;
      button.setAttribute('aria-current', isActive ? 'true' : 'false');
      button.classList.toggle('is-active', isActive);
    });
  }

  function enterChapter(index, options) {
    const opts = options || {};
    const total = CHAPTER_ORDER.length;
    autopilot.chapterIndex = ((Math.round(index) % total) + total) % total;
    const chapterKey = CHAPTER_ORDER[autopilot.chapterIndex];
    autopilot.currentChapter = chapterKey;
    autopilot.currentNpcKey = pickNpcForChapter(chapterKey);
    autopilot.mood = (chapterKey === 'car' || chapterKey === 'skate') ? 'excited' :
      'neutral';
    autopilot.swipeStart = performance.now();

    const info = CHAPTER_INFO[chapterKey];
    if (dom.chapterName) dom.chapterName.textContent = info.label;
    updateChapterListUI();

    triggerScopeExcite();
    triggerRasterBurst();
    requestNpcReply(autopilot.currentNpcKey, chapterKey);
    logEvent(`${opts.manual ? 'Manual jump' : 'Auto-advance'} to chapter: ${info.label}`);

    const [minMs, maxMs] = CHAPTER_DURATIONS[chapterKey] || [CHAPTER_MIN_MS,
      CHAPTER_MAX_MS];
    const delay = minMs + Math.random() * (maxMs - minMs);
    autopilot.nextChangeAt = performance.now() + delay;
  }

  function wireChapterControls() {
    if (dom.chapterPrev) {
      dom.chapterPrev.addEventListener('click', () => {
        enterChapter(autopilot.chapterIndex - 1, { manual: true });
      });
    }
    if (dom.chapterNext) {
      dom.chapterNext.addEventListener('click', () => {
        enterChapter(autopilot.chapterIndex + 1, { manual: true });
      });
    }
    if (dom.chapterAutoplay) {
      dom.chapterAutoplay.addEventListener('click', () => {
        autopilot.autoplayEnabled = !autopilot.autoplayEnabled;
        dom.chapterAutoplay.setAttribute('aria-pressed', autopilot.autoplayEnabled ?
          'true' : 'false');
        logEvent(`Autoplay ${autopilot.autoplayEnabled ? 'enabled' : 'disabled'}`);
        if (autopilot.autoplayEnabled) {
          const [minMs] = CHAPTER_DURATIONS[autopilot.currentChapter] || [
            CHAPTER_MIN_MS];
          autopilot.nextChangeAt = performance.now() + minMs;
        }
      });
    }
  }

  // Motor sliders are read-only telemetry: block direct interaction and
  // announce that via aria-readonly, while still visually animating each
  // frame from updateMotorTelemetry().
  function markMotorControlsReadOnly() {
    [dom.motorWalk, dom.motorTurn, dom.motorEscape].forEach((el) => {
      if (!el) return;
      el.setAttribute('aria-readonly', 'true');
      el.addEventListener('pointerdown', (e) => e.preventDefault());
      el.addEventListener('keydown', (e) => e.preventDefault());
    });
  }

  function updateMotorTelemetry() {
    if (dom.motorWalk) {
      const value = String(Math.round(autopilot.walk * 100));
      dom.motorWalk.value = value;
      dom.motorWalk.setAttribute('aria-valuenow', value);
    }
    if (dom.motorTurn) {
      const value = String(Math.round(autopilot.turn * 100));
      dom.motorTurn.value = value;
      dom.motorTurn.setAttribute('aria-valuenow', value);
    }
    if (dom.motorEscape) {
      const value = String(Math.round(autopilot.escape * 100));
      dom.motorEscape.value = value;
      dom.motorEscape.setAttribute('aria-valuenow', value);
    }
  }

  function updateFlyAutopilot(dt) {
    const info = CHAPTER_INFO[autopilot.currentChapter];
    if (!info) return;
    const toTarget = info.target.clone().sub(autopilot.flyPos);
    const distance = toTarget.length();
    const desiredHeading = Math.atan2(toTarget.x, toTarget.z);
    let deltaHeading = desiredHeading - autopilot.flyHeading;
    while (deltaHeading > Math.PI) deltaHeading -= Math.PI * 2;
    while (deltaHeading < -Math.PI) deltaHeading += Math.PI * 2;

    const turnRate = 2.2;
    autopilot.flyHeading += Math.max(-turnRate * dt, Math.min(turnRate * dt,
      deltaHeading));
    autopilot.turn = Math.max(-1, Math.min(1, deltaHeading / 0.6));

    const closeness = Math.min(1, distance / 6);
    autopilot.walk = 0.25 + closeness * 0.75;
    autopilot.escape = distance < 0.8 ? 0.15 : 0.0;

    const speed = autopilot.walk * 2.4;
    autopilot.flyPos.x += Math.sin(autopilot.flyHeading) * speed * dt;
    autopilot.flyPos.z += Math.cos(autopilot.flyHeading) * speed * dt;
    autopilot.flyPos.y += (info.target.y - autopilot.flyPos.y) * Math.min(1, dt * 1.5);

    updateMotorTelemetry();
  }

  // ----------------------------------------------------------------------
  // Prop animation (skateboard wheels, car wheels)
  // ----------------------------------------------------------------------
  function animateProps(skateboard, car, dt, chapterKey) {
    if (skateboard && skateboard.userData.wheels) {
      const speed = chapterKey === 'skate' ? 10 : 1.5;
      skateboard.userData.wheels.forEach((wheel) => { wheel.rotation.x += speed * dt; });
    }
    if (car && car.userData.wheels) {
      const speed = chapterKey === 'car' ? 6 : 0.4;
      car.userData.wheels.forEach((wheel) => { wheel.rotation.y += speed * dt; });
    }
  }

  // ----------------------------------------------------------------------
  // Main
  // ----------------------------------------------------------------------
  function main() {
    setStatus('Booting Flying Fly simulation…', 'loading');
    logEvent('Booting simulation');

    buildEqualizerBars();
    markMotorControlsReadOnly();
    wireSoundToggle();
    wireChapterControls();
    renderChapterList();
    setup2dCanvasSizing();

    // Brain scene
    const brainRenderer = makeRenderer(dom.brainCanvas);
    const brainScene = new THREE.Scene();
    const brainCamera = new THREE.PerspectiveCamera(50, 1, 0.1, 100);
    brainCamera.position.set(0, 0, 9);
    brainScene.add(new THREE.AmbientLight(0x445566, 1.2));
    let brainPoints = null;

    // World scene
    const worldRenderer = makeRenderer(dom.worldCanvas);
    const worldScene = new THREE.Scene();
    const worldCamera = new THREE.PerspectiveCamera(46, 1, 0.05, 200);
    worldScene.add(new THREE.HemisphereLight(0xbfd9ff, 0x1a1a1a, 0.9));
    const sun = new THREE.DirectionalLight(0xffffff, 1.1);
    sun.position.set(10, 16, 6);
    worldScene.add(sun);

    buildStreet(worldScene);
    buildNpcMarkers(worldScene);
    const skateboard = buildSkateboard();
    skateboard.position.set(-2, 0, -1);
    worldScene.add(skateboard);
    const car = buildCar();
    worldScene.add(car);
    const podium = buildPodium();
    worldScene.add(podium);
    const phoneProp = buildPhoneProp();
    worldScene.add(phoneProp);

    let flyRig = null;
    let flyOk = false;
    let neuronsOk = false;

    async function initFly() {
      try {
        const rig = await loadFly(THREE, (progress) => {
          setStatus(`Loading fly body model… ${Math.round((progress || 0) * 100)}%`,
            'loading');
        });
        flyRig = rig;
        worldScene.add(rig.group);
        flyOk = true;
        logEvent('Fly body model loaded');
      } catch (err) {
        flyOk = false;
        const msg = `Fly body model failed to load: ${err.message}`;
        logEvent(msg);
        showErrorBanner(msg, true);
      }
    }

    async function initNeurons() {
      if (dom.sourceLabel) dom.sourceLabel.textContent = 'Loading MaleCNS neuron sample…';
      try {
        const data = await loadNeurons();
        brainPoints = buildBrainPoints(data);
        brainScene.add(brainPoints);
        if (dom.sourceLabel) {
          dom.sourceLabel.textContent = `${data.total || data.points.length} neurons · ${
            data.source || 'MaleCNS v1.0'}`;
        }
        neuronsOk = true;
        logEvent(`Neuron sample loaded: ${data.points.length} points from ${data.source ||
          'MaleCNS v1.0'}`);
      } catch (err) {
        neuronsOk = false;
        if (dom.sourceLabel) dom.sourceLabel.textContent = 'Neuron data unavailable.';
        const msg = `Neuron data failed to load: ${err.message}`;
        logEvent(msg);
        showErrorBanner(msg, true);
      }
    }

    // Initial chapter + camera framing before the first frame, to avoid a
    // visible pop when the close-follow camera engages.
    enterChapter(0);
    const firstOffset = CHAPTER_CAMERA_OFFSET[autopilot.currentChapter] ||
      CHAPTER_CAMERA_OFFSET.street;
    worldCamera.position.copy(autopilot.flyPos.clone().add(firstOffset));
    worldCamera.lookAt(autopilot.flyPos);

    Promise.all([initFly(), initNeurons()]).then(() => {
      hideLoadingScreen();
      if (flyOk && neuronsOk) {
        setStatus('Simulation running.', 'ready');
      } else {
        const parts = [];
        if (!flyOk) parts.push('fly body model failed to load');
        if (!neuronsOk) parts.push('neuron data failed to load');
        const msg = `Simulation running with errors: ${parts.join('; ')}.`;
        fatalErrorActive = true;
        setStatus(msg, 'error');
        showErrorBanner(msg, true);
      }
    });

    const scopeCtx = dom.scopeCanvas ? dom.scopeCanvas.getContext('2d') : null;
    const rasterCtx = dom.rasterCanvas ? dom.rasterCanvas.getContext('2d') : null;

    let running = true;
    let lastFrame = performance.now();

    function pauseAnimation() {
      running = false;
      if (!fatalErrorActive) setStatus('Paused (tab hidden).', 'loading');
      logEvent('Paused: tab hidden');
    }

    function resumeAnimation() {
      if (running) return;
      running = true;
      lastFrame = performance.now();
      if (!fatalErrorActive) setStatus('Simulation running.', 'ready');
      logEvent('Resumed: tab visible');
      requestAnimationFrame(animate);
    }

    document.addEventListener('visibilitychange', () => {
      if (document.hidden) pauseAnimation(); else resumeAnimation();
    });

    function animate(now) {
      if (!running) return;
      const dt = Math.min(0.05, Math.max(0.001, (now - lastFrame) / 1000));
      lastFrame = now;
      const tSeconds = now / 1000;

      fitRendererToCanvas(brainRenderer, brainCamera, dom.brainCanvas);
      fitRendererToCanvas(worldRenderer, worldCamera, dom.worldCanvas);

      if (brainPoints && brainPoints.material.uniforms) {
        brainPoints.material.uniforms.uTime.value = tSeconds;
        brainPoints.material.uniforms.uMood.value = moodToScalar(autopilot.mood);
        brainPoints.rotation.y += dt * 0.08;
      }
      if (brainRenderer) brainRenderer.render(brainScene, brainCamera);

      if (autopilot.autoplayEnabled && now >= autopilot.nextChangeAt) {
        enterChapter(autopilot.chapterIndex + 1);
      }
      updateFlyAutopilot(dt);

      if (flyRig && flyRig.update) {
        flyRig.group.position.set(autopilot.flyPos.x, autopilot.flyPos.y,
          autopilot.flyPos.z);
        flyRig.group.rotation.y = autopilot.flyHeading;
        flyRig.update(tSeconds, autopilot.walk, autopilot.mood);
      }

      const camOffset = CHAPTER_CAMERA_OFFSET[autopilot.currentChapter] ||
        CHAPTER_CAMERA_OFFSET.street;
      const desiredCamPos = autopilot.flyPos.clone().add(camOffset);
      worldCamera.position.lerp(desiredCamPos, Math.min(1, dt * 2.2));
      worldCamera.lookAt(autopilot.flyPos.clone().add(new THREE.Vector3(0, 0.08, 0)));

      animateProps(skateboard, car, dt, autopilot.currentChapter);
      drawPhoneScreen(phoneProp, autopilot.currentChapter, autopilot.currentNpcKey,
        tSeconds, autopilot.swipeStart);

      if (worldRenderer) worldRenderer.render(worldScene, worldCamera);

      if (scopeCtx && dom.scopeCanvas) {
        drawOscilloscope(scopeCtx, dom.scopeCanvas.clientWidth,
          dom.scopeCanvas.clientHeight, tSeconds);
      }
      if (rasterCtx && dom.rasterCanvas) {
        drawRaster(rasterCtx, dom.rasterCanvas.clientWidth, dom.rasterCanvas.clientHeight,
          tSeconds);
      }
      if (audioState.analyser && audioState.freqData) {
        drawEqualizer(audioState.analyser, audioState.freqData);
      }

      requestAnimationFrame(animate);
    }

    requestAnimationFrame(animate);
  }

  function reportFatal(err) {
    const msg = `Fatal error: ${err && err.message ? err.message : err}`;
    fatalErrorActive = true;
    hideLoadingScreen();
    setStatus(msg, 'error');
    showErrorBanner(msg, true);
    logEvent(msg);
  }

  try {
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', () => {
        try { main(); } catch (err) { reportFatal(err); }
      });
    } else {
      main();
    }
  } catch (err) {
    reportFatal(err);
  }
})();
