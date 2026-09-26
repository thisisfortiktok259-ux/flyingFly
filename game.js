/**
 * game.js
 * ------------------------------------------------------------------------
 * Flying Fly — autonomous 3D fly simulation front end.
 *
 * This file is the ONLY frontend module this change touches. It coordinates
 * with an index.html built in parallel that declares an importmap for
 * "three" (three@0.169.0) and the following element ids:
 *   brain-canvas, world-canvas, scope-canvas, raster-canvas, equalizer,
 *   chapter-name, caption, source-label, telemetry-rate,
 *   motor-walk, motor-turn, motor-escape, status, sound-toggle
 *
 * Important framing, matching the backend README:
 *  - The fly is fully autonomous. There are no manual flight controls;
 *    the motor-walk / motor-turn / motor-escape elements are read-only
 *    telemetry reflecting what the autopilot is doing, not inputs. Chapter
 *    buttons, if present in the HTML, are optional shortcuts and are not
 *    required for the simulation to run.
 *  - GET /api/neurons returns REAL sampled MaleCNS soma coordinates. If
 *    that request fails, this file shows an explicit error and renders NO
 *    point cloud. It never fabricates or invents connectome data.
 *  - The oscilloscope, 96-channel raster, and the point-cloud pulse shader
 *    are procedurally modeled visual flourishes (loosely referencing a
 *    89.3 Hz baseline that ramps into a 125-140 Hz "excited" band, and
 *    PAM-style burst timing). None of this is a recording of real
 *    electrophysiology, none of it is labeled as an actual recording, and
 *    this project does not claim a whole-fly neural emulation anywhere.
 *  - The 28-bar equalizer is driven by a REAL WebAudio AnalyserNode
 *    listening to a synthesized, original 140 BPM procedural loop built
 *    from oscillators/noise buffers (no samples, no copyrighted audio).
 *    Audio only starts after the user clicks the sound toggle.
 *  - The fly body model is loaded from the real STL/model.json assets via
 *    fly_rig.js's loadFly(). If loading fails, the status text says so and
 *    NO primitive placeholder fly is drawn.
 *  - Phone "screen" content is drawn with an original, procedurally drawn
 *    CanvasTexture. No downloaded video or copyrighted media is used.
 *  - The skateboard, compact car, and podium are stylized, generic
 *    geometric props with no logos/decals and make no trademark or brand
 *    claims.
 *  - Chat uses POST /api/chat with the backend's exact NPC name whitelist
 *    (Cyrillic: Зина, Артем, Григорий, Петрович, Барсик, Даня).
 *    Replies are model-generated text (see server.py / README), shown as
 *    subtitles and, opt-in only after a user gesture, read aloud with the
 *    Web Speech API in ru-RU.
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

  const CHAPTER_MIN_MS = 6000;
  const CHAPTER_MAX_MS = 8000;

  const EQUALIZER_BARS = 28;
  const RASTER_ROWS = 48;        // sampled rows drawn out of...
  const RASTER_CHANNELS = 96;    // ...a modeled 96-channel layout (2 channels/row)

  const CHAPTER_LINES = {
    street: ['Что там видно с высоты?', 'Куда лучше свернуть на этой улице?'],
    skate: ['Покажешь трюк на доске?', 'Оранжевые колёса не подводят?'],
    car: ['Куда едет эта зелёная машина?', 'Усуеешь обогнать её на повороте?'],
    podium: ['Как ощущения на подиуме?', 'Отражение пола тебя не слепит?'],
    phone: ['Что там на экране телефона?', 'Что ты сейчас читаешь?'],
  };

  // ----------------------------------------------------------------------
  // DOM references (coordinated ids; every lookup is null-safe)
  // ----------------------------------------------------------------------

  const dom = {
    brainCanvas: document.getElementById('brain-canvas'),
    worldCanvas: document.getElementById('world-canvas'),
    scopeCanvas: document.getElementById('scope-canvas'),
    rasterCanvas: document.getElementById('raster-canvas'),
    equalizerCanvas: document.getElementById('equalizer'),
    chapterName: document.getElementById('chapter-name'),
    caption: document.getElementById('caption'),
    sourceLabel: document.getElementById('source-label'),
    telemetryRate: document.getElementById('telemetry-rate'),
    motorWalk: document.getElementById('motor-walk'),
    motorTurn: document.getElementById('motor-turn'),
    motorEscape: document.getElementById('motor-escape'),
    status: document.getElementById('status'),
    soundToggle: document.getElementById('sound-toggle'),
  };

  function setStatus(message, isError) {
    if (!dom.status) return;
    dom.status.textContent = message;
    if (dom.status.classList) dom.status.classList.toggle('error', !!isError);
  }

  function showCaption(npcName, replyText) {
    if (!dom.caption) return;
    dom.caption.textContent = replyText ? `${npcName}: ${replyText}` : `${npcName}: …`;
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
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, MAX_PIXEL_RATIO));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    return renderer;
  }

  function fitRendererToCanvas(renderer, camera, canvas) {
    if (!renderer || !canvas) return;
    const width = Math.max(1, canvas.clientWidth);
    const height = Math.max(1, canvas.clientHeight);
    const ratio = renderer.getPixelRatio();
    const needResize = canvas.width !== Math.floor(width * ratio) || canvas.height !== Math.floor(height * ratio);
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
    const canvases = [dom.scopeCanvas, dom.rasterCanvas, dom.equalizerCanvas];
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

  const CHAPTER_INFO = {
    street: { label: 'Street cruise', target: new THREE.Vector3(0, 1.4, 0), camera: new THREE.Vector3(0, 5, 14) },
    skate: { label: 'Skate spot', target: new THREE.Vector3(-2, 1.2, -1), camera: new THREE.Vector3(-2, 3.4, 5) },
    car: { label: 'Compact car', target: new THREE.Vector3(6, 1.6, -3), camera: new THREE.Vector3(6, 3.2, 3) },
    podium: { label: 'Podium moment', target: new THREE.Vector3(-6, 1.8, 8), camera: new THREE.Vector3(-6, 3.6, 12) },
    phone: { label: 'Phone check', target: new THREE.Vector3(4, 1.8, 5), camera: new THREE.Vector3(4, 2.1, 6.6) },
  };

  function buildStreet(scene) {
    const group = new THREE.Group();
    group.name = 'street';

    const roadMat = new THREE.MeshStandardMaterial({ color: 0x2b2b2f, roughness: 0.95, metalness: 0.02 });
    const road = new THREE.Mesh(new THREE.PlaneGeometry(60, 20), roadMat);
    road.rotation.x = -Math.PI / 2;
    group.add(road);

    const sidewalkMat = new THREE.MeshStandardMaterial({ color: 0x9a9a92, roughness: 0.9 });
    [-11, 11].forEach((z) => {
      const walk = new THREE.Mesh(new THREE.BoxGeometry(60, 0.2, 6), sidewalkMat);
      walk.position.set(0, 0.1, z);
      group.add(walk);
    });

    const buildingMat = new THREE.MeshStandardMaterial({ color: 0x51525e, roughness: 0.8 });
    for (let i = -3; i <= 3; i++) {
      if (i === 0) continue;
      const height = 3 + Math.abs(i) * 1.4;
      const building = new THREE.Mesh(new THREE.BoxGeometry(4, height, 4), buildingMat);
      building.position.set(i * 8, height / 2, i % 2 === 0 ? 14 : -14);
      group.add(building);
    }

    const lampMat = new THREE.MeshStandardMaterial({ color: 0x2f2f33, metalness: 0.6, roughness: 0.35 });
    const lampHeadMat = new THREE.MeshStandardMaterial({
      color: 0xfff2c4,
      emissive: 0xffdd88,
      emissiveIntensity: 0.8,
    });
    for (let i = -2; i <= 2; i++) {
      const pole = new THREE.Mesh(new THREE.CylinderGeometry(0.05, 0.07, 3.2, 8), lampMat);
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
        new THREE.MeshStandardMaterial({ color: place.color, roughness: 0.5, metalness: 0.1 }),
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
    const deckMat = new THREE.MeshStandardMaterial({ color: 0x2a2a2a, roughness: 0.7 });
    const deck = new THREE.Mesh(new THREE.BoxGeometry(1.9, 0.06, 0.5), deckMat);
    deck.position.y = 0.22;
    group.add(deck);

    const truckMat = new THREE.MeshStandardMaterial({ color: 0xb8b8bc, metalness: 0.7, roughness: 0.3 });
    const wheelMat = new THREE.MeshStandardMaterial({ color: 0xff7a1a, roughness: 0.4, metalness: 0.05 });
    const wheelGeo = new THREE.CylinderGeometry(0.09, 0.09, 0.06, 16);

    const wheels = [];
    [[-0.7, -0.18], [-0.7, 0.18], [0.7, -0.18], [0.7, 0.18]].forEach(([x, z]) => {
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
    const bodyMat = new THREE.MeshStandardMaterial({ color: 0x2e8b57, roughness: 0.45, metalness: 0.35 });
    const glassMat = new THREE.MeshPhysicalMaterial({
      color: 0xbfe3ff,
      roughness: 0.05,
      metalness: 0,
      transmission: 0.7,
      thickness: 0.05,
      transparent: true,
      opacity: 0.55,
    });
    const wheelMat = new THREE.MeshStandardMaterial({ color: 0x161616, roughness: 0.8 });

    const lower = new THREE.Mesh(new THREE.BoxGeometry(2.2, 0.55, 1.1), bodyMat);
    lower.position.y = 0.45;
    group.add(lower);

    const cabin = new THREE.Mesh(new THREE.BoxGeometry(1.2, 0.5, 1.02), glassMat);
    cabin.position.set(-0.1, 0.95, 0);
    group.add(cabin);

    const wheelGeo = new THREE.CylinderGeometry(0.28, 0.28, 0.22, 18);
    const wheels = [];
    [[-0.75, -0.55], [-0.75, 0.55], [0.75, -0.55], [0.75, 0.55]].forEach(([x, z]) => {
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
    const floor = new THREE.Mesh(new THREE.CylinderGeometry(2.4, 2.6, 0.25, 32), floorMat);
    floor.position.y = 0.12;
    group.add(floor);

    const rimMat = new THREE.MeshStandardMaterial({ color: 0xd8b04a, metalness: 0.7, roughness: 0.3 });
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

    const bodyMat = new THREE.MeshStandardMaterial({ color: 0x141414, roughness: 0.4, metalness: 0.5 });
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
      case 'phone':
        roundedRectPath(ctx, -30, -50 + wobble, 60, 100, 10);
        ctx.stroke();
        break;
      default:
        ctx.beginPath();
        ctx.moveTo(-60, 20 + wobble);
        ctx.lineTo(60, 20 + wobble);
        ctx.moveTo(-40, -10 + wobble);
        ctx.lineTo(40, -10 + wobble);
        ctx.stroke();
    }
    ctx.restore();
  }

  function drawPhoneScreen(phone, chapterKey, npcKey, tSeconds, swipeStart) {
    const { ctx, canvas, texture } = phone.userData;
    const width = canvas.width;
    const height = canvas.height;
    const swipeElapsed = performance.now() - swipeStart;
    const swipeProgress = Math.min(1, swipeElapsed / 550);
    const slide = (1 - easeOutCubic(swipeProgress)) * width;

    ctx.fillStyle = '#0b0d12';
    ctx.fillRect(0, 0, width, height);

    ctx.save();
    ctx.translate(slide, 0);

    const info = CHAPTER_INFO[chapterKey] || CHAPTER_INFO.street;
    const gradient = ctx.createLinearGradient(0, 0, 0, height);
    gradient.addColorStop(0, chapterColor(chapterKey, 0));
    gradient.addColorStop(1, chapterColor(chapterKey, 1));
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, width, height * 0.55);

    ctx.fillStyle = 'rgba(255,255,255,0.9)';
    ctx.font = '700 34px sans-serif';
    ctx.fillText(info.label, 24, 70);

    ctx.font = '400 22px sans-serif';
    ctx.fillStyle = 'rgba(255,255,255,0.75)';
    ctx.fillText(NPC_NAMES[npcKey] || '', 24, 110);

    drawChapterGlyph(ctx, chapterKey, width / 2, height * 0.32, tSeconds);

    ctx.fillStyle = '#171a22';
    ctx.fillRect(0, height * 0.55, width, height * 0.45);
    ctx.fillStyle = '#f4f1e8';
    ctx.font = '400 20px sans-serif';
    wrapText(ctx, dom.caption ? dom.caption.textContent : '', 24, height * 0.62, width - 48, 26);

    ctx.restore();
    texture.needsUpdate = true;
  }

  // ----------------------------------------------------------------------
  // Neuron point cloud (real MaleCNS sample from GET /api/neurons)
  // ----------------------------------------------------------------------

  async function loadNeurons() {
    let response;
    try {
      response = await fetch(ENDPOINTS.neurons);
    } catch (err) {
      throw new Error(`network error reaching ${ENDPOINTS.neurons}: ${err.message}`);
    }
    if (!response.ok) {
      let detail = `HTTP ${response.status}`;
      try {
        const body = await response.json();
        if (body && body.error) detail = body.error;
      } catch (_ignored) {
        // response body was not JSON; keep the HTTP status detail
      }
      throw new Error(detail);
    }
    const data = await response.json();
    if (!data || !Array.isArray(data.points) || data.points.length === 0) {
      throw new Error('neuron sample response had no usable points');
    }
    return data;
  }

  // Builds a THREE.Points cloud from real sampled soma coordinates. The
  // vertex/fragment shaders below only add a cosmetic time/mood pulse for
  // visual life — they do NOT compute, claim, or display any real firing
  // rate or neural activity. This is explicitly documented so nobody reads
  // the animated pulse as a live recording.
  function buildBrainPoints(data) {
    const count = data.points.length;
    const positions = new Float32Array(count * 3);
    const colorMix = new Float32Array(count);
    const seed = new Float32Array(count);

    for (let i = 0; i < count; i++) {
      const point = data.points[i];
      positions[i * 3] = point.x || 0;
      positions[i * 3 + 1] = point.y || 0;
      positions[i * 3 + 2] = point.z || 0;
      const side = (point.side || '').toString().toUpperCase();
      colorMix[i] = side === 'L' ? 0.15 : side === 'R' ? 0.85 : 0.5;
      seed[i] = Math.random() * Math.PI * 2;
    }

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('aColorMix', new THREE.BufferAttribute(colorMix, 1));
    geometry.setAttribute('aSeed', new THREE.BufferAttribute(seed, 1));

    const material = new THREE.ShaderMaterial({
      uniforms: {
        uTime: { value: 0 },
        uMood: { value: 1.0 },
        uGold: { value: new THREE.Color(0xffd166) },
        uCyan: { value: new THREE.Color(0x4be3ff) },
        uPointSize: { value: 2.2 },
      },
      vertexShader: `
        uniform float uTime;
        uniform float uMood;
        uniform float uPointSize;
        attribute float aColorMix;
        attribute float aSeed;
        varying float vMix;
        varying float vPulse;
        void main() {
          vMix = aColorMix;
          // Cosmetic phase only, tied to time and current mood/swipe state.
          // This is NOT a recording or simulation of real neural firing.
          vPulse = 0.55 + 0.45 * sin(uTime * (0.6 + uMood * 0.9) + aSeed);
          vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
          gl_PointSize = uPointSize * vPulse * (300.0 / max(1.0, -mvPosition.z));
          gl_Position = projectionMatrix * mvPosition;
        }
      `,
      fragmentShader: `
        uniform vec3 uGold;
        uniform vec3 uCyan;
        varying float vMix;
        varying float vPulse;
        void main() {
          vec2 centered = gl_PointCoord - vec2(0.5);
          float dist = length(centered);
          if (dist > 0.5) discard;
          vec3 color = mix(uCyan, uGold, vMix);
          float alpha = (1.0 - dist * 2.0) * (0.35 + 0.65 * vPulse);
          gl_FragColor = vec4(color, alpha);
        }
      `,
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    });

    const points = new THREE.Points(geometry, material);
    points.frustumCulled = false;
    return points;
  }

  // ----------------------------------------------------------------------
  // Oscilloscope: modeled baseline 89.3 Hz ramping to a 125-140 Hz
  // "excited" band. Explicitly modeled, not a real recording.
  // ----------------------------------------------------------------------

  let scopeFrequency = 89.3;
  let scopeExciteUntil = 0;

  function triggerScopeExcite(durationMs) {
    scopeExciteUntil = performance.now() + (durationMs || 900);
  }

  function updateScopeFrequency(nowMs) {
    const excited = nowMs < scopeExciteUntil;
    const target = excited ? 125 + Math.random() * 15 : 89.3;
    scopeFrequency += (target - scopeFrequency) * 0.06;
    return scopeFrequency;
  }

  function drawOscilloscope(ctx, canvas, nowMs, tSeconds) {
    if (!ctx || !canvas) return scopeFrequency;
    const width = canvas.clientWidth || canvas.width;
    const height = canvas.clientHeight || canvas.height;
    ctx.clearRect(0, 0, width, height);

    ctx.strokeStyle = 'rgba(255,209,102,0.15)';
    ctx.beginPath();
    ctx.moveTo(0, height / 2);
    ctx.lineTo(width, height / 2);
    ctx.stroke();

    const freq = updateScopeFrequency(nowMs);
    ctx.beginPath();
    ctx.strokeStyle = '#7dfcff';
    ctx.lineWidth = 2;
    const cycles = 4;
    for (let x = 0; x <= width; x++) {
      const phase = (x / width) * cycles * Math.PI * 2 + tSeconds * freq * 0.02;
      const amplitude = (height / 2) * 0.7;
      const y = height / 2 + Math.sin(phase) * amplitude * (0.85 + 0.15 * Math.sin(tSeconds * 1.7));
      if (x === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
    return freq;
  }

  // ----------------------------------------------------------------------
  // 96-channel raster (48 sampled rows), event-driven ticker. Loosely
  // modeled on PAM-style burst timing; NOT a recording of real
  // electrophysiology, and no whole-fly emulation is claimed.
  // ----------------------------------------------------------------------

  const rasterEvents = [];

  function triggerRasterBurst(count) {
    const n = count || 24;
    for (let i = 0; i < n; i++) {
      rasterEvents.push({ row: Math.floor(Math.random() * RASTER_ROWS), age: 0 });
    }
  }

  function drawRaster(ctx, canvas, dtSeconds) {
    if (!ctx || !canvas) return;
    const width = canvas.clientWidth || canvas.width;
    const height = canvas.clientHeight || canvas.height;
    ctx.fillStyle = 'rgba(6,8,12,0.35)';
    ctx.fillRect(0, 0, width, height);

    const rowHeight = height / RASTER_ROWS;

    // Low-rate ambient ticks (each row represents 2 of the modeled 96
    // channels, sampled down to RASTER_ROWS for display).
    if (Math.random() < 0.35) {
      rasterEvents.push({ row: Math.floor(Math.random() * RASTER_ROWS), age: 0 });
    }

    ctx.fillStyle = '#ffd166';
    for (let i = rasterEvents.length - 1; i >= 0; i--) {
      const ev = rasterEvents[i];
      ev.age += dtSeconds;
      const x = width - ev.age * (width / 2.4);
      if (x < 0) {
        rasterEvents.splice(i, 1);
        continue;
      }
      const alpha = Math.max(0, 1 - ev.age / 2.4);
      ctx.globalAlpha = alpha;
      ctx.fillRect(x, ev.row * rowHeight + 1, 2, Math.max(1, rowHeight - 2));
    }
    ctx.globalAlpha = 1;
  }

  // ----------------------------------------------------------------------
  // Audio: opt-in procedural 140 BPM loop feeding a real AnalyserNode that
  // drives 28 equalizer bars. Purely synthesized (oscillators + noise
  // buffers); no samples or copyrighted audio.
  // ----------------------------------------------------------------------

  let audioCtx = null;
  let analyser = null;
  let masterGain = null;
  let sequencerTimer = null;
  let audioEnabled = false;
  let speechEnabled = false;
  const freqData = new Uint8Array(32);

  function scheduleDrumStep(time, stepIndex) {
    const kickOsc = audioCtx.createOscillator();
    const kickGain = audioCtx.createGain();
    kickOsc.type = 'sine';
    kickOsc.frequency.setValueAtTime(120, time);
    kickOsc.frequency.exponentialRampToValueAtTime(45, time + 0.12);
    kickGain.gain.setValueAtTime(0, time);
    kickGain.gain.linearRampToValueAtTime(0.9, time + 0.005);
    kickGain.gain.exponentialRampToValueAtTime(0.001, time + 0.18);
    kickOsc.connect(kickGain).connect(masterGain);
    kickOsc.start(time);
    kickOsc.stop(time + 0.2);

    if (stepIndex % 2 === 1) {
      const bufferSize = Math.floor(audioCtx.sampleRate * 0.05);
      const buffer = audioCtx.createBuffer(1, bufferSize, audioCtx.sampleRate);
      const data = buffer.getChannelData(0);
      for (let i = 0; i < bufferSize; i++) data[i] = (Math.random() * 2 - 1) * (1 - i / bufferSize);
      const hat = audioCtx.createBufferSource();
      hat.buffer = buffer;
      const hatGain = audioCtx.createGain();
      hatGain.gain.value = 0.25;
      hat.connect(hatGain).connect(masterGain);
      hat.start(time);
    }
  }

  function startSequencer() {
    const bpm = 140;
    const stepSeconds = 60 / bpm / 2; // eighth notes
    let stepIndex = 0;
    let nextStepTime = audioCtx.currentTime + 0.05;

    function tick() {
      if (!audioCtx) return;
      if (audioCtx.state !== 'running') {
        sequencerTimer = setTimeout(tick, 250);
        return;
      }
      while (nextStepTime < audioCtx.currentTime + 0.2) {
        scheduleDrumStep(nextStepTime, stepIndex);
        stepIndex += 1;
        nextStepTime += stepSeconds;
      }
      sequencerTimer = setTimeout(tick, 60);
    }
    tick();
  }

  function ensureAudioContext() {
    if (audioCtx) {
      audioCtx.resume().catch(() => {});
      return;
    }
    const AudioContextCtor = window.AudioContext || window.webkitAudioContext;
    if (!AudioContextCtor) return;
    audioCtx = new AudioContextCtor();
    masterGain = audioCtx.createGain();
    masterGain.gain.value = 0.22;
    analyser = audioCtx.createAnalyser();
    analyser.fftSize = 64; // 32 bins; the equalizer uses the first 28
    masterGain.connect(analyser);
    analyser.connect(audioCtx.destination);
    startSequencer();
  }

  function suspendAudio() {
    if (audioCtx) audioCtx.suspend().catch(() => {});
    if (sequencerTimer) {
      clearTimeout(sequencerTimer);
      sequencerTimer = null;
    }
  }

  function drawEqualizer(ctx, canvas) {
    if (!ctx || !canvas) return;
    const width = canvas.clientWidth || canvas.width;
    const height = canvas.clientHeight || canvas.height;
    ctx.clearRect(0, 0, width, height);
    const barWidth = width / EQUALIZER_BARS;

    if (analyser && audioEnabled && audioCtx && audioCtx.state === 'running') {
      analyser.getByteFrequencyData(freqData);
    } else {
      freqData.fill(0);
    }

    for (let i = 0; i < EQUALIZER_BARS; i++) {
      const value = freqData[i] || 0;
      const barHeight = Math.max(2, (value / 255) * height);
      const hue = 45 + (i / EQUALIZER_BARS) * 140;
      ctx.fillStyle = `hsl(${hue}, 85%, ${audioEnabled ? 55 : 25}%)`;
      ctx.fillRect(i * barWidth + 1, height - barHeight, Math.max(1, barWidth - 2), barHeight);
    }
  }

  function speak(text) {
    if (!speechEnabled || !text) return;
    if (!('speechSynthesis' in window)) return;
    try {
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.lang = 'ru-RU';
      const voices = window.speechSynthesis.getVoices();
      const ruVoice = voices.find((v) => v.lang && v.lang.toLowerCase().startsWith('ru'));
      if (ruVoice) utterance.voice = ruVoice;
      window.speechSynthesis.speak(utterance);
    } catch (_err) {
      // Speech synthesis is best-effort; failures never break the scene.
    }
  }

  function wireSoundToggle() {
    if (!dom.soundToggle) return;
    dom.soundToggle.addEventListener(
      'click',
      () => {
        // Web Speech / WebAudio only ever start from this user gesture.
        if (!audioEnabled) {
          audioEnabled = true;
          speechEnabled = true;
          ensureAudioContext();
          dom.soundToggle.setAttribute('aria-pressed', 'true');
        } else {
          audioEnabled = false;
          speechEnabled = false;
          suspendAudio();
          if ('speechSynthesis' in window) window.speechSynthesis.cancel();
          dom.soundToggle.setAttribute('aria-pressed', 'false');
        }
      },
      { passive: true },
    );
  }

  // ----------------------------------------------------------------------
  // Chat: POST /api/chat with the exact backend NPC whitelist, throttled.
  // ----------------------------------------------------------------------

  function pickChapterLine(chapterKey) {
    const lines = CHAPTER_LINES[chapterKey] || CHAPTER_LINES.street;
    return lines[Math.floor(Math.random() * lines.length)];
  }

  const chatState = {
    lastChatAt: 0,
    abortController: null,
  };

  async function requestNpcReply(npcKey, chapterKey) {
    const npcName = NPC_NAMES[npcKey];
    if (!npcName) return; // never send an un-whitelisted name

    const now = performance.now();
    if (now - chatState.lastChatAt < CHAT_MIN_INTERVAL_MS) return; // request throttling
    if (document.hidden) return; // do not fire background requests while hidden
    chatState.lastChatAt = now;

    if (chatState.abortController) chatState.abortController.abort();
    const controller = new AbortController();
    chatState.abortController = controller;
    const timeoutId = setTimeout(() => controller.abort(), CHAT_TIMEOUT_MS);

    const message = pickChapterLine(chapterKey);

    try {
      const response = await fetch(ENDPOINTS.chat, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ npc: npcName, message }),
        signal: controller.signal,
      });
      let data = null;
      try {
        data = await response.json();
      } catch (_ignored) {
        // non-JSON body; handled by the !data check below
      }
      if (!response.ok || !data || typeof data.reply !== 'string') {
        const errMsg = (data && data.error) || `HTTP ${response.status}`;
        throw new Error(errMsg);
      }
      showCaption(npcName, data.reply);
      speak(data.reply);
    } catch (err) {
      if (err && err.name === 'AbortError') return;
      showCaption(npcName, null);
      setStatus(`Chat with ${npcName} unavailable: ${err.message}`, true);
    } finally {
      clearTimeout(timeoutId);
    }
  }

  // ----------------------------------------------------------------------
  // Autopilot: chooses chapters/NPCs and steers the fly. No manual controls
  // exist; motor-walk/turn/escape are read-only telemetry of this autopilot.
  // ----------------------------------------------------------------------

  const autopilot = {
    chapters: ['street', 'skate', 'car', 'podium', 'phone'],
    chapterIndex: -1,
    currentChapter: 'street',
    currentNpcKey: 'zina',
    nextChangeAt: 0,
    swipeStart: 0,
    mood: 'neutral',
    flyPos: new THREE.Vector3(0, 1.6, 0),
    heading: 0,
    walk: 0,
    turn: 0,
    escape: 0,
  };

  function pickNpcForChapter(chapterKey) {
    const info = CHAPTER_INFO[chapterKey];
    let best = NPC_PLACES[0];
    let bestDist = Infinity;
    NPC_PLACES.forEach((place) => {
      const d = place.position.distanceTo(info.target);
      if (d < bestDist) {
        bestDist = d;
        best = place;
      }
    });
    return best.key;
  }

  function advanceChapter() {
    autopilot.chapterIndex = (autopilot.chapterIndex + 1) % autopilot.chapters.length;
    const chapterKey = autopilot.chapters[autopilot.chapterIndex];
    autopilot.currentChapter = chapterKey;
    autopilot.currentNpcKey = pickNpcForChapter(chapterKey);
    autopilot.mood = chapterKey === 'car' || chapterKey === 'skate' ? 'excited' : 'neutral';
    autopilot.swipeStart = performance.now();

    const info = CHAPTER_INFO[chapterKey];
    if (dom.chapterName) dom.chapterName.textContent = info.label;

    triggerScopeExcite();
    triggerRasterBurst();
    requestNpcReply(autopilot.currentNpcKey, chapterKey);

    const delay = CHAPTER_MIN_MS + Math.random() * (CHAPTER_MAX_MS - CHAPTER_MIN_MS);
    autopilot.nextChangeAt = performance.now() + delay;
  }

  function updateFlyAutopilot(dtSeconds, tSeconds, flyRig) {
    const info = CHAPTER_INFO[autopilot.currentChapter] || CHAPTER_INFO.street;
    const target = info.target;
    const toTarget = target.clone().sub(autopilot.flyPos);
    toTarget.y = 0;
    const distance = toTarget.length();

    const desiredHeading = Math.atan2(toTarget.x, toTarget.z);
    let headingDelta = desiredHeading - autopilot.heading;
    headingDelta = Math.atan2(Math.sin(headingDelta), Math.cos(headingDelta));

    autopilot.turn = THREE.MathUtils.clamp(headingDelta / Math.PI, -1, 1);
    autopilot.heading += headingDelta * Math.min(1, dtSeconds * 2.2);

    const closeness = THREE.MathUtils.clamp(distance / 4, 0, 1);
    autopilot.walk = 0.25 + closeness * 0.75;
    autopilot.escape =
      distance < 0.6
        ? Math.min(1, autopilot.escape + dtSeconds * 1.5)
        : Math.max(0, autopilot.escape - dtSeconds * 2);

    const speed = 1.1 * autopilot.walk;
    autopilot.flyPos.x += Math.sin(autopilot.heading) * speed * dtSeconds;
    autopilot.flyPos.z += Math.cos(autopilot.heading) * speed * dtSeconds;
    autopilot.flyPos.y = target.y + Math.sin(tSeconds * 2.3) * 0.05;

    if (flyRig) {
      flyRig.group.position.copy(autopilot.flyPos);
      flyRig.group.rotation.y = autopilot.heading;
      flyRig.update(tSeconds, autopilot.walk, autopilot.mood);
    }
  }

  function updateMotorTelemetry() {
    if (dom.motorWalk) dom.motorWalk.value = autopilot.walk.toFixed(2);
    if (dom.motorTurn) dom.motorTurn.value = autopilot.turn.toFixed(2);
    if (dom.motorEscape) dom.motorEscape.value = autopilot.escape.toFixed(2);
  }

  function moodToScalar(mood) {
    if (mood === 'excited') return 1.6;
    if (mood === 'calm') return 0.7;
    return 1.0;
  }

  // ----------------------------------------------------------------------
  // Prop animation (skate wheels roll, car wheel turns and drives a bit)
  // ----------------------------------------------------------------------

  function animateProps(t, dt, skateboard, car) {
    if (skateboard && skateboard.userData.wheels) {
      skateboard.userData.wheels.forEach((wheel) => {
        wheel.rotation.x = t * 6;
      });
    }
    if (car && car.userData.wheels) {
      const driving = autopilot.currentChapter === 'car';
      const wheelSpeed = driving ? 4 : 0.4;
      car.userData.wheels.forEach((wheel) => {
        wheel.rotation.x += wheelSpeed * dt;
      });
      if (driving) {
        car.position.x = 6 + Math.sin(t * 0.3) * 2.4;
      }
    }
  }

  // ----------------------------------------------------------------------
  // Main application
  // ----------------------------------------------------------------------

  async function main() {
    setStatus('Booting Flying Fly simulation…');

    // World scene ------------------------------------------------------
    const worldScene = new THREE.Scene();
    worldScene.background = new THREE.Color(0x07080b);
    const worldCamera = new THREE.PerspectiveCamera(50, 1, 0.1, 200);
    worldCamera.position.set(0, 5, 14);

    const hemi = new THREE.HemisphereLight(0xbfd7ff, 0x1a1710, 0.9);
    worldScene.add(hemi);
    const sun = new THREE.DirectionalLight(0xffffff, 1.1);
    sun.position.set(8, 12, 6);
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

    const worldRenderer = makeRenderer(dom.worldCanvas);
    if (!worldRenderer) setStatus('World canvas element is missing; 3D scene disabled.', true);

    // Brain scene ---------------------------------------------------------
    const brainScene = new THREE.Scene();
    const brainCamera = new THREE.PerspectiveCamera(45, 1, 0.01, 20);
    brainCamera.position.set(0, 0, 2.6);
    const brainRenderer = makeRenderer(dom.brainCanvas);
    if (!brainRenderer) setStatus('Brain canvas element is missing; neuron view disabled.', true);

    let brainPoints = null;

    // 2D canvases -----------------------------------------------------
    const scopeCtx = dom.scopeCanvas ? dom.scopeCanvas.getContext('2d') : null;
    const rasterCtx = dom.rasterCanvas ? dom.rasterCanvas.getContext('2d') : null;
    const equalizerCtx = dom.equalizerCanvas ? dom.equalizerCanvas.getContext('2d') : null;
    setup2dCanvasSizing();
    wireSoundToggle();

    // Fly body model (real STL assets via fly_rig.js; no fallback fly) --
    let flyRig = null;
    async function initFly() {
      try {
        setStatus('Loading fly body model…');
        flyRig = await loadFly(THREE, (progress) => {
          setStatus(`Loading fly body model… ${Math.round((progress || 0) * 100)}%`);
        });
        worldScene.add(flyRig.group);
        flyRig.group.position.copy(autopilot.flyPos);
        setStatus('Fly body model loaded.');
      } catch (err) {
        flyRig = null;
        setStatus(`Fly body model failed to load: ${err.message}. No placeholder fly is shown.`, true);
      }
    }

    // Neuron point cloud (real MaleCNS sample; explicit failure, no invented data)
    async function initNeurons() {
      try {
        const data = await loadNeurons();
        brainPoints = buildBrainPoints(data);
        brainScene.add(brainPoints);
        if (dom.sourceLabel) {
          dom.sourceLabel.textContent = `${data.dataset} — ${data.points.length} of ${data.total} neurons — ${data.source}`;
        }
      } catch (err) {
        brainPoints = null;
        if (dom.sourceLabel) dom.sourceLabel.textContent = 'Neuron data unavailable';
        setStatus(`Failed to load real MaleCNS neuron data: ${err.message}. No connectome data was invented.`, true);
      }
    }

    await Promise.all([initFly(), initNeurons()]);

    autopilot.chapterIndex = -1;
    advanceChapter();

    // Camera follows the current chapter's vantage point, chasing the fly.
    function updateCamera(dt) {
      const info = CHAPTER_INFO[autopilot.currentChapter] || CHAPTER_INFO.street;
      worldCamera.position.lerp(info.camera, Math.min(1, dt * 1.5));
      const lookTarget = autopilot.flyPos.clone().lerp(info.target, 0.4);
      worldCamera.lookAt(lookTarget);
    }

    // rAF loop, with document-hidden pause/resume and request throttling --
    let rafId = null;
    let lastFrameTime = performance.now();
    let paused = false;

    function animate(nowMs) {
      rafId = requestAnimationFrame(animate);
      const dt = Math.min(0.1, (nowMs - lastFrameTime) / 1000);
      lastFrameTime = nowMs;
      const t = nowMs / 1000;

      if (nowMs >= autopilot.nextChangeAt) advanceChapter();

      updateFlyAutopilot(dt, t, flyRig);
      updateMotorTelemetry();
      updateCamera(dt);
      animateProps(t, dt, skateboard, car);

      if (worldRenderer) {
        fitRendererToCanvas(worldRenderer, worldCamera, dom.worldCanvas);
        drawPhoneScreen(phoneProp, autopilot.currentChapter, autopilot.currentNpcKey, t, autopilot.swipeStart);
        worldRenderer.render(worldScene, worldCamera);
      }

      if (brainRenderer) {
        fitRendererToCanvas(brainRenderer, brainCamera, dom.brainCanvas);
        brainCamera.position.x = Math.sin(t * 0.08) * 2.6;
        brainCamera.position.z = Math.cos(t * 0.08) * 2.6;
        brainCamera.lookAt(0, 0, 0);
        if (brainPoints) {
          brainPoints.material.uniforms.uTime.value = t;
          brainPoints.material.uniforms.uMood.value = moodToScalar(autopilot.mood);
        }
        brainRenderer.render(brainScene, brainCamera);
      }

      const freq = drawOscilloscope(scopeCtx, dom.scopeCanvas, nowMs, t);
      if (dom.telemetryRate) dom.telemetryRate.textContent = `${freq.toFixed(1)} Hz (modeled)`;
      drawRaster(rasterCtx, dom.rasterCanvas, dt);
      drawEqualizer(equalizerCtx, dom.equalizerCanvas);
    }

    function pauseAnimation() {
      if (paused) return;
      paused = true;
      if (rafId) {
        cancelAnimationFrame(rafId);
        rafId = null;
      }
      suspendAudio();
      setStatus('Paused (tab hidden).');
    }

    function resumeAnimation() {
      if (!paused) return;
      paused = false;
      lastFrameTime = performance.now();
      // Avoid an immediate burst of chapter/chat activity right after resume.
      autopilot.nextChangeAt = performance.now() + 1500;
      if (audioEnabled) ensureAudioContext();
      rafId = requestAnimationFrame(animate);
    }

    document.addEventListener('visibilitychange', () => {
      if (document.hidden) pauseAnimation();
      else resumeAnimation();
    });

    lastFrameTime = performance.now();
    rafId = requestAnimationFrame(animate);
    setStatus('Simulation running.');
  }

  function reportFatal(err) {
    setStatus(`Flying Fly failed to start: ${err && err.message ? err.message : err}`, true);
    // eslint-disable-next-line no-console
    console.error('Flying Fly fatal error:', err);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
      main().catch(reportFatal);
    });
  } else {
    main().catch(reportFatal);
  }
})();
