/**
 * game.js
 * ---------------------------------------------------------------------------
 * Гневный комментатор — fully autonomous 3D fly simulation front end.
 *
 * This file is the ONLY frontend module this change touches (aside from the
 * coordinated markup/CSS in index.html/style.css). It coordinates with:
 *   - index.html, which declares an importmap for "three" (three@0.169.0) and
 *     the element ids: brain-canvas, world-canvas, scope-canvas,
 *     raster-canvas, equalizer, backend-label, brain-progress-fill,
 *     brain-progress-text, source-label, telemetry-rate, spikes-rate,
 *     status, sound-toggle, event-log, comment-log, subtitle,
 *     comments-count, forced-count, fatigue-fill, fatigue-value,
 *     anger-fill, anger-value, loading-screen, loading-text, error-banner.
 *   - fly_rig.js, imported as `import { loadFly } from './fly_rig.js'`.
 *     `await loadFly(THREE, onProgress)` resolves to a rig object shaped as
 *     { group, update(t, walkingStrength, mood, options), parts, setRagdoll,
 *     applyImpulse, isRagdolling, floorY }. This file measures the loaded
 *     rig's bounding box at load time (THREE.Box3) rather than assuming a
 *     fixed scale, and guards setRagdoll/applyImpulse with typeof checks so
 *     the game keeps running (with a visible warning in the event log) if a
 *     concurrently-evolving rig build temporarily omits either method.
 *   - neuro_sim.js, imported lazily via dynamic `import('./neuro_sim.js')`
 *     inside try/catch. `await createConnectomeSim({ displayIndices,
 *     rasterCount, onProgress })` resolves to a sim object exposing
 *     backend, neuronCount/edgeCount/synapseCount, dataset, groups,
 *     step(dt), getSummary(), stimulate(target, rateHz, durationMs),
 *     setDrive(0..2), dispose(). If the import or the async factory throws
 *     for any reason (missing file, WebGPU/CPU worker failure, network
 *     error), this file never fabricates neural activity: the brain panel
 *     shows "симуляция недоступна", the oscilloscope/raster render flat
 *     "нет данных" traces, and every stimulate()/setDrive() call becomes a
 *     no-op guarded by `if (sim) ...`.
 *   - GET /api/neurons, which returns REAL sampled MaleCNS soma coordinates
 *     as { dataset, total, source, points:[{x,y,z,index,type,side,nt,
 *     superclass}] }. If that request fails, this file shows an explicit
 *     error (status text, event log, error banner) and renders NO point
 *     cloud. It never fabricates or invents connectome data. point.index
 *     values are used as the sim's displayIndices so displaySpikes color
 *     the exact same real points (cyan rest -> gold spike).
 *
 * Game design (see project brief "Гневный комментатор"):
 *   The fly stands at a laptop and autonomously types angry-but-harmless
 *   Russian comments about mundane fictional annoyances (crumbs, weather,
 *   lamp light, short videos, wifi, missing sugar, ...). There are NO manual
 *   controls; the viewer only watches. Typing raises fatigue; when fatigue
 *   saturates the fly collapses into a ragdoll heap. A procedurally built
 *   beetle overseer ("жук-надзиратель", primitive geometry only — the
 *   fly itself is always the real STL rig) walks in, pokes the fly with an
 *   impulse, and the loop resumes forever. All Gemini/API chat, the TikTok
 *   phone scene, the street/NPC scene, the skateboard, the compact car, the
 *   podium, and the chapter system from the previous version are removed.
 *   There is no /api/chat call anywhere in this file and no speech synthesis
 *   of any AI-generated reply.
 *   The 28-bar equalizer is real DOM elements (.eq-bar inside #equalizer)
 *   driven by a real WebAudio AnalyserNode listening to a synthesized,
 *   original keyboard-click + ambient pad loop (oscillators/noise buffers
 *   only, no samples, no copyrighted audio). Audio only starts after the
 *   user clicks the sound toggle (autoplay policy).
 *   The oscilloscope, spike raster, point-cloud colors, spikes/second
 *   counter, and "anger" reading are all computed from neuro_sim.js's real
 *   step()/getSummary() output when the sim loaded successfully; they are
 *   explicitly labeled as model output computed on the real MaleCNS
 *   connectome, never as recorded electrophysiology.
 * ---------------------------------------------------------------------------
 */
import * as THREE from 'three';
import { loadFly } from './fly_rig.js';

(() => {
  'use strict';

  // -------------------------------------------------------------------------
  // Config
  // -------------------------------------------------------------------------
  const ENDPOINTS = {
    neurons: '/api/neurons',
  };

  const MAX_EVENT_LOG_ENTRIES = 30;
  const MAX_COMMENT_LOG_ENTRIES = 40;
  const MAX_PIXEL_RATIO = 2;
  const EQUALIZER_BARS = 28;
  const RASTER_COUNT = 48;
  const OSCILLOSCOPE_HISTORY = 180;

  const FLY_TARGET_LENGTH = 1.0; // desired longest bounding-box dimension, world units

  const BASE_TYPING_CHARS_PER_SEC = 3.4;
  const FATIGUE_PER_KEYSTROKE = 0.011;
  const FATIGUE_PER_SECOND_TYPING = 0.006;
  const FATIGUE_RECOVERY_PER_SECOND = 0.0022;
  const BEETLE_ENTRY_DELAY_MS = 2000;
  const RAGDOLL_RECOVER_DURATION_MS = 1100;
  const RESUME_FATIGUE = 0.35;
  const COMMENT_GAP_MS = [350, 950];
  const MOTOR_BASELINE_SMOOTHING = 0.02; // exponential moving average weight per second

  const COMMENT_NICKNAME = 'Муха_3000';

  const COMMENT_SUBJECTS = [
    'крошки на столе', 'погода сегодня', 'этот свет лампы', 'короткие видео',
    'вай-фай', 'этот стул', 'холодный чай', 'скрипящая дверь', 'слишком тихая музыка',
    'этот шрифт', 'бесконечная реклама', 'мигающая лампочка', 'этот сквозняк',
    'слишком быстрый таймер', 'эти уведомления', 'скрип кресла на столе', 'отсутствие ответа от кого-то',
  ];

  const COMMENT_TEMPLATES = [
    (s) => `${s} — это ПОЗОР!!!`,
    (s) => `Кто вообще решил, что так может быть: ${s}?! возмутительно!`,
    (s) => `Опять ${s}?! сколько можно!`,
    (s) => `${s} — это довело меня до предела!!!`,
    (s) => `Нет, ну серьёзно, ${s} — это уже слишком!`,
    (s) => `где справедливость?! ${s} каждый день!`,
    (s) => `мухи требуют объяснений по поводу ${s}!!!`,
    (s) => `${s}... я больше не могу!!!`,
    (s) => `каждый раз одно и то же: ${s}. ПОЗОР!`,
  ];

  const COMMENT_FIXED = [
    'где МОЙ САХАР?! кто его взял?!',
    'почему видео такие короткие?! я даже не успела разозлиться!!!',
    'почему вай-фай тормозит именно когда я пишу?! заговор!',
    'лампа снова мигает!!! это невыносимо!',
    'кто-то съел мои крошки?! буду жаловаться наверху!',
  ];

  const BEETLE_LINES = [
    'А ну пиши дальше! Комментарии сами себя не напишут!',
    'Хватит валяться! Вставать и работать!',
    'Перерыв окончен, за работу!',
    'Кто разрешил отдыхать?! Пиши!',
    'Вставай, муха! Начальство не ждёт!',
  ];

  // -------------------------------------------------------------------------
  // DOM references (coordinated ids; every lookup is null-safe)
  // -------------------------------------------------------------------------
  const dom = {
    brainCanvas: document.getElementById('brain-canvas'),
    worldCanvas: document.getElementById('world-canvas'),
    scopeCanvas: document.getElementById('scope-canvas'),
    rasterCanvas: document.getElementById('raster-canvas'),
    equalizer: document.getElementById('equalizer'),
    backendLabel: document.getElementById('backend-label'),
    brainProgressFill: document.getElementById('brain-progress-fill'),
    brainProgressText: document.getElementById('brain-progress-text'),
    sourceLabel: document.getElementById('source-label'),
    telemetryRate: document.getElementById('telemetry-rate'),
    spikesRate: document.getElementById('spikes-rate'),
    status: document.getElementById('status'),
    soundToggle: document.getElementById('sound-toggle'),
    eventLog: document.getElementById('event-log'),
    commentLog: document.getElementById('comment-log'),
    subtitle: document.getElementById('subtitle'),
    commentsCount: document.getElementById('comments-count'),
    forcedCount: document.getElementById('forced-count'),
    fatigueFill: document.getElementById('fatigue-fill'),
    fatigueValue: document.getElementById('fatigue-value'),
    angerFill: document.getElementById('anger-fill'),
    angerValue: document.getElementById('anger-value'),
    loadingScreen: document.getElementById('loading-screen'),
    loadingText: document.getElementById('loading-text'),
    errorBanner: document.getElementById('error-banner'),
  };

  // -------------------------------------------------------------------------
  // Status text, event log, error banner, loading screen
  // -------------------------------------------------------------------------
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

  function setLoadingText(text) {
    if (dom.loadingText) dom.loadingText.textContent = text;
  }

  function appendCommentToSideLog(nickname, text, likes) {
    if (!dom.commentLog) return;
    const li = document.createElement('li');
    const nick = document.createElement('span');
    nick.className = 'comment-nick';
    nick.textContent = nickname;
    const body = document.createElement('span');
    body.className = 'comment-body';
    body.textContent = text;
    const likesEl = document.createElement('span');
    likesEl.className = 'comment-likes';
    likesEl.textContent = `♥ ${likes}`;
    li.appendChild(nick);
    li.appendChild(body);
    li.appendChild(document.createTextNode(' '));
    li.appendChild(likesEl);
    dom.commentLog.appendChild(li);
    while (dom.commentLog.children.length > MAX_COMMENT_LOG_ENTRIES) {
      dom.commentLog.removeChild(dom.commentLog.firstChild);
    }
    dom.commentLog.scrollTop = dom.commentLog.scrollHeight;
  }

  // -------------------------------------------------------------------------
  // Renderer helpers (two THREE.WebGLRenderer instances, responsive resize)
  // -------------------------------------------------------------------------
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
    const width = Math.max(1, canvas.clientWidth || parseInt(canvas.getAttribute('width'), 10) || 320);
    const height = Math.max(1, canvas.clientHeight || parseInt(canvas.getAttribute('height'), 10) || 110);
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

  // -------------------------------------------------------------------------
  // Equalizer: #equalizer is a <div role="img">, not a canvas. Build
  // EQUALIZER_BARS real .eq-bar <div> children once and drive their height
  // from a real WebAudio AnalyserNode each frame.
  // -------------------------------------------------------------------------
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
      const avg = sum / bucketSize;
      const pct = Math.max(8, Math.min(100, (avg / 255) * 100));
      equalizerBarEls[i].style.height = `${pct.toFixed(1)}%`;
    }
  }

  function decayEqualizerIdle() {
    equalizerBarEls.forEach((bar) => {
      const current = parseFloat(bar.style.height) || 8;
      bar.style.height = `${Math.max(8, current * 0.9).toFixed(1)}%`;
    });
  }

  // -------------------------------------------------------------------------
  // WebAudio: synthesized keyboard clicks + ambient pad, gated by user gesture
  // -------------------------------------------------------------------------
  const audio = {
    ctx: null,
    master: null,
    analyser: null,
    freqData: null,
    padNodes: null,
    enabled: false,
  };

  function ensureAudioContext() {
    if (audio.ctx) return audio.ctx;
    const Ctor = window.AudioContext || window.webkitAudioContext;
    if (!Ctor) return null;
    const ctx = new Ctor();
    const master = ctx.createGain();
    master.gain.value = 0.55;
    const analyser = ctx.createAnalyser();
    analyser.fftSize = 256;
    analyser.smoothingTimeConstant = 0.75;
    master.connect(analyser);
    analyser.connect(ctx.destination);
    audio.ctx = ctx;
    audio.master = master;
    audio.analyser = analyser;
    audio.freqData = new Uint8Array(analyser.frequencyBinCount);
    return ctx;
  }

  function startAmbientPad() {
    if (!audio.ctx || audio.padNodes) return;
    const ctx = audio.ctx;
    const padGain = ctx.createGain();
    padGain.gain.value = 0.05;
    padGain.connect(audio.master);

    const osc1 = ctx.createOscillator();
    osc1.type = 'sine';
    osc1.frequency.value = 82.4;
    const osc2 = ctx.createOscillator();
    osc2.type = 'sine';
    osc2.frequency.value = 82.4 * 1.005;

    const filter = ctx.createBiquadFilter();
    filter.type = 'lowpass';
    filter.frequency.value = 400;

    const lfo = ctx.createOscillator();
    lfo.type = 'sine';
    lfo.frequency.value = 0.08;
    const lfoGain = ctx.createGain();
    lfoGain.gain.value = 120;
    lfo.connect(lfoGain);
    lfoGain.connect(filter.frequency);

    osc1.connect(filter);
    osc2.connect(filter);
    filter.connect(padGain);

    osc1.start();
    osc2.start();
    lfo.start();

    audio.padNodes = { osc1, osc2, lfo, padGain, filter };
  }

  function playKeyClick() {
    if (!audio.enabled || !audio.ctx) return;
    const ctx = audio.ctx;
    const now = ctx.currentTime;

    const bufferSize = Math.floor(ctx.sampleRate * 0.02);
    const buffer = ctx.createBuffer(1, bufferSize, ctx.sampleRate);
    const data = buffer.getChannelData(0);
    for (let i = 0; i < bufferSize; i++) {
      data[i] = (Math.random() * 2 - 1) * (1 - i / bufferSize);
    }
    const noise = ctx.createBufferSource();
    noise.buffer = buffer;
    const noiseFilter = ctx.createBiquadFilter();
    noiseFilter.type = 'highpass';
    noiseFilter.frequency.value = 1800 + Math.random() * 800;
    const noiseGain = ctx.createGain();
    noiseGain.gain.setValueAtTime(0.22, now);
    noiseGain.gain.exponentialRampToValueAtTime(0.001, now + 0.05);
    noise.connect(noiseFilter);
    noiseFilter.connect(noiseGain);
    noiseGain.connect(audio.master);
    noise.start(now);
    noise.stop(now + 0.06);

    const tick = ctx.createOscillator();
    tick.type = 'square';
    tick.frequency.value = 1200 + Math.random() * 400;
    const tickGain = ctx.createGain();
    tickGain.gain.setValueAtTime(0.05, now);
    tickGain.gain.exponentialRampToValueAtTime(0.001, now + 0.03);
    tick.connect(tickGain);
    tickGain.connect(audio.master);
    tick.start(now);
    tick.stop(now + 0.03);
  }

  function playStartleSting() {
    if (!audio.enabled || !audio.ctx) return;
    const ctx = audio.ctx;
    const now = ctx.currentTime;
    const osc = ctx.createOscillator();
    osc.type = 'sawtooth';
    osc.frequency.setValueAtTime(220, now);
    osc.frequency.exponentialRampToValueAtTime(60, now + 0.3);
    const gain = ctx.createGain();
    gain.gain.setValueAtTime(0.18, now);
    gain.gain.exponentialRampToValueAtTime(0.001, now + 0.35);
    osc.connect(gain);
    gain.connect(audio.master);
    osc.start(now);
    osc.stop(now + 0.36);
  }

  function setupSoundToggle() {
    if (!dom.soundToggle) return;
    dom.soundToggle.addEventListener('click', () => {
      const ctx = ensureAudioContext();
      if (!ctx) {
        showErrorBanner('Аудио недоступно в этом браузере.', false);
        return;
      }
      if (ctx.state === 'suspended') ctx.resume();
      audio.enabled = !audio.enabled;
      dom.soundToggle.setAttribute('aria-pressed', String(audio.enabled));
      const label = dom.soundToggle.querySelector('.sound-label');
      if (label) label.textContent = audio.enabled ? 'Выключить звук' : 'Включить звук';
      if (audio.enabled) {
        startAmbientPad();
        logEvent('Звук включён.');
      } else {
        logEvent('Звук выключен.');
      }
    });
  }

  // -------------------------------------------------------------------------
  // Comment generator (procedural, Russian, harmless/mundane topics only)
  // -------------------------------------------------------------------------
  let lastSubjectIndex = -1;

  function generateComment() {
    if (Math.random() < 0.22) {
      return COMMENT_FIXED[Math.floor(Math.random() * COMMENT_FIXED.length)];
    }
    let subjIdx = Math.floor(Math.random() * COMMENT_SUBJECTS.length);
    if (subjIdx === lastSubjectIndex) {
      subjIdx = (subjIdx + 1) % COMMENT_SUBJECTS.length;
    }
    lastSubjectIndex = subjIdx;
    const subject = COMMENT_SUBJECTS[subjIdx];
    const template = COMMENT_TEMPLATES[Math.floor(Math.random() * COMMENT_TEMPLATES.length)];
    return template(subject);
  }

  let lastBeetleLineIndex = -1;
  function pickBeetleLine() {
    let idx = Math.floor(Math.random() * BEETLE_LINES.length);
    if (idx === lastBeetleLineIndex) idx = (idx + 1) % BEETLE_LINES.length;
    lastBeetleLineIndex = idx;
    return BEETLE_LINES[idx];
  }

  // -------------------------------------------------------------------------
  // Small canvas drawing helpers (original procedural graphics only)
  // -------------------------------------------------------------------------
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
    if (!text) return y;
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
    return cursorY + lineHeight;
  }

  // =========================================================================
  // BRAIN PANEL (left): real MaleCNS point cloud, oscilloscope, raster,
  // connectome download progress, spikes/s counter, backend label.
  // =========================================================================
  const brain = {
    scene: null,
    camera: null,
    renderer: null,
    points: null,
    colorAttr: null,
    glow: null, // Float32Array per-point decaying glow used for cyan->gold blend
    indexOrder: null, // point.index values, in the same order as the geometry
    rotationSpeed: 0.06,
    oscHistory: new Float32Array(OSCILLOSCOPE_HISTORY),
    oscHistoryFilled: 0,
  };

  async function fetchNeurons() {
    setStatus('Загрузка нейронов MaleCNS…', 'loading');
    let response;
    try {
      response = await fetch(ENDPOINTS.neurons);
    } catch (err) {
      throw new Error(`сетевая ошибка: ${err && err.message ? err.message : err}`);
    }
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    let payload;
    try {
      payload = await response.json();
    } catch (err) {
      throw new Error('ответ /api/neurons не является корректным JSON');
    }
    if (!payload || !Array.isArray(payload.points) || payload.points.length === 0) {
      throw new Error('/api/neurons вернул пустой набор точек');
    }
    return payload;
  }

  function buildBrainPointCloud(payload) {
    const points = payload.points;
    const count = points.length;
    const positions = new Float32Array(count * 3);
    const colors = new Float32Array(count * 3);
    const glow = new Float32Array(count);
    const indexOrder = new Array(count);

    let minX = Infinity, maxX = -Infinity;
    let minY = Infinity, maxY = -Infinity;
    let minZ = Infinity, maxZ = -Infinity;
    for (let i = 0; i < count; i++) {
      const p = points[i];
      minX = Math.min(minX, p.x); maxX = Math.max(maxX, p.x);
      minY = Math.min(minY, p.y); maxY = Math.max(maxY, p.y);
      minZ = Math.min(minZ, p.z); maxZ = Math.max(maxZ, p.z);
    }
    const spanX = Math.max(1e-6, maxX - minX);
    const spanY = Math.max(1e-6, maxY - minY);
    const spanZ = Math.max(1e-6, maxZ - minZ);
    const span = Math.max(spanX, spanY, spanZ);
    const cx = (minX + maxX) / 2;
    const cy = (minY + maxY) / 2;
    const cz = (minZ + maxZ) / 2;

    for (let i = 0; i < count; i++) {
      const p = points[i];
      positions[i * 3] = ((p.x - cx) / span) * 2.2;
      positions[i * 3 + 1] = ((p.y - cy) / span) * 2.2;
      positions[i * 3 + 2] = ((p.z - cz) / span) * 2.2;
      colors[i * 3] = 0.0;
      colors[i * 3 + 1] = 0.85;
      colors[i * 3 + 2] = 1.0;
      indexOrder[i] = typeof p.index === 'number' ? p.index : i;
    }

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    const colorAttr = new THREE.BufferAttribute(colors, 3);
    geometry.setAttribute('color', colorAttr);

    const material = new THREE.PointsMaterial({
      size: 0.028,
      vertexColors: true,
      transparent: true,
      opacity: 0.9,
      sizeAttenuation: true,
    });

    const pointCloud = new THREE.Points(geometry, material);
    brain.points = pointCloud;
    brain.colorAttr = colorAttr;
    brain.glow = glow;
    brain.indexOrder = indexOrder;
    brain.scene.add(pointCloud);

    if (dom.sourceLabel) {
      dom.sourceLabel.textContent = `${payload.source || payload.dataset || 'MaleCNS'} · ${payload.total || count} нейронов (выборка ${count})`;
    }
  }

  function createBrainScene() {
    if (!dom.brainCanvas) return;
    brain.scene = new THREE.Scene();
    brain.camera = new THREE.PerspectiveCamera(50, 4 / 3, 0.05, 20);
    brain.camera.position.set(0, 0.4, 3.4);
    brain.camera.lookAt(0, 0, 0);
    brain.renderer = makeRenderer(dom.brainCanvas);
  }

  function updateBrainColors(displaySpikes, dt) {
    if (!brain.colorAttr || !brain.glow) return;
    const glow = brain.glow;
    const colors = brain.colorAttr.array;
    const decay = Math.pow(0.02, dt); // fast decay so spikes read as flashes
    const n = glow.length;
    for (let i = 0; i < n; i++) {
      if (displaySpikes && displaySpikes[i]) glow[i] = 1;
      else glow[i] *= decay;
      const g = glow[i];
      colors[i * 3] = g * 1.0; // red channel ramps toward gold
      colors[i * 3 + 1] = 0.85 - g * 0.13; // green stays high (cyan+gold both high-G)
      colors[i * 3 + 2] = 1.0 - g * 0.99; // blue drops out as it goes gold
    }
    brain.colorAttr.needsUpdate = true;
  }

  function renderBrainScene(t) {
    if (!brain.renderer || !brain.scene || !brain.camera) return;
    fitRendererToCanvas(brain.renderer, brain.camera, dom.brainCanvas);
    if (brain.points) brain.points.rotation.y = t * brain.rotationSpeed;
    brain.renderer.render(brain.scene, brain.camera);
  }

  function updateBrainProgress(fraction, message) {
    const pct = Math.max(0, Math.min(100, Math.round((fraction || 0) * 100)));
    if (dom.brainProgressFill) dom.brainProgressFill.style.width = `${pct}%`;
    if (dom.brainProgressText) {
      dom.brainProgressText.textContent = message ? `${message} (${pct}%)` : `${pct}%`;
    }
  }

  function drawOscilloscope(ctx, width, height, value, hasSim) {
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = '#04060a';
    ctx.fillRect(0, 0, width, height);

    const hist = brain.oscHistory;
    if (hasSim) {
      hist.copyWithin(0, 1);
      hist[hist.length - 1] = value;
      brain.oscHistoryFilled = Math.min(hist.length, brain.oscHistoryFilled + 1);
    }

    ctx.strokeStyle = hasSim ? '#00f3ff' : 'rgba(139, 163, 191, 0.4)';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    const usable = hasSim ? brain.oscHistoryFilled : hist.length;
    const maxVal = hasSim ? Math.max(1, ...Array.from(hist)) : 1;
    for (let i = 0; i < hist.length; i++) {
      const x = (i / (hist.length - 1)) * width;
      const v = hasSim ? hist[i] / maxVal : 0.5;
      const y = height - v * (height - 8) - 4;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    ctx.fillStyle = hasSim ? '#8fa3bf' : '#5b6b82';
    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.fillText(hasSim ? 'PAM, рассчитано моделью' : 'нет данных', 6, 12);
  }

  let rasterOffscreen = null;
  function drawRaster(ctx, width, height, rasterBits, hasSim) {
    if (!rasterOffscreen || rasterOffscreen.width !== width || rasterOffscreen.height !== height) {
      rasterOffscreen = document.createElement('canvas');
      rasterOffscreen.width = width;
      rasterOffscreen.height = height;
      const octx = rasterOffscreen.getContext('2d');
      octx.fillStyle = '#04060a';
      octx.fillRect(0, 0, width, height);
    }
    const octx = rasterOffscreen.getContext('2d');
    // Scroll the persistent buffer left by 2px.
    octx.drawImage(rasterOffscreen, -2, 0);
    octx.fillStyle = '#04060a';
    octx.fillRect(width - 2, 0, 2, height);

    if (hasSim && rasterBits && rasterBits.length) {
      const rowH = height / rasterBits.length;
      for (let r = 0; r < rasterBits.length; r++) {
        if (rasterBits[r]) {
          octx.fillStyle = '#ffb703';
          octx.fillRect(width - 2, r * rowH, 2, Math.max(1, rowH - 0.5));
        }
      }
    }

    ctx.clearRect(0, 0, width, height);
    ctx.drawImage(rasterOffscreen, 0, 0);
    ctx.fillStyle = hasSim ? '#8fa3bf' : '#5b6b82';
    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.fillText(hasSim ? '48 нейронов растра' : 'нет данных', 6, height - 6);
  }

  function formatSpikesPerSecond(n) {
    if (!Number.isFinite(n)) return '— спайков/с';
    if (n >= 1e6) return `${(n / 1e6).toFixed(2)}M спайков/с`;
    if (n >= 1e3) return `${(n / 1e3).toFixed(2)}K спайков/с`;
    return `${Math.round(n)} спайков/с`;
  }

  // =========================================================================
  // NEURO SIM (neuro_sim.js), loaded dynamically, never fabricated on failure
  // =========================================================================
  let neuroSim = null;
  let neuroSimFailed = false;

  async function loadNeuroSim(displayIndices) {
    try {
      logEvent('Загрузка модели коннектома (~80 Мб)…');
      updateBrainProgress(0, 'Скачивание коннектома');
      const mod = await import('./neuro_sim.js');
      const sim = await mod.createConnectomeSim({
        displayIndices,
        rasterCount: RASTER_COUNT,
        onProgress: (fraction, message) => updateBrainProgress(fraction, message),
      });
      updateBrainProgress(1, 'Готово');
      if (dom.backendLabel) {
        dom.backendLabel.textContent = `Бэкенд: ${sim.backend === 'webgpu' ? 'WebGPU' : 'CPU'}`;
      }
      logEvent(`Симуляция коннектома загружена: ${sim.neuronCount} нейронов, бэкенд ${sim.backend}.`);
      return sim;
    } catch (err) {
      neuroSimFailed = true;
      const msg = err && err.message ? err.message : String(err);
      logEvent(`Симуляция коннектома недоступна: ${msg}`);
      showErrorBanner('Симуляция коннектома недоступна: активность мозга не отображается.', false);
      if (dom.backendLabel) dom.backendLabel.textContent = 'Бэкенд: недоступен';
      if (dom.brainProgressText) dom.brainProgressText.textContent = 'симуляция недоступна';
      if (dom.brainProgressFill) dom.brainProgressFill.style.width = '0%';
      return null;
    }
  }

  // =========================================================================
  // WORLD (right): room + desk + laptop + fly rig + beetle overseer
  // =========================================================================
  const world = {
    scene: null,
    camera: null,
    renderer: null,
    clock: new THREE.Clock(),
    deskTopY: 0.78,
    laptopPos: new THREE.Vector3(0, 0, 0),
  };

  const fly = {
    rig: null,
    group: null,
    scale: 1,
    warnedMissingRagdoll: false,
    warnedMissingImpulse: false,
  };

  const beetle = {
    group: null,
    legs: [],
    homeX: -2.6,
    workX: 0.55,
    state: 'offstage', // offstage | entering | poking | leaving
    progress: 0,
  };

  function createWorldScene() {
    if (!dom.worldCanvas) return;
    world.scene = new THREE.Scene();
    world.scene.background = new THREE.Color(0x05070c);
    world.scene.fog = new THREE.Fog(0x05070c, 3, 9);

    world.camera = new THREE.PerspectiveCamera(45, 4 / 3, 0.05, 30);
    world.camera.position.set(0.9, 1.35, 1.9);
    world.camera.lookAt(0, 0.75, 0);

    world.renderer = makeRenderer(dom.worldCanvas);

    const ambient = new THREE.AmbientLight(0x293040, 0.7);
    world.scene.add(ambient);

    const lamp = new THREE.PointLight(0xffd9a0, 1.4, 4.5, 2);
    lamp.position.set(-0.35, 1.25, -0.2);
    world.scene.add(lamp);

    const rim = new THREE.DirectionalLight(0x3a5a8f, 0.35);
    rim.position.set(2, 3, 1);
    world.scene.add(rim);

    // Floor
    const floor = new THREE.Mesh(
      new THREE.PlaneGeometry(8, 8),
      new THREE.MeshStandardMaterial({ color: 0x11151d, roughness: 0.95, metalness: 0.02 }),
    );
    floor.rotation.x = -Math.PI / 2;
    world.scene.add(floor);

    // Back wall
    const wall = new THREE.Mesh(
      new THREE.PlaneGeometry(8, 4),
      new THREE.MeshStandardMaterial({ color: 0x0c0f16, roughness: 1 }),
    );
    wall.position.set(0, 2, -1.4);
    world.scene.add(wall);

    // Desk
    const deskMat = new THREE.MeshStandardMaterial({ color: 0x3a2b1e, roughness: 0.7, metalness: 0.05 });
    const deskTop = new THREE.Mesh(new THREE.BoxGeometry(1.7, 0.05, 0.8), deskMat);
    deskTop.position.set(0, world.deskTopY, 0);
    world.scene.add(deskTop);
    const legGeo = new THREE.BoxGeometry(0.06, world.deskTopY - 0.025, 0.06);
    const legPositions = [
      [-0.8, -0.35], [0.8, -0.35], [-0.8, 0.35], [0.8, 0.35],
    ];
    legPositions.forEach(([lx, lz]) => {
      const leg = new THREE.Mesh(legGeo, deskMat);
      leg.position.set(lx, (world.deskTopY - 0.025) / 2, lz);
      world.scene.add(leg);
    });

    // Desk lamp
    const lampGroup = new THREE.Group();
    const lampBase = new THREE.Mesh(new THREE.CylinderGeometry(0.07, 0.09, 0.03, 16),
      new THREE.MeshStandardMaterial({ color: 0x1c1c22, metalness: 0.6, roughness: 0.3 }));
    const lampArm = new THREE.Mesh(new THREE.CylinderGeometry(0.012, 0.012, 0.5, 8),
      new THREE.MeshStandardMaterial({ color: 0x2a2a30, metalness: 0.7, roughness: 0.25 }));
    lampArm.position.set(0, 0.25, 0);
    lampArm.rotation.z = 0.25;
    const lampHead = new THREE.Mesh(new THREE.ConeGeometry(0.09, 0.14, 16, 1, true),
      new THREE.MeshStandardMaterial({ color: 0xffe9b8, emissive: 0xffb703, emissiveIntensity: 0.6, side: THREE.DoubleSide }));
    lampHead.position.set(0.12, 0.48, 0);
    lampHead.rotation.x = Math.PI;
    lampGroup.add(lampBase, lampArm, lampHead);
    lampGroup.position.set(-0.62, world.deskTopY + 0.025, -0.22);
    world.scene.add(lampGroup);

    // Laptop base + keyboard
    const laptopMat = new THREE.MeshStandardMaterial({ color: 0x1c1e24, metalness: 0.5, roughness: 0.4 });
    const laptopBase = new THREE.Mesh(new THREE.BoxGeometry(0.42, 0.02, 0.3), laptopMat);
    world.laptopPos.set(0.05, world.deskTopY + 0.035, 0.05);
    laptopBase.position.copy(world.laptopPos);
    world.scene.add(laptopBase);

    const keyboardMat = new THREE.MeshStandardMaterial({ color: 0x14161c, roughness: 0.8 });
    const keyboard = new THREE.Mesh(new THREE.BoxGeometry(0.36, 0.01, 0.18), keyboardMat);
    keyboard.position.set(world.laptopPos.x, world.laptopPos.y + 0.012, world.laptopPos.z + 0.05);
    world.scene.add(keyboard);

    // Laptop screen with CanvasTexture comment feed
    const feedCanvas = document.createElement('canvas');
    feedCanvas.width = 512;
    feedCanvas.height = 384;
    const feedCtx = feedCanvas.getContext('2d');
    const feedTexture = new THREE.CanvasTexture(feedCanvas);
    feedTexture.colorSpace = THREE.SRGBColorSpace;

    const screenGroup = new THREE.Group();
    const screenBack = new THREE.Mesh(new THREE.BoxGeometry(0.42, 0.28, 0.015), laptopMat);
    screenBack.position.set(0, 0.14, -0.008);
    const screenFace = new THREE.Mesh(
      new THREE.PlaneGeometry(0.38, 0.24),
      new THREE.MeshBasicMaterial({ map: feedTexture }),
    );
    screenFace.position.set(0, 0.14, 0.001);
    screenGroup.add(screenBack, screenFace);
    screenGroup.position.set(world.laptopPos.x, world.laptopPos.y + 0.01, world.laptopPos.z - 0.14);
    screenGroup.rotation.x = -0.28;
    world.scene.add(screenGroup);

    world.feedCanvas = feedCanvas;
    world.feedCtx = feedCtx;
    world.feedTexture = feedTexture;
  }

  function drawCommentFeedTexture(feedList, typingText, cursorOn) {
    const ctx = world.feedCtx;
    if (!ctx) return;
    const w = world.feedCanvas.width;
    const h = world.feedCanvas.height;

    ctx.fillStyle = '#0a0d14';
    ctx.fillRect(0, 0, w, h);

    ctx.fillStyle = '#00f3ff';
    ctx.font = 'bold 20px "JetBrains Mono", monospace';
    ctx.fillText('Комментарии', 16, 30);
    ctx.strokeStyle = 'rgba(0,243,255,0.35)';
    ctx.beginPath();
    ctx.moveTo(16, 42);
    ctx.lineTo(w - 16, 42);
    ctx.stroke();

    let y = 66;
    const visible = feedList.slice(-4);
    visible.forEach((c) => {
      ctx.fillStyle = '#ffb703';
      ctx.font = 'bold 15px "JetBrains Mono", monospace';
      ctx.fillText(c.nickname, 16, y);
      ctx.fillStyle = '#8fa3bf';
      ctx.font = '12px "JetBrains Mono", monospace';
      ctx.fillText(`♥ ${c.likes}`, w - 70, y);
      y += 20;
      ctx.fillStyle = '#e6f1ff';
      ctx.font = '14px Inter, sans-serif';
      y = wrapText(ctx, c.text, 16, y, w - 32, 18) + 10;
    });

    // Currently-typing comment area at the bottom
    ctx.strokeStyle = 'rgba(255,183,3,0.35)';
    ctx.beginPath();
    ctx.moveTo(16, h - 78);
    ctx.lineTo(w - 16, h - 78);
    ctx.stroke();
    ctx.fillStyle = '#ffb703';
    ctx.font = 'bold 15px "JetBrains Mono", monospace';
    ctx.fillText(COMMENT_NICKNAME, 16, h - 56);
    ctx.fillStyle = '#e6f1ff';
    ctx.font = '14px Inter, sans-serif';
    const cursor = cursorOn ? '|' : '';
    wrapText(ctx, `${typingText}${cursor}`, 16, h - 34, w - 32, 18);

    world.feedTexture.needsUpdate = true;
  }

  // -------------------------------------------------------------------------
  // Beetle overseer: procedural primitive geometry only (never the STL rig)
  // -------------------------------------------------------------------------
  function buildBeetle() {
    const group = new THREE.Group();
    const shellMat = new THREE.MeshStandardMaterial({ color: 0x1b2a1e, metalness: 0.65, roughness: 0.25 });
    const legMat = new THREE.MeshStandardMaterial({ color: 0x14201a, metalness: 0.4, roughness: 0.5 });
    const eyeMat = new THREE.MeshStandardMaterial({ color: 0x7a0e12, emissive: 0x3a0508, emissiveIntensity: 0.5 });

    const body = new THREE.Mesh(new THREE.SphereGeometry(0.16, 20, 16), shellMat);
    body.scale.set(1.5, 0.85, 1.0);
    body.position.y = 0.16;
    group.add(body);

    const head = new THREE.Mesh(new THREE.SphereGeometry(0.075, 16, 12), shellMat);
    head.position.set(0.24, 0.17, 0);
    group.add(head);

    const eyeL = new THREE.Mesh(new THREE.SphereGeometry(0.018, 8, 8), eyeMat);
    eyeL.position.set(0.29, 0.19, 0.045);
    const eyeR = eyeL.clone();
    eyeR.position.z = -0.045;
    group.add(eyeL, eyeR);

    const mandibleMat = new THREE.MeshStandardMaterial({ color: 0x0e0e12, metalness: 0.5, roughness: 0.4 });
    const mandibleL = new THREE.Mesh(new THREE.BoxGeometry(0.05, 0.012, 0.012), mandibleMat);
    mandibleL.position.set(0.32, 0.15, 0.02);
    const mandibleR = mandibleL.clone();
    mandibleR.position.z = -0.02;
    group.add(mandibleL, mandibleR);

    const antennaMat = new THREE.MeshStandardMaterial({ color: 0x1b2a1e, roughness: 0.6 });
    const antennaL = new THREE.Mesh(new THREE.CylinderGeometry(0.004, 0.004, 0.14, 6), antennaMat);
    antennaL.position.set(0.3, 0.24, 0.03);
    antennaL.rotation.z = -0.5;
    const antennaR = antennaL.clone();
    antennaR.position.z = -0.03;
    group.add(antennaL, antennaR);

    const legs = [];
    const legAnchorsX = [0.08, 0, -0.08];
    legAnchorsX.forEach((lx, i) => {
      [1, -1].forEach((side) => {
        const legGroup = new THREE.Group();
        const upper = new THREE.Mesh(new THREE.CylinderGeometry(0.012, 0.01, 0.13, 6), legMat);
        upper.rotation.z = side > 0 ? 1.1 : -1.1;
        upper.position.set(0, -0.05, side * 0.06);
        const lower = new THREE.Mesh(new THREE.CylinderGeometry(0.009, 0.006, 0.12, 6), legMat);
        lower.position.set(0, -0.13, side * 0.13);
        lower.rotation.z = side > 0 ? 0.5 : -0.5;
        legGroup.add(upper, lower);
        legGroup.position.set(lx, 0.16, side * 0.11);
        group.add(legGroup);
        legs.push({ group: legGroup, phase: i * 1.2 + (side > 0 ? 0 : Math.PI) });
      });
    });

    group.visible = false;
    group.position.set(beetle.homeX, 0, 0.55);
    group.rotation.y = -Math.PI / 2;
    beetle.group = group;
    beetle.legs = legs;
    world.scene.add(group);
  }

  function updateBeetleWalkAnimation(t, walking) {
    if (!beetle.group) return;
    beetle.legs.forEach((leg) => {
      const swing = walking ? Math.sin(t * 9 + leg.phase) * 0.35 : 0;
      leg.group.rotation.x = swing;
    });
    beetle.group.position.y = walking ? Math.abs(Math.sin(t * 9)) * 0.01 : 0;
  }

  // -------------------------------------------------------------------------
  // Fly rig loading, measured scale, ragdoll/impulse guards
  // -------------------------------------------------------------------------
  async function initFlyRig() {
    setStatus('Загрузка тела мухи…', 'loading');
    let rig;
    try {
      rig = await loadFly(THREE, (fraction, message) => {
        setLoadingText(`${message || 'Загрузка мухи'} (${Math.round((fraction || 0) * 100)}%)`);
      });
    } catch (err) {
      const msg = err && err.message ? err.message : String(err);
      setStatus(`Ошибка загрузки мухи: ${msg}`, 'error');
      logEvent(`Ошибка загрузки fly_rig.js: ${msg}`);
      showErrorBanner(`Не удалось загрузить тело мухи: ${msg}`, true);
      return false;
    }

    fly.rig = rig;
    fly.group = rig.group;

    // Measure the loaded rig's real bounding box instead of guessing scale.
    const box = new THREE.Box3().setFromObject(rig.group);
    const size = new THREE.Vector3();
    box.getSize(size);
    const longestDim = Math.max(size.x, size.y, size.z, 1e-6);
    fly.scale = FLY_TARGET_LENGTH / longestDim;
    rig.group.scale.setScalar(fly.scale);

    const floorYLocal = typeof rig.floorY === 'number' ? rig.floorY : 0;
    rig.group.position.set(
      world.laptopPos.x - 0.02,
      world.deskTopY + 0.035 - floorYLocal * fly.scale,
      world.laptopPos.z + 0.16,
    );
    // Facing rotation must NOT be written onto rig.group: fly_rig.js sets
    // that group's rotation.x = -PI/2 to convert its native +Z-up space
    // into three.js +Y-up. Writing .y there yaws about a pre-conversion
    // axis (Euler XYZ applies Ry inside Rx) and also corrupts
    // applyImpulse()'s group-quaternion inverse. Wrap the rig instead and
    // put position + facing on the wrapper.
    const flyRoot = new THREE.Group();
    flyRoot.name = 'fly-root';
    flyRoot.position.copy(rig.group.position);
    flyRoot.rotation.y = Math.PI;
    rig.group.position.set(0, 0, 0);
    flyRoot.add(rig.group);
    fly.root = flyRoot;

    // fly_rig.js documents floorY as world-space and defaults it to 0, so
    // its leg-tip floor contact never engaged for a fly standing on the
    // desk. Hand it the desk surface height.
    rig.floorY = world.deskTopY + 0.035;

    world.scene.add(flyRoot);

    if (typeof rig.setRagdoll !== 'function' && !fly.warnedMissingRagdoll) {
      fly.warnedMissingRagdoll = true;
      logEvent('Предупреждение: fly_rig.js пока не предоставляет setRagdoll(); коллапс будет только в позе.');
    }
    if (typeof rig.applyImpulse !== 'function' && !fly.warnedMissingImpulse) {
      fly.warnedMissingImpulse = true;
      logEvent('Предупреждение: fly_rig.js пока не предоставляет applyImpulse(); толчок жука будет только визуальный.');
    }

    logEvent('Физическая модель мухи загружена.');
    return true;
  }

  function safeSetRagdoll(weight) {
    if (fly.rig && typeof fly.rig.setRagdoll === 'function') {
      fly.rig.setRagdoll(weight);
    }
  }

  function safeApplyImpulse(direction, strength) {
    if (fly.rig && typeof fly.rig.applyImpulse === 'function') {
      // fly_rig.js only undoes its OWN group rotation, so bring the world
      // direction into the wrapper's frame before handing it over.
      let impulseDir = direction;
      if (fly.root && direction && typeof direction.clone === 'function') {
        impulseDir = direction.clone().applyQuaternion(
          fly.root.getWorldQuaternion(new THREE.Quaternion()).invert());
      }
      fly.rig.applyImpulse(impulseDir, strength);
    } else {
      logEvent('Симуляция толчка недоступна: applyImpulse() отсутствует в fly_rig.js.');
    }
  }

  function isFlyRagdolling() {
    if (fly.rig && typeof fly.rig.isRagdolling === 'function') {
      return !!fly.rig.isRagdolling();
    }
    return game.phase === 'collapsed' || game.phase === 'poking';
  }

  // =========================================================================
  // GAME STATE MACHINE
  // =========================================================================
  const game = {
    phase: 'typing', // typing | collapsed | beetle_waiting | beetle_entering | poking | beetle_leaving
    fatigue: 0,
    anger: 0,
    commentsWritten: 0,
    forcedCount: 0,
    feed: [],
    currentComment: '',
    typedChars: 0,
    typedCharsAccum: 0,
    nextCommentDelayMs: 0,
    collapseTimer: 0,
    ragdollRamp: 0,
    motorBaseline: null,
    motorFactor: 1,
    likeTickTimer: 0,
  };

  function startNewComment() {
    game.currentComment = generateComment();
    game.typedChars = 0;
    game.typedCharsAccum = 0;
  }

  function postCurrentComment() {
    const likes = Math.floor(Math.random() * 4);
    const entry = { nickname: COMMENT_NICKNAME, text: game.currentComment, likes };
    game.feed.push(entry);
    if (game.feed.length > 12) game.feed.shift();
    appendCommentToSideLog(entry.nickname, entry.text, entry.likes);
    game.commentsWritten += 1;
    if (dom.commentsCount) dom.commentsCount.textContent = String(game.commentsWritten);

    if (neuroSim) {
      neuroSim.stimulate('reward', 40, 400);
    }

    game.currentComment = '';
    game.typedChars = 0;
    game.nextCommentDelayMs = COMMENT_GAP_MS[0] + Math.random() * (COMMENT_GAP_MS[1] - COMMENT_GAP_MS[0]);
  }

  function updateTypingRate(dtSeconds) {
    if (!neuroSim) {
      game.motorFactor = 1;
      return;
    }
    const summary = neuroSim.getSummary();
    const motorRate = (summary.groupRates && summary.groupRates.motor) || 0;
    if (game.motorBaseline === null) {
      game.motorBaseline = Math.max(1, motorRate);
    } else if (game.phase === 'typing') {
      const w = Math.min(1, MOTOR_BASELINE_SMOOTHING * dtSeconds * 60);
      game.motorBaseline = game.motorBaseline * (1 - w) + motorRate * w;
    }
    const ratio = motorRate / Math.max(1, game.motorBaseline);
    game.motorFactor = Math.max(0.5, Math.min(2, ratio));

    const angerHz = ((summary.groupRates && summary.groupRates.dan) || 0) +
      ((summary.groupRates && summary.groupRates.pam) || 0);
    game.anger = Math.max(0, Math.min(100, (angerHz / 2 / 150) * 100));
  }

  function stepTyping(dtSeconds) {
    if (!game.currentComment) {
      if (game.nextCommentDelayMs > 0) {
        game.nextCommentDelayMs -= dtSeconds * 1000;
        return;
      }
      startNewComment();
    }

    const rate = BASE_TYPING_CHARS_PER_SEC * game.motorFactor * (1 - game.fatigue * 0.5);
    game.typedCharsAccum += Math.max(0.2, rate) * dtSeconds;
    const targetChars = Math.min(game.currentComment.length, Math.floor(game.typedCharsAccum));

    while (game.typedChars < targetChars) {
      game.typedChars += 1;
      game.fatigue = Math.min(1.4, game.fatigue + FATIGUE_PER_KEYSTROKE);
      playKeyClick();
      if (neuroSim) neuroSim.stimulate('screen', 8, 120);
    }

    if (game.typedChars >= game.currentComment.length) {
      postCurrentComment();
    }

    game.fatigue = Math.min(1.4, game.fatigue + FATIGUE_PER_SECOND_TYPING * dtSeconds);
    game.fatigue = Math.max(0, game.fatigue - FATIGUE_RECOVERY_PER_SECOND * dtSeconds);
  }

  function enterCollapse() {
    game.phase = 'collapsed';
    game.collapseTimer = 0;
    safeSetRagdoll(1);
    setStatus('Муха выдохлась.', 'error');
    logEvent('Муха выдохлась и рушилась.');
  }

  function startBeetleEntry() {
    game.phase = 'beetle_entering';
    beetle.state = 'entering';
    beetle.progress = 0;
    if (beetle.group) beetle.group.visible = true;
    logEvent('Жук-надзиратель идёт к мухе.');
  }

  function performPoke() {
    game.phase = 'poking';
    if (dom.subtitle) dom.subtitle.textContent = pickBeetleLine();
    playStartleSting();
    const dir = new THREE.Vector3(0, 0.6, -0.7).normalize();
    safeApplyImpulse(dir, 2.4);
    if (neuroSim) neuroSim.stimulate('startle', 60, 250);
    game.forcedCount += 1;
    if (dom.forcedCount) dom.forcedCount.textContent = String(game.forcedCount);
    game.ragdollRamp = 1;
    setStatus('жук-надзиратель заставляет муху работать.', 'loading');
  }

  function stepBeetlePhase(dtSeconds) {
    if (game.phase === 'beetle_entering') {
      beetle.progress += dtSeconds / 2.4;
      const p = Math.min(1, beetle.progress);
      if (beetle.group) {
        beetle.group.position.x = beetle.homeX + (beetle.workX - beetle.homeX) * p;
      }
      updateBeetleWalkAnimation(world.clock.getElapsedTime(), true);
      if (p >= 1) {
        performPoke();
      }
      return;
    }

    if (game.phase === 'poking') {
      game.ragdollRamp = Math.max(0, game.ragdollRamp - dtSeconds * (1000 / RAGDOLL_RECOVER_DURATION_MS));
      safeSetRagdoll(game.ragdollRamp);
      updateBeetleWalkAnimation(world.clock.getElapsedTime(), false);
      if (game.ragdollRamp <= 0) {
        game.fatigue = RESUME_FATIGUE;
        game.phase = 'beetle_leaving';
        beetle.progress = 0;
        setStatus('Муха вернулась к работе.', 'ready');
        logEvent('Муха вернулась к комментариям.');
        if (dom.subtitle) dom.subtitle.textContent = '';
      }
      return;
    }

    if (game.phase === 'beetle_leaving') {
      beetle.progress += dtSeconds / 2.0;
      const p = Math.min(1, beetle.progress);
      if (beetle.group) {
        beetle.group.position.x = beetle.workX + (beetle.homeX - beetle.workX) * p;
      }
      updateBeetleWalkAnimation(world.clock.getElapsedTime(), true);
      if (p >= 1) {
        if (beetle.group) beetle.group.visible = false;
        game.phase = 'typing';
      }
    }
  }

  function stepGame(dtSeconds) {
    updateTypingRate(dtSeconds);

    if (game.phase === 'typing') {
      stepTyping(dtSeconds);
      if (game.fatigue >= 1) {
        enterCollapse();
      }
    } else if (game.phase === 'collapsed') {
      game.collapseTimer += dtSeconds * 1000;
      if (game.collapseTimer >= BEETLE_ENTRY_DELAY_MS) {
        startBeetleEntry();
      }
    } else {
      stepBeetlePhase(dtSeconds);
    }

    if (dom.fatigueFill) dom.fatigueFill.style.width = `${Math.min(100, game.fatigue * 100).toFixed(0)}%`;
    if (dom.fatigueValue) dom.fatigueValue.textContent = `${Math.min(100, Math.round(game.fatigue * 100))}%`;
    if (dom.angerFill) dom.angerFill.style.width = `${game.anger.toFixed(0)}%`;
    if (dom.angerValue) {
      dom.angerValue.textContent = neuroSim ? `${Math.round(game.anger)}%` : 'нет данных';
    }

    // Like counters tick up gently on already-posted comments for life.
    game.likeTickTimer -= dtSeconds;
    if (game.likeTickTimer <= 0 && game.feed.length) {
      game.likeTickTimer = 0.6 + Math.random() * 1.2;
      const idx = Math.floor(Math.random() * game.feed.length);
      game.feed[idx].likes += 1;
    }

    drawCommentFeedTexture(
      game.feed,
      game.currentComment.slice(0, game.typedChars),
      Math.floor(world.clock.getElapsedTime() * 2) % 2 === 0,
    );

    const flyPose = game.phase === 'typing' ? 'typing'
      : (game.phase === 'collapsed' || game.phase === 'beetle_entering' || game.phase === 'poking') ? 'collapsed'
      : 'typing';
    const typingRate = game.phase === 'typing'
      ? BASE_TYPING_CHARS_PER_SEC * game.motorFactor * (1 - game.fatigue * 0.5)
      : 0;

    if (fly.rig && typeof fly.rig.update === 'function') {
      fly.rig.update(world.clock.getElapsedTime(), 0, 'agitated', {
        pose: flyPose,
        typingRate: Math.max(0, Math.min(12, typingRate)),
        fatigue: Math.min(1, game.fatigue),
        dt: dtSeconds,
      });
    }
  }

  // =========================================================================
  // MAIN LOOP
  // =========================================================================
  let lastFrameTime = performance.now();

  function frame(now) {
    requestAnimationFrame(frame);
    if (document.hidden) {
      lastFrameTime = now;
      return;
    }
    let dt = (now - lastFrameTime) / 1000;
    dt = Math.max(0, Math.min(0.1, dt)); // clamp to avoid huge jumps after a pause
    lastFrameTime = now;
    const t = world.clock.getElapsedTime();

    if (neuroSim) {
      try {
        neuroSim.step(dt);
      } catch (err) {
        logEvent(`Симуляция коннектома остановилась: ${err && err.message ? err.message : err}`);
        neuroSim = null;
      }
      if (neuroSim) neuroSim.setDrive(Math.max(0, Math.min(2, 1 - 0.6 * Math.min(1, game.fatigue))));
    }

    stepGame(dt);

    let summary = null;
    if (neuroSim) {
      try {
        summary = neuroSim.getSummary();
      } catch (err) {
        summary = null;
      }
    }

    if (summary) {
      updateBrainColors(summary.displaySpikes, dt);
      if (dom.spikesRate) dom.spikesRate.textContent = formatSpikesPerSecond(summary.spikesPerSecond);
      if (dom.telemetryRate) {
        const pamHz = (summary.groupRates && summary.groupRates.pam) || 0;
        dom.telemetryRate.textContent = `${pamHz.toFixed(1)} Гц (модель)`;
      }
      const scopeCtx = dom.scopeCanvas && dom.scopeCanvas.getContext('2d');
      if (scopeCtx) {
        drawOscilloscope(scopeCtx, dom.scopeCanvas.clientWidth || 320, dom.scopeCanvas.clientHeight || 110,
          (summary.groupRates && summary.groupRates.pam) || 0, true);
      }
      const rasterCtx = dom.rasterCanvas && dom.rasterCanvas.getContext('2d');
      if (rasterCtx) {
        drawRaster(rasterCtx, dom.rasterCanvas.clientWidth || 320, dom.rasterCanvas.clientHeight || 110,
          summary.raster, true);
      }
    } else {
      if (dom.spikesRate) dom.spikesRate.textContent = 'нет данных';
      if (dom.telemetryRate) dom.telemetryRate.textContent = 'нет данных';
      const scopeCtx = dom.scopeCanvas && dom.scopeCanvas.getContext('2d');
      if (scopeCtx) {
        drawOscilloscope(scopeCtx, dom.scopeCanvas.clientWidth || 320, dom.scopeCanvas.clientHeight || 110, 0, false);
      }
      const rasterCtx = dom.rasterCanvas && dom.rasterCanvas.getContext('2d');
      if (rasterCtx) {
        drawRaster(rasterCtx, dom.rasterCanvas.clientWidth || 320, dom.rasterCanvas.clientHeight || 110, null, false);
      }
    }

    if (audio.enabled && audio.analyser) {
      drawEqualizer(audio.analyser, audio.freqData);
    } else {
      decayEqualizerIdle();
    }

    renderBrainScene(t);

    if (world.renderer && world.scene && world.camera) {
      fitRendererToCanvas(world.renderer, world.camera, dom.worldCanvas);
      world.renderer.render(world.scene, world.camera);
    }
  }

  // =========================================================================
  // INIT
  // =========================================================================
  async function init() {
    setup2dCanvasSizing();
    buildEqualizerBars();
    setupSoundToggle();
    createBrainScene();
    createWorldScene();
    buildBeetle();

    setStatus('Загрузка сцены…', 'loading');

    const flyReady = await initFlyRig();

    let neuronPayload = null;
    try {
      neuronPayload = await fetchNeurons();
      buildBrainPointCloud(neuronPayload);
      logEvent(`Загружено ${neuronPayload.points.length} точек из /api/neurons.`);
    } catch (err) {
      const msg = err && err.message ? err.message : String(err);
      setStatus(`Ошибка загрузки нейронов: ${msg}`, 'error');
      logEvent(`Ошибка /api/neurons: ${msg}`);
      showErrorBanner(`Не удалось загрузить данные нейронов: ${msg}`, true);
      if (dom.sourceLabel) dom.sourceLabel.textContent = 'источник данных недоступен';
    }

    hideLoadingScreen();
    if (flyReady) {
      setStatus('Муха пишет комментарии.', 'ready');
    }

    // Connectome sim loads in the background; the brain panel shows its own
    // progress bar and never blocks the rest of the scene.
    if (neuronPayload) {
      const displayIndices = neuronPayload.points.map((p, i) => (typeof p.index === 'number' ? p.index : i));
      neuroSim = await loadNeuroSim(displayIndices);
    } else {
      neuroSimFailed = true;
      if (dom.backendLabel) dom.backendLabel.textContent = 'Бэкенд: недоступен';
      if (dom.brainProgressText) dom.brainProgressText.textContent = 'симуляция недоступна (нет точек для сопоставления)';
    }

    startNewComment();
    requestAnimationFrame(frame);
  }

  init().catch((err) => {
    const msg = err && err.message ? err.message : String(err);
    setStatus(`Критическая ошибка: ${msg}`, 'error');
    showErrorBanner(`Критическая ошибка инициализации: ${msg}`, true);
    hideLoadingScreen();
  });
})();
