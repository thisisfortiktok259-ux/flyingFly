/**
 * neuro_sim.js
 * =============================================================================
 * Whole-connectome spiking simulation for the browser.
 *
 * WHAT THIS IS (honest scope statement)
 * -------------------------------------
 * This module simulates ALL 166,700 neurons and 25,582,938 directed edges of
 * the MaleCNS v1.0 connectome with an ADAPTED leaky integrate-and-fire (LIF)
 * model. It is NOT a validated emulation of a fly brain. The wiring is real;
 * the dynamics are a simple point-neuron approximation in the style of
 * Shiu et al. 2024 (Nature, s41586-024-07763-9), with the background-drive and
 * stimulation parameters chosen here for interactive use. Nothing produced by
 * this module should be read as a biological prediction.
 *
 * BACKENDS
 * --------
 *   'webgpu'     : compute shaders, i32 atomics, chunked edge buffers.
 *   'cpu-worker' : Web Worker (Blob URL) running the same model on typed
 *                  arrays at a reduced simulation rate.
 * If neither is available, createConnectomeSim() throws.
 *
 * DATA CONTRACT (served by the local Python backend as raw gzip bytes)
 * --------------------------------------------------------------------
 *   GET /data/manifest.json
 *   GET /data/<part file>        gzip of a little-endian binary array part
 *   GET /data/neurons.json.gz    gzip of a JSON array of metadata rows
 *
 * The element dtype of the binary arrays is NOT documented by the manifest.
 * It is inferred as bytesPerElement = totalDecompressedBytes / array.length
 * and must be exactly 1, 2, 4 or 8. Anything else fails loudly.
 *
 * CSR layout is BY SOURCE:
 *   for neuron i, its outgoing edges are e in [offsets[i], offsets[i+1])
 *   sources[e] = TARGET neuron index   (the array name is misleading)
 *   counts[e]  = synapse count for that edge
 *
 * @module neuro_sim
 */

/* =========================================================================
 * 1. MODEL CONSTANTS  (single place to tune anything)
 * =========================================================================
 * Membrane / spike parameters follow Shiu et al. 2024:
 *   tau_m 20 ms, V_th -45 mV, V_rest = V_reset = -52 mV, refractory 2.2 ms,
 *   0.275 mV of depolarisation per synapse, sign from the PRESYNAPTIC
 *   neuron's fast transmitter (ACh +1, GABA/Glu -1, other 0).
 * Added here, NOT from the paper:
 *   a 5 ms exponential synaptic current, a one-step synaptic delay, a small
 *   per-neuron Poisson background drive so the network is not silent, and an
 *   external stimulation API that injects extra Poisson events.
 */
const DT_MS = 0.5;                  // integration step, milliseconds
const TAU_M_MS = 20.0;              // membrane time constant
const TAU_SYN_MS = 5.0;             // synaptic current decay
const V_REST_MV = -52.0;            // resting potential
const V_RESET_MV = -52.0;           // reset potential (same as rest)
const V_TH_MV = -45.0;              // threshold (7 mV above rest)
const REFRACTORY_MS = 2.2;          // absolute refractory period
const REFRACTORY_STEPS = Math.max(1, Math.ceil(REFRACTORY_MS / DT_MS)); // 5 steps = 2.5 ms
const W_PER_SYNAPSE_MV = 0.275;     // depolarisation per synapse (time integral)
const WEIGHT_FIXED_SCALE = 4096;    // fixed-point scale for i32 atomicAdd
const BACKGROUND_RATE_HZ = 50.0;    // per-neuron Poisson background (tunable)
const EXT_KICK_MV = 0.9;            // voltage kick per background / stim event
const MAX_STEPS_PER_FRAME = 20;     // 20 * 0.5 ms = 10 ms of sim per frame
const RATE_WINDOW_MS = 50.0;        // averaging window for reported rates
const MAX_DISPLAY_INDICES = 20000;  // cap on the display readback list
const DATA_BASE_DEFAULT = '/data';  // backend mount point
const FETCH_CONCURRENCY = 4;        // parallel part downloads
const STAGING_BUFFERS = 3;          // async readback ring (never blocks)

const V_LEAK = 1.0 / TAU_M_MS;                  // 1/tau_m, per ms
const G_DECAY = Math.exp(-DT_MS / TAU_SYN_MS);  // synaptic decay per step
const DT_SEC = DT_MS / 1000.0;

const GROUP_NAMES = ['pam', 'dan', 'motor', 'descending', 'visual'];
const GROUP_BIT = { pam: 0, dan: 1, motor: 2, descending: 3, visual: 4 };

/* Group membership regexes, applied to the metadata columns:
 *   pam        : type starts with 'PAM'
 *   dan        : consensusNT contains 'dopamine' (all dopaminergic, incl. PAM)
 *   motor      : superclass matches /motor/i
 *   descending : superclass matches /descending/i
 *   visual     : superclass matches /visual|optic|sensory/i
 * An empty group is reported with size 0. Nothing is invented to fill it.
 */
const RX_PAM = /^PAM/;
const RX_DOPAMINE = /dopamine/i;
const RX_MOTOR = /motor/i;
const RX_DESCENDING = /descending/i;
const RX_VISUAL = /visual|optic|sensory/i;
const RX_MECHANO = /mechano/i;
const RX_GIANT_FIBER = /^DNp01/;
const SCREEN_STIM_FRACTION = 0.02;  // 2% of the visual/sensory superclass

/* =========================================================================
 * 2. UTILITIES
 * ========================================================================= */

class NeuroSimError extends Error {
  constructor(message, cause) {
    super(message);
    this.name = 'NeuroSimError';
    if (cause !== undefined) this.cause = cause;
  }
}

function assert(cond, message) {
  if (!cond) throw new NeuroSimError(message);
}

function hasDecompressionStream() {
  return typeof DecompressionStream === 'function';
}

/**
 * Download one gzip part and return its decompressed bytes. Reports raw
 * (compressed) byte counts to onBytes so progress can use manifest sizes.
 */
async function fetchGzipPart(url, onBytes) {
  let res;
  try {
    res = await fetch(url);
  } catch (err) {
    throw new NeuroSimError('Network failure fetching ' + url, err);
  }
  if (!res.ok) throw new NeuroSimError('HTTP ' + res.status + ' fetching ' + url);
  assert(res.body, 'Streaming body unavailable for ' + url);

  const counter = new TransformStream({
    transform(chunk, controller) {
      if (onBytes) onBytes(chunk.byteLength);
      controller.enqueue(chunk);
    }
  });

  let stream;
  try {
    stream = res.body.pipeThrough(counter).pipeThrough(new DecompressionStream('gzip'));
  } catch (err) {
    throw new NeuroSimError('Failed to start gzip decompression for ' + url, err);
  }

  const chunks = [];
  let total = 0;
  const reader = stream.getReader();
  try {
    for (;;) {
      const r = await reader.read();
      if (r.done) break;
      chunks.push(r.value);
      total += r.value.byteLength;
    }
  } catch (err) {
    throw new NeuroSimError(
      'gzip decompression failed for ' + url + ' (corrupt data, or the server sent ' +
      'Content-Encoding: gzip so the browser already inflated the body)', err);
  }

  const out = new Uint8Array(total);
  let off = 0;
  for (let i = 0; i < chunks.length; i++) {
    out.set(chunks[i], off);
    off += chunks[i].byteLength;
    chunks[i] = null; // release early, these are large
  }
  return out;
}

/** Run async jobs with a bounded number in flight. Preserves result order. */
async function mapConcurrent(items, limit, worker) {
  const results = new Array(items.length);
  let next = 0;
  const runners = [];
  const n = Math.min(limit, items.length);
  for (let k = 0; k < n; k++) {
    runners.push((async () => {
      for (;;) {
        const idx = next++;
        if (idx >= items.length) return;
        results[idx] = await worker(items[idx], idx);
      }
    })());
  }
  await Promise.all(runners);
  return results;
}

/** Infer bytes-per-element from the decompressed size. Fails loudly. */
function inferBytesPerElement(totalBytes, length, name) {
  assert(length > 0, 'Array ' + name + ' declares length ' + length);
  const bpe = totalBytes / length;
  if (!Number.isInteger(bpe)) {
    throw new NeuroSimError(
      'Cannot infer dtype for array ' + name + ': ' + totalBytes +
      ' decompressed bytes / ' + length + ' elements = ' + bpe + ' (not an integer)');
  }
  if (bpe !== 1 && bpe !== 2 && bpe !== 4 && bpe !== 8) {
    throw new NeuroSimError(
      'Unsupported dtype for array ' + name + ': ' + bpe +
      ' bytes per element (only 1, 2, 4, 8 are supported)');
  }
  return bpe;
}

/** Convert raw little-endian bytes to a Uint32Array of the given length. */
function bytesToUint32(bytes, length, bpe, name) {
  if (bpe === 4) {
    return new Uint32Array(bytes.buffer, bytes.byteOffset, length);
  }
  const out = new Uint32Array(length);
  if (bpe === 1) {
    for (let i = 0; i < length; i++) out[i] = bytes[i];
  } else if (bpe === 2) {
    out.set(new Uint16Array(bytes.buffer, bytes.byteOffset, length));
  } else {
    assert(typeof BigUint64Array === 'function', 'BigUint64Array unavailable for ' + name);
    const v = new BigUint64Array(bytes.buffer, bytes.byteOffset, length);
    const LIMIT = 4294967296n;
    for (let i = 0; i < length; i++) {
      const b = v[i];
      if (b >= LIMIT) {
        throw new NeuroSimError(
          'Array ' + name + ' element ' + i + ' = ' + b.toString() + ' does not fit in uint32');
      }
      out[i] = Number(b);
    }
  }
  return out;
}

/* =========================================================================
 * 3. DATA LOADING
 * ========================================================================= */

async function loadManifest(base) {
  const url = base + '/manifest.json';
  let res;
  try {
    res = await fetch(url);
  } catch (err) {
    throw new NeuroSimError('Cannot reach the data backend at ' + url, err);
  }
  if (!res.ok) throw new NeuroSimError('HTTP ' + res.status + ' fetching ' + url);
  const m = await res.json();
  assert(Array.isArray(m.arrays), 'manifest.arrays missing');
  assert(typeof m.neurons === 'number', 'manifest.neurons missing');
  assert(typeof m.edges === 'number', 'manifest.edges missing');
  return m;
}

function manifestArray(manifest, name) {
  const a = manifest.arrays.find((x) => x.name === name);
  assert(a, 'manifest is missing the ' + name + ' array');
  assert(Array.isArray(a.parts) && a.parts.length > 0, 'array ' + name + ' has no parts');
  return a;
}

/** Download every part of one manifest array and return it as Uint32Array. */
async function loadUint32Array(base, arrayDesc, onBytes) {
  const parts = await mapConcurrent(arrayDesc.parts, FETCH_CONCURRENCY, (p) =>
    fetchGzipPart(base + '/' + p.file, onBytes));

  let total = 0;
  for (let i = 0; i < parts.length; i++) total += parts[i].byteLength;

  const bpe = inferBytesPerElement(total, arrayDesc.length, arrayDesc.name);
  const joined = new Uint8Array(total);
  let off = 0;
  for (let i = 0; i < parts.length; i++) {
    joined.set(parts[i], off);
    off += parts[i].byteLength;
    parts[i] = null;
  }
  return { data: bytesToUint32(joined, arrayDesc.length, bpe, arrayDesc.name), bytesPerElement: bpe };
}

/** Parse neurons.json.gz. Rows may be arrays (column order) or objects. */
async function loadMetadata(base, manifest) {
  const file = manifest.metadata || 'neurons.json.gz';
  const bytes = await fetchGzipPart(base + '/' + file, null);
  let rows;
  try {
    rows = JSON.parse(new TextDecoder().decode(bytes));
  } catch (err) {
    throw new NeuroSimError('neurons metadata is not valid JSON', err);
  }
  assert(Array.isArray(rows), 'neurons metadata must be a JSON array of rows');

  const cols = manifest.metadataColumns ||
    ['bodyId', 'type', 'superclass', 'side', 'consensusNT', 'fastSign', 'somaLocation8nm'];
  const colIndex = {};
  cols.forEach((c, i) => { colIndex[c] = i; });

  const n = rows.length;
  const type = new Array(n);
  const superclass = new Array(n);
  const consensusNT = new Array(n);
  const bodyId = new Float64Array(n);
  const fastSign = new Int8Array(n);

  const pick = (row, key) => {
    if (Array.isArray(row)) {
      const i = colIndex[key];
      return i === undefined ? null : row[i];
    }
    return row[key] === undefined ? null : row[key];
  };

  for (let i = 0; i < n; i++) {
    const r = rows[i];
    type[i] = String(pick(r, 'type') || '');
    superclass[i] = String(pick(r, 'superclass') || '');
    consensusNT[i] = String(pick(r, 'consensusNT') || '');
    const b = pick(r, 'bodyId');
    bodyId[i] = b === null ? 0 : Number(b);
    const fs = pick(r, 'fastSign');
    if (fs === null || fs === '' || fs === undefined) {
      // Fall back to the manifest assumption mapping when fastSign is absent.
      const nt = consensusNT[i].toLowerCase();
      fastSign[i] = (nt.indexOf('acetylcholine') >= 0 || nt === 'ach') ? 1
        : ((nt.indexOf('gaba') >= 0 || nt.indexOf('glutamate') >= 0) ? -1 : 0);
    } else {
      const v = Number(fs);
      fastSign[i] = v > 0 ? 1 : (v < 0 ? -1 : 0);
    }
    rows[i] = null;
  }
  return { count: n, type: type, superclass: superclass, consensusNT: consensusNT, bodyId: bodyId, fastSign: fastSign };
}

/* =========================================================================
 * 4. GROUPS AND STIMULATION TARGETS
 * ========================================================================= */

function buildGroups(meta, neuronCount) {
  const mask = new Uint32Array(neuronCount);
  const members = { pam: [], dan: [], motor: [], descending: [], visual: [] };
  const superclassCounts = Object.create(null);

  for (let i = 0; i < neuronCount; i++) {
    const known = i < meta.count;
    const t = known ? meta.type[i] : '';
    const sc = known ? meta.superclass[i] : '';
    const nt = known ? meta.consensusNT[i] : '';
    const key = sc || '(unknown)';
    superclassCounts[key] = (superclassCounts[key] || 0) + 1;

    let m = 0;
    if (RX_PAM.test(t)) { m |= 1 << GROUP_BIT.pam; members.pam.push(i); }
    if (RX_DOPAMINE.test(nt)) { m |= 1 << GROUP_BIT.dan; members.dan.push(i); }
    if (RX_MOTOR.test(sc)) { m |= 1 << GROUP_BIT.motor; members.motor.push(i); }
    if (RX_DESCENDING.test(sc)) { m |= 1 << GROUP_BIT.descending; members.descending.push(i); }
    if (RX_VISUAL.test(sc)) { m |= 1 << GROUP_BIT.visual; members.visual.push(i); }
    mask[i] = m;
  }

  const groups = {};
  for (let k = 0; k < GROUP_NAMES.length; k++) {
    groups[GROUP_NAMES[k]] = { size: members[GROUP_NAMES[k]].length };
  }
  return { mask: mask, members: members, groups: groups, superclassCounts: superclassCounts };
}

/**
 * Stimulation targets:
 *   screen  : deterministic 2% sample of the visual/sensory superclass
 *   reward  : the PAM group
 *   startle : neurons whose type or superclass matches /mechano/i; if empty,
 *             the DNp01 giant-fibre descending type; if that is also absent the
 *             target stays empty and stimulate() warns once and does nothing.
 */
function buildStimTargets(meta, groupInfo, neuronCount) {
  const visual = groupInfo.members.visual;
  const screen = [];
  const stride = Math.max(1, Math.round(1 / SCREEN_STIM_FRACTION));
  for (let k = 0; k < visual.length; k += stride) screen.push(visual[k]);

  const limit = Math.min(neuronCount, meta.count);
  const mechano = [];
  for (let i = 0; i < limit; i++) {
    if (RX_MECHANO.test(meta.type[i]) || RX_MECHANO.test(meta.superclass[i])) mechano.push(i);
  }
  let startle = mechano;
  let startleSource = 'type/superclass matches /mechano/i';
  if (startle.length === 0) {
    const gf = [];
    for (let i = 0; i < limit; i++) {
      if (RX_GIANT_FIBER.test(meta.type[i])) gf.push(i);
    }
    startle = gf;
    startleSource = 'DNp01 giant fibre (no mechanosensory match found)';
  }

  return {
    screen: { indices: Int32Array.from(screen), source: '2% sample of visual/sensory superclass' },
    reward: { indices: Int32Array.from(groupInfo.members.pam), source: 'PAM type prefix' },
    startle: { indices: Int32Array.from(startle), source: startleSource }
  };
}

/** Raster indices: spread across the named groups, then across all neurons. */
function buildRasterIndices(groupInfo, neuronCount, rasterCount) {
  const out = [];
  const seen = new Set();
  const push = (i) => {
    if (i >= 0 && i < neuronCount && !seen.has(i) && out.length < rasterCount) {
      seen.add(i);
      out.push(i);
    }
  };
  const perGroup = Math.floor(rasterCount / (GROUP_NAMES.length + 1));
  for (let n = 0; n < GROUP_NAMES.length; n++) {
    const list = groupInfo.members[GROUP_NAMES[n]];
    if (list.length === 0) continue;
    const step = Math.max(1, Math.floor(list.length / Math.max(1, perGroup)));
    for (let k = 0; k < list.length && out.length < rasterCount; k += step) push(list[k]);
  }
  const step = Math.max(1, Math.floor(neuronCount / Math.max(1, rasterCount)));
  for (let i = 0; i < neuronCount && out.length < rasterCount; i += step) push(i);
  for (let i = 0; i < neuronCount && out.length < rasterCount; i++) push(i);
  return Int32Array.from(out);
}

/* =========================================================================
 * 5. WGSL SHADERS
 * ========================================================================= */

const WGSL_INTEGRATE = `
struct Params {
  neuronCount : u32,
  refrSteps : u32,
  pad0 : u32,
  pad1 : u32,
  vRest : f32,
  vTh : f32,
  vReset : f32,
  dtMs : f32,
  vLeak : f32,
  gDecay : f32,
  weightScale : f32,
  pBase : f32,
  extKick : f32,
  dtSec : f32,
  pad2 : f32,
  pad3 : f32,
};

@group(0) @binding(0) var<storage, read_write> v : array<f32>;
@group(0) @binding(1) var<storage, read_write> g : array<f32>;
@group(0) @binding(2) var<storage, read_write> gin : array<atomic<i32>>;
@group(0) @binding(3) var<storage, read_write> state : array<u32>;
@group(0) @binding(4) var<storage, read_write> rng : array<u32>;
@group(0) @binding(5) var<storage, read> stim : array<f32>;
@group(0) @binding(6) var<storage, read> meta : array<u32>;
@group(0) @binding(7) var<storage, read_write> counters : array<atomic<u32>>;
@group(0) @binding(8) var<uniform> P : Params;

fn hash01(seed : u32) -> f32 {
  var x = seed;
  x = x ^ (x >> 16u);
  x = x * 2246822519u;
  x = x ^ (x >> 13u);
  x = x * 3266489917u;
  x = x ^ (x >> 16u);
  return f32(x) * 2.3283064365386963e-10;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= P.neuronCount) { return; }

  // Arrivals deposited by the previous step: this is the one-step delay.
  let arrived = f32(atomicExchange(&gin[i], 0)) / P.weightScale;
  let gv = g[i] * P.gDecay + arrived;
  g[i] = gv;

  let seed = rng[i] + 2891336453u;
  rng[i] = seed;
  let u = hash01(seed ^ (i * 747796405u));
  let p = P.pBase + stim[i] * P.dtSec;
  var kick = 0.0;
  if (u < p) { kick = P.extKick; }

  var refr = state[i] & 0xffffu;
  var vv = v[i];
  var fired = 0u;
  if (refr > 0u) {
    refr = refr - 1u;
    vv = P.vReset;
  } else {
    vv = vv + P.dtMs * (P.vLeak * (P.vRest - vv) + gv) + kick;
    if (vv >= P.vTh) {
      vv = P.vReset;
      refr = P.refrSteps;
      fired = 1u;
    }
  }
  v[i] = vv;
  state[i] = refr | (fired << 16u);

  if (fired == 1u) {
    atomicAdd(&counters[0], 1u);
    let m = meta[i];
    for (var b = 0u; b < 5u; b = b + 1u) {
      if ((m & (1u << b)) != 0u) { atomicAdd(&counters[1u + b], 1u); }
    }
  }
}
`;

const WGSL_PROPAGATE = `
struct Chunk {
  nStart : u32,
  nEnd : u32,
  edgeBase : u32,
  pad : u32,
};

@group(0) @binding(0) var<storage, read_write> gin : array<atomic<i32>>;
@group(0) @binding(1) var<storage, read> state : array<u32>;
@group(0) @binding(2) var<storage, read> offsets : array<u32>;
@group(0) @binding(3) var<storage, read> targets : array<u32>;
@group(0) @binding(4) var<storage, read> weights : array<i32>;
@group(0) @binding(5) var<uniform> C : Chunk;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = C.nStart + gid.x;
  if (i >= C.nEnd) { return; }
  if ((state[i] >> 16u) == 0u) { return; }
  let s = offsets[i] - C.edgeBase;
  let e = offsets[i + 1u] - C.edgeBase;
  for (var k = s; k < e; k = k + 1u) {
    let w = weights[k];
    if (w != 0) { atomicAdd(&gin[targets[k]], w); }
  }
}
`;

const WGSL_SAMPLE = `
struct SampleParams {
  displayCount : u32,
  rasterCount : u32,
  mode : u32,
  pad : u32,
};

@group(0) @binding(0) var<storage, read> state : array<u32>;
@group(0) @binding(1) var<storage, read> dIdx : array<i32>;
@group(0) @binding(2) var<storage, read_write> dOut : array<u32>;
@group(0) @binding(3) var<storage, read> rIdx : array<i32>;
@group(0) @binding(4) var<storage, read_write> rOut : array<u32>;
@group(0) @binding(5) var<uniform> S : SampleParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let k = gid.x;
  if (k < S.displayCount) {
    if (S.mode == 0u) {
      dOut[k] = 0u;
    } else {
      let idx = dIdx[k];
      if (idx >= 0) {
        if ((state[u32(idx)] >> 16u) == 1u) { dOut[k] = 1u; }
      }
    }
  }
  if (k < S.rasterCount) {
    if (S.mode == 0u) {
      rOut[k] = 0u;
    } else {
      let idx = rIdx[k];
      if (idx >= 0) {
        if ((state[u32(idx)] >> 16u) == 1u) { rOut[k] = 1u; }
      }
    }
  }
}
`;

/* =========================================================================
 * 6. CPU WORKER SOURCE (same model, single thread, reduced rate)
 * ========================================================================= */

function cpuWorkerSource() {
  return [
    'let N = 0;',
    'let offsets, targets, weights, meta, stim;',
    'let v, g, gin, refr, spike;',
    'let P = null;',
    'let displayIdx = null, rasterIdx = null;',
    'let displayAcc = null, rasterAcc = null;',
    'let groupSpikes = new Float64Array(5);',
    'let totalSpikes = 0;',
    'let accMs = 0;',
    'let simTimeMs = 0;',
    'let running = false;',
    'let timer = null;',
    'const BUDGET_MS = 30;',
    'const TICK_MS = 50;',
    'const clock = self.performance || Date;',
    '',
    'function stepOnce() {',
    '  const dtMs = P.dtMs, vLeak = P.vLeak, vRest = P.vRest, vReset = P.vReset;',
    '  const vTh = P.vTh, gDecay = P.gDecay, scale = P.weightScale;',
    '  const refrSteps = P.refrSteps, pBase = P.pBase, extKick = P.extKick, dtSec = P.dtSec;',
    '  for (let i = 0; i < N; i++) {',
    '    const arrived = gin[i] / scale;',
    '    gin[i] = 0;',
    '    const gv = g[i] * gDecay + arrived;',
    '    g[i] = gv;',
    '    let r = refr[i];',
    '    let vv = v[i];',
    '    let fired = 0;',
    '    const p = pBase + stim[i] * dtSec;',
    '    const kick = Math.random() < p ? extKick : 0;',
    '    if (r > 0) { r = r - 1; vv = vReset; }',
    '    else {',
    '      vv = vv + dtMs * (vLeak * (vRest - vv) + gv) + kick;',
    '      if (vv >= vTh) { vv = vReset; r = refrSteps; fired = 1; }',
    '    }',
    '    v[i] = vv; refr[i] = r; spike[i] = fired;',
    '    if (fired === 1) {',
    '      totalSpikes++;',
    '      const m = meta[i];',
    '      for (let b = 0; b < 5; b++) { if ((m & (1 << b)) !== 0) groupSpikes[b]++; }',
    '    }',
    '  }',
    '  for (let i = 0; i < N; i++) {',
    '    if (spike[i] === 0) continue;',
    '    const s = offsets[i], e = offsets[i + 1];',
    '    for (let k = s; k < e; k++) { gin[targets[k]] += weights[k]; }',
    '  }',
    '  for (let k = 0; k < displayIdx.length; k++) {',
    '    const idx = displayIdx[k];',
    '    if (idx >= 0 && spike[idx] === 1) displayAcc[k] = 1;',
    '  }',
    '  for (let k = 0; k < rasterIdx.length; k++) {',
    '    const idx = rasterIdx[k];',
    '    if (idx >= 0 && spike[idx] === 1) rasterAcc[k] = 1;',
    '  }',
    '  simTimeMs += dtMs;',
    '  accMs += dtMs;',
    '}',
    '',
    'function tick() {',
    '  if (!running) return;',
    '  const t0 = clock.now();',
    '  let steps = 0;',
    '  while (steps < 64 && (clock.now() - t0) < BUDGET_MS) { stepOnce(); steps++; }',
    '  self.postMessage({',
    '    type: \'summary\',',
    '    simTimeMs: simTimeMs,',
    '    windowMs: accMs,',
    '    totalSpikes: totalSpikes,',
    '    groupSpikes: Array.prototype.slice.call(groupSpikes),',
    '    display: displayAcc.slice(),',
    '    raster: rasterAcc.slice()',
    '  });',
    '  totalSpikes = 0;',
    '  accMs = 0;',
    '  groupSpikes.fill(0);',
    '  displayAcc.fill(0);',
    '  rasterAcc.fill(0);',
    '  timer = setTimeout(tick, TICK_MS);',
    '}',
    '',
    'self.onmessage = function (ev) {',
    '  const msg = ev.data;',
    '  if (msg.type === \'init\') {',
    '    N = msg.neuronCount;',
    '    offsets = msg.offsets;',
    '    targets = msg.targets;',
    '    weights = msg.weights;',
    '    meta = msg.meta;',
    '    P = msg.params;',
    '    stim = new Float32Array(N);',
    '    v = new Float32Array(N); v.fill(P.vRest);',
    '    g = new Float32Array(N);',
    '    gin = new Float64Array(N);',
    '    refr = new Int32Array(N);',
    '    spike = new Uint8Array(N);',
    '    displayIdx = msg.displayIndices;',
    '    rasterIdx = msg.rasterIndices;',
    '    displayAcc = new Uint8Array(displayIdx.length);',
    '    rasterAcc = new Uint8Array(rasterIdx.length);',
    '    running = true;',
    '    self.postMessage({ type: \'ready\' });',
    '    timer = setTimeout(tick, 0);',
    '  } else if (msg.type === \'stim\') {',
    '    for (let k = 0; k < msg.indices.length; k++) stim[msg.indices[k]] = msg.rateHz;',
    '  } else if (msg.type === \'clearStim\') {',
    '    for (let k = 0; k < msg.indices.length; k++) stim[msg.indices[k]] = 0;',
    '  } else if (msg.type === \'drive\') {',
    '    P.pBase = msg.pBase;',
    '  } else if (msg.type === \'dispose\') {',
    '    running = false;',
    '    if (timer) clearTimeout(timer);',
    '    self.close();',
    '  }',
    '};'
  ].join('\n');
}

/* =========================================================================
 * 7. SHARED PREPARATION
 * ========================================================================= */

/**
 * Fixed-point weights.
 *   weight[e] = round(counts[e] * fastSign[source] * W_PER_SYNAPSE_MV
 *                     / TAU_SYN_MS * WEIGHT_FIXED_SCALE)
 * Dividing by tau_syn turns 'total depolarisation per spike' into the amplitude
 * of an exponential current whose time integral equals that value.
 */
function buildWeights(offsets, counts, fastSign, neuronCount, edgeCount) {
  const weights = new Int32Array(edgeCount);
  const base = (W_PER_SYNAPSE_MV / TAU_SYN_MS) * WEIGHT_FIXED_SCALE;
  let synapseCount = 0;
  for (let i = 0; i < neuronCount; i++) {
    const s = offsets[i];
    const e = offsets[i + 1];
    const sign = i < fastSign.length ? fastSign[i] : 0;
    const k = sign * base;
    for (let j = s; j < e; j++) {
      const c = counts[j];
      synapseCount += c;
      if (k !== 0) weights[j] = Math.round(c * k);
    }
  }
  return { weights: weights, synapseCount: synapseCount };
}

function sanitizeDisplayIndices(list, neuronCount) {
  const out = [];
  for (let k = 0; k < list.length && out.length < MAX_DISPLAY_INDICES; k++) {
    const i = list[k] | 0;
    out.push(i >= 0 && i < neuronCount ? i : -1);
  }
  return Int32Array.from(out);
}

/* =========================================================================
 * 8. WEBGPU BACKEND
 * ========================================================================= */

async function createWebGpuBackend(ctx, report) {
  if (typeof navigator === 'undefined' || !navigator.gpu) return null;
  let adapter = null;
  try {
    adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  } catch (err) {
    console.warn('[neuro_sim] requestAdapter failed', err);
    return null;
  }
  if (!adapter) return null;

  const lim = adapter.limits;
  let device = null;
  try {
    device = await adapter.requestDevice({
      requiredLimits: {
        maxStorageBufferBindingSize: lim.maxStorageBufferBindingSize,
        maxBufferSize: lim.maxBufferSize,
        maxComputeWorkgroupsPerDimension: lim.maxComputeWorkgroupsPerDimension
      }
    });
  } catch (err) {
    console.warn('[neuro_sim] requestDevice failed', err);
    return null;
  }

  const N = ctx.neuronCount;
  const maxBindBytes = Math.min(device.limits.maxStorageBufferBindingSize, device.limits.maxBufferSize);
  const maxEdgesPerBuffer = Math.floor(maxBindBytes / 4);
  assert(maxEdgesPerBuffer > 0, 'WebGPU device reports a zero-sized storage buffer limit');

  // Split the edge arrays into contiguous neuron ranges that each fit a buffer.
  const chunks = [];
  let start = 0;
  while (start < N) {
    const edgeBase = ctx.offsets[start];
    let end = start;
    while (end < N && (ctx.offsets[end + 1] - edgeBase) <= maxEdgesPerBuffer) end++;
    if (end === start) {
      throw new NeuroSimError(
        'Neuron ' + start + ' has ' + (ctx.offsets[start + 1] - edgeBase) +
        ' outgoing edges, more than a single storage buffer can hold (' +
        maxEdgesPerBuffer + ' elements). The WebGPU path cannot run on this device.');
    }
    chunks.push({ nStart: start, nEnd: end, edgeBase: edgeBase, edgeEnd: ctx.offsets[end] });
    start = end;
  }

  const created = [];
  const S = GPUBufferUsage.STORAGE;
  const CD = GPUBufferUsage.COPY_DST;
  const CS = GPUBufferUsage.COPY_SRC;
  const UNI = GPUBufferUsage.UNIFORM;
  const mk = (size, usage, label) => {
    const b = device.createBuffer({ size: Math.max(4, size), usage: usage, label: label });
    created.push(b);
    return b;
  };

  const bufV = mk(N * 4, S | CD, 'v');
  const bufG = mk(N * 4, S | CD, 'g');
  const bufGin = mk(N * 4, S | CD, 'gin');
  const bufState = mk(N * 4, S | CD, 'state');
  const bufRng = mk(N * 4, S | CD, 'rng');
  const bufStim = mk(N * 4, S | CD, 'stim');
  const bufMeta = mk(N * 4, S | CD, 'meta');
  const bufCounters = mk(32, S | CD | CS, 'counters');
  const bufParams = mk(64, UNI | CD, 'params');
  const bufOffsets = mk((N + 1) * 4, S | CD, 'offsets');

  const displayCount = ctx.displayIndices.length;
  const rasterCount = ctx.rasterIndices.length;
  const bufDIdx = mk(displayCount * 4, S | CD, 'displayIdx');
  const bufDOut = mk(displayCount * 4, S | CD | CS, 'displayOut');
  const bufRIdx = mk(rasterCount * 4, S | CD, 'rasterIdx');
  const bufROut = mk(rasterCount * 4, S | CD | CS, 'rasterOut');
  const bufSampleClear = mk(16, UNI | CD, 'sampleClear');
  const bufSampleRun = mk(16, UNI | CD, 'sampleRun');

  report(0.90, 'Uploading connectome to GPU');

  const vInit = new Float32Array(N);
  vInit.fill(V_REST_MV);
  const rngInit = new Uint32Array(N);
  for (let i = 0; i < N; i++) rngInit[i] = ((i + 1) * 2654435761) >>> 0;
  device.queue.writeBuffer(bufV, 0, vInit);
  device.queue.writeBuffer(bufG, 0, new Float32Array(N));
  device.queue.writeBuffer(bufGin, 0, new Int32Array(N));
  device.queue.writeBuffer(bufState, 0, new Uint32Array(N));
  device.queue.writeBuffer(bufRng, 0, rngInit);
  device.queue.writeBuffer(bufStim, 0, new Float32Array(N));
  device.queue.writeBuffer(bufMeta, 0, ctx.groupMask);
  device.queue.writeBuffer(bufOffsets, 0, ctx.offsets);
  if (displayCount > 0) device.queue.writeBuffer(bufDIdx, 0, ctx.displayIndices);
  if (rasterCount > 0) device.queue.writeBuffer(bufRIdx, 0, ctx.rasterIndices);
  device.queue.writeBuffer(bufSampleClear, 0, new Uint32Array([displayCount, rasterCount, 0, 0]));
  device.queue.writeBuffer(bufSampleRun, 0, new Uint32Array([displayCount, rasterCount, 1, 0]));

  for (let c = 0; c < chunks.length; c++) {
    const ch = chunks[c];
    const len = ch.edgeEnd - ch.edgeBase;
    ch.bufTargets = mk(len * 4, S | CD, 'targets' + c);
    ch.bufWeights = mk(len * 4, S | CD, 'weights' + c);
    ch.bufChunk = mk(16, UNI | CD, 'chunk' + c);
    if (len > 0) {
      device.queue.writeBuffer(ch.bufTargets, 0, ctx.targets, ch.edgeBase, len);
      device.queue.writeBuffer(ch.bufWeights, 0, ctx.weights, ch.edgeBase, len);
    }
    device.queue.writeBuffer(ch.bufChunk, 0, new Uint32Array([ch.nStart, ch.nEnd, ch.edgeBase, 0]));
    report(0.90 + 0.09 * ((c + 1) / chunks.length),
      'Uploading edge chunk ' + (c + 1) + ' of ' + chunks.length);
  }

  const pipeIntegrate = device.createComputePipeline({
    layout: 'auto',
    compute: { module: device.createShaderModule({ code: WGSL_INTEGRATE }), entryPoint: 'main' }
  });
  const pipePropagate = device.createComputePipeline({
    layout: 'auto',
    compute: { module: device.createShaderModule({ code: WGSL_PROPAGATE }), entryPoint: 'main' }
  });
  const pipeSample = device.createComputePipeline({
    layout: 'auto',
    compute: { module: device.createShaderModule({ code: WGSL_SAMPLE }), entryPoint: 'main' }
  });

  const bgIntegrate = device.createBindGroup({
    layout: pipeIntegrate.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: bufV } },
      { binding: 1, resource: { buffer: bufG } },
      { binding: 2, resource: { buffer: bufGin } },
      { binding: 3, resource: { buffer: bufState } },
      { binding: 4, resource: { buffer: bufRng } },
      { binding: 5, resource: { buffer: bufStim } },
      { binding: 6, resource: { buffer: bufMeta } },
      { binding: 7, resource: { buffer: bufCounters } },
      { binding: 8, resource: { buffer: bufParams } }
    ]
  });

  for (let c = 0; c < chunks.length; c++) {
    const ch = chunks[c];
    ch.bindGroup = device.createBindGroup({
      layout: pipePropagate.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: bufGin } },
        { binding: 1, resource: { buffer: bufState } },
        { binding: 2, resource: { buffer: bufOffsets } },
        { binding: 3, resource: { buffer: ch.bufTargets } },
        { binding: 4, resource: { buffer: ch.bufWeights } },
        { binding: 5, resource: { buffer: ch.bufChunk } }
      ]
    });
    ch.workgroups = Math.ceil((ch.nEnd - ch.nStart) / 256);
  }

  const sampleBindGroup = (uniform) => device.createBindGroup({
    layout: pipeSample.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: bufState } },
      { binding: 1, resource: { buffer: bufDIdx } },
      { binding: 2, resource: { buffer: bufDOut } },
      { binding: 3, resource: { buffer: bufRIdx } },
      { binding: 4, resource: { buffer: bufROut } },
      { binding: 5, resource: { buffer: uniform } }
    ]
  });
  const bgSampleClear = sampleBindGroup(bufSampleClear);
  const bgSampleRun = sampleBindGroup(bufSampleRun);

  // Async readback ring. mapAsync is never awaited on the critical path.
  const COUNTER_BYTES = 32;
  const D_OFFSET = COUNTER_BYTES;
  const D_BYTES = Math.max(4, displayCount * 4);
  const R_OFFSET = D_OFFSET + D_BYTES;
  const R_BYTES = Math.max(4, rasterCount * 4);
  const STAGE_BYTES = R_OFFSET + R_BYTES;
  const staging = [];
  for (let i = 0; i < STAGING_BUFFERS; i++) {
    staging.push({
      buf: device.createBuffer({
        size: STAGE_BYTES,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        label: 'staging' + i
      }),
      busy: false
    });
  }

  const zeroCounters = new Uint32Array(8);
  const paramsBuf = new ArrayBuffer(64);
  const pU32 = new Uint32Array(paramsBuf);
  const pF32 = new Float32Array(paramsBuf);
  let drive = 1.0;
  function writeParams() {
    pU32[0] = N;
    pU32[1] = REFRACTORY_STEPS;
    pU32[2] = 0;
    pU32[3] = 0;
    pF32[4] = V_REST_MV;
    pF32[5] = V_TH_MV;
    pF32[6] = V_RESET_MV;
    pF32[7] = DT_MS;
    pF32[8] = V_LEAK;
    pF32[9] = G_DECAY;
    pF32[10] = WEIGHT_FIXED_SCALE;
    pF32[11] = BACKGROUND_RATE_HZ * drive * DT_SEC;
    pF32[12] = EXT_KICK_MV;
    pF32[13] = DT_SEC;
    pF32[14] = 0;
    pF32[15] = 0;
    device.queue.writeBuffer(bufParams, 0, paramsBuf);
  }
  writeParams();

  const stimCPU = new Float32Array(N);
  let stimDirty = false;
  let lost = false;
  let lostReason = '';
  device.lost.then((info) => {
    lost = true;
    lostReason = (info && info.message) || 'device lost';
    console.error('[neuro_sim] WebGPU device lost: ' + lostReason);
  });
  device.onuncapturederror = (ev) => {
    console.error('[neuro_sim] WebGPU uncaptured error:', ev.error);
  };

  const neuronWorkgroups = Math.ceil(N / 256);
  const sampleWorkgroups = Math.ceil(Math.max(1, displayCount, rasterCount) / 256);

  return {
    backend: 'webgpu',
    chunkCount: chunks.length,
    isLost: () => lost,
    lostReason: () => lostReason,
    setDrive(d) { drive = d; writeParams(); },
    setStim(indices, rateHz) {
      for (let k = 0; k < indices.length; k++) stimCPU[indices[k]] = rateHz;
      stimDirty = true;
    },
    run(steps, onResult) {
      if (lost) return;
      if (stimDirty) {
        device.queue.writeBuffer(bufStim, 0, stimCPU);
        stimDirty = false;
      }
      device.queue.writeBuffer(bufCounters, 0, zeroCounters);

      const enc = device.createCommandEncoder();
      const p0 = enc.beginComputePass();
      p0.setPipeline(pipeSample);
      p0.setBindGroup(0, bgSampleClear);
      p0.dispatchWorkgroups(sampleWorkgroups);
      p0.end();

      for (let s = 0; s < steps; s++) {
        const p1 = enc.beginComputePass();
        p1.setPipeline(pipeIntegrate);
        p1.setBindGroup(0, bgIntegrate);
        p1.dispatchWorkgroups(neuronWorkgroups);
        p1.end();

        for (let c = 0; c < chunks.length; c++) {
          const ch = chunks[c];
          if (ch.workgroups === 0) continue;
          const p2 = enc.beginComputePass();
          p2.setPipeline(pipePropagate);
          p2.setBindGroup(0, ch.bindGroup);
          p2.dispatchWorkgroups(ch.workgroups);
          p2.end();
        }

        const p3 = enc.beginComputePass();
        p3.setPipeline(pipeSample);
        p3.setBindGroup(0, bgSampleRun);
        p3.dispatchWorkgroups(sampleWorkgroups);
        p3.end();
      }

      let slot = null;
      for (let i = 0; i < staging.length; i++) {
        if (!staging[i].busy) { slot = staging[i]; break; }
      }
      if (slot) {
        slot.busy = true;
        enc.copyBufferToBuffer(bufCounters, 0, slot.buf, 0, COUNTER_BYTES);
        enc.copyBufferToBuffer(bufDOut, 0, slot.buf, D_OFFSET, D_BYTES);
        enc.copyBufferToBuffer(bufROut, 0, slot.buf, R_OFFSET, R_BYTES);
      }
      device.queue.submit([enc.finish()]);

      if (slot) {
        slot.buf.mapAsync(GPUMapMode.READ).then(() => {
          const ab = slot.buf.getMappedRange();
          const counters = new Uint32Array(ab.slice(0, COUNTER_BYTES));
          const dRaw = new Uint32Array(ab.slice(D_OFFSET, D_OFFSET + D_BYTES));
          const rRaw = new Uint32Array(ab.slice(R_OFFSET, R_OFFSET + R_BYTES));
          slot.buf.unmap();
          slot.busy = false;
          const display = new Uint8Array(displayCount);
          for (let k = 0; k < displayCount; k++) display[k] = dRaw[k] ? 1 : 0;
          const raster = new Uint8Array(rasterCount);
          for (let k = 0; k < rasterCount; k++) raster[k] = rRaw[k] ? 1 : 0;
          onResult({
            totalSpikes: counters[0],
            groupSpikes: [counters[1], counters[2], counters[3], counters[4], counters[5]],
            display: display,
            raster: raster,
            windowMs: steps * DT_MS
          });
        }).catch((err) => {
          slot.busy = false;
          if (!lost) console.warn('[neuro_sim] activity readback failed', err);
        });
      }
    },
    dispose() {
      for (let i = 0; i < created.length; i++) {
        try { created[i].destroy(); } catch (e) { /* ignore */ }
      }
      for (let i = 0; i < staging.length; i++) {
        try { staging[i].buf.destroy(); } catch (e) { /* ignore */ }
      }
      try { device.destroy(); } catch (e) { /* ignore */ }
    }
  };
}

/* =========================================================================
 * 9. CPU WORKER BACKEND
 * ========================================================================= */

function createCpuWorkerBackend(ctx, onResult) {
  if (typeof Worker !== 'function' || typeof Blob !== 'function' ||
      typeof URL === 'undefined' || !URL.createObjectURL) {
    return null;
  }
  const blob = new Blob([cpuWorkerSource()], { type: 'text/javascript' });
  const url = URL.createObjectURL(blob);
  let worker;
  try {
    worker = new Worker(url);
  } catch (err) {
    URL.revokeObjectURL(url);
    console.warn('[neuro_sim] cannot start CPU worker', err);
    return null;
  }

  let drive = 1.0;
  worker.onmessage = (ev) => {
    const m = ev.data;
    if (m && m.type === 'summary') {
      onResult({
        totalSpikes: m.totalSpikes,
        groupSpikes: m.groupSpikes,
        display: m.display,
        raster: m.raster,
        windowMs: m.windowMs
      });
    }
  };
  worker.onerror = (err) => {
    console.error('[neuro_sim] CPU worker error', err.message || err);
  };

  worker.postMessage({
    type: 'init',
    neuronCount: ctx.neuronCount,
    offsets: ctx.offsets,
    targets: ctx.targets,
    weights: ctx.weights,
    meta: ctx.groupMask,
    displayIndices: ctx.displayIndices,
    rasterIndices: ctx.rasterIndices,
    params: {
      dtMs: DT_MS,
      dtSec: DT_SEC,
      vLeak: V_LEAK,
      vRest: V_REST_MV,
      vReset: V_RESET_MV,
      vTh: V_TH_MV,
      gDecay: G_DECAY,
      weightScale: WEIGHT_FIXED_SCALE,
      refrSteps: REFRACTORY_STEPS,
      pBase: BACKGROUND_RATE_HZ * DT_SEC,
      extKick: EXT_KICK_MV
    }
  }, [ctx.offsets.buffer, ctx.targets.buffer, ctx.weights.buffer]);

  return {
    backend: 'cpu-worker',
    chunkCount: 1,
    isLost: () => false,
    lostReason: () => '',
    setDrive(d) {
      drive = d;
      worker.postMessage({ type: 'drive', pBase: BACKGROUND_RATE_HZ * drive * DT_SEC });
    },
    setStim(indices, rateHz) {
      const copy = Int32Array.from(indices);
      worker.postMessage(rateHz > 0
        ? { type: 'stim', indices: copy, rateHz: rateHz }
        : { type: 'clearStim', indices: copy });
    },
    run() { /* the worker free-runs on its own timer */ },
    dispose() {
      try { worker.postMessage({ type: 'dispose' }); } catch (e) { /* ignore */ }
      try { worker.terminate(); } catch (e) { /* ignore */ }
      URL.revokeObjectURL(url);
    }
  };
}

/* =========================================================================
 * 10. PUBLIC FACTORY
 * ========================================================================= */

/**
 * Build a whole-connectome simulation.
 *
 * @param {Object} [opts]
 * @param {number[]|Int32Array} [opts.displayIndices=[]] neuron indices whose
 *        spike flags are read back each frame (capped at 20000).
 * @param {number} [opts.rasterCount=48] number of raster trace neurons.
 * @param {function(number,string)} [opts.onProgress] progress callback with a
 *        fraction in [0,1] and a human message.
 * @param {string} [opts.dataBase='/data'] backend mount point (extra option).
 * @param {'auto'|'webgpu'|'cpu-worker'} [opts.prefer='auto'] backend override
 *        (extra option, mainly for debugging).
 * @returns {Promise<Object>} the simulation handle.
 */
export async function createConnectomeSim({
  displayIndices = [],
  rasterCount = 48,
  onProgress = (fraction, message) => {},
  dataBase = DATA_BASE_DEFAULT,
  prefer = 'auto'
} = {}) {
  if (!hasDecompressionStream()) {
    throw new NeuroSimError(
      'DecompressionStream(gzip) is not available in this browser, so the ' +
      'connectome arrays cannot be decompressed client-side.');
  }

  const report = (f, m) => {
    try { onProgress(Math.max(0, Math.min(1, f)), m); } catch (e) { /* ignore */ }
  };

  report(0.0, 'Fetching manifest');
  const manifest = await loadManifest(dataBase);
  const neuronCount = manifest.neurons;
  const edgeCount = manifest.edges;
  const dataset = manifest.dataset || 'unknown-dataset';

  const aOffsets = manifestArray(manifest, 'offsets');
  const aSources = manifestArray(manifest, 'sources');
  const aCounts = manifestArray(manifest, 'counts');
  assert(aOffsets.length === neuronCount + 1,
    'offsets length ' + aOffsets.length + ' != neurons + 1 (' + (neuronCount + 1) + ')');
  assert(aSources.length === edgeCount, 'sources length != manifest edges');
  assert(aCounts.length === edgeCount, 'counts length != manifest edges');

  let totalCompressed = 0;
  for (let i = 0; i < manifest.arrays.length; i++) {
    const parts = manifest.arrays[i].parts;
    for (let j = 0; j < parts.length; j++) totalCompressed += (parts[j].bytes || 0);
  }
  if (totalCompressed <= 0) totalCompressed = 1;

  let seen = 0;
  const DOWNLOAD_SPAN = 0.72; // progress 0.05 .. 0.77
  const onBytes = (n) => {
    seen += n;
    report(0.05 + DOWNLOAD_SPAN * Math.min(1, seen / totalCompressed),
      'Downloading connectome: ' + (seen / 1048576).toFixed(1) + ' MB of about ' +
      (totalCompressed / 1048576).toFixed(1) + ' MB compressed');
  };

  report(0.03, 'Downloading neuron metadata');
  const meta = await loadMetadata(dataBase, manifest);
  if (meta.count !== neuronCount) {
    console.warn('[neuro_sim] metadata rows (' + meta.count + ') != manifest neurons (' +
      neuronCount + '); missing rows are treated as unknown with fastSign 0');
  }

  report(0.05, 'Downloading connectome arrays');
  const offsetsRes = await loadUint32Array(dataBase, aOffsets, onBytes);
  const offsets = offsetsRes.data;
  assert(offsets[neuronCount] === edgeCount,
    'offsets[last] = ' + offsets[neuronCount] + ' but the manifest declares ' +
    edgeCount + ' edges');

  const both = await Promise.all([
    loadUint32Array(dataBase, aSources, onBytes),
    loadUint32Array(dataBase, aCounts, onBytes)
  ]);
  const targets = both[0].data; // CSR is by source, so this array holds TARGETS
  const counts = both[1].data;
  for (let probe = 0; probe < Math.min(edgeCount, 1024); probe++) {
    assert(targets[probe] < neuronCount,
      'target index ' + targets[probe] + ' at edge ' + probe + ' is out of range');
  }

  report(0.80, 'Building synaptic weights');
  const built = buildWeights(offsets, counts, meta.fastSign, neuronCount, edgeCount);
  const weights = built.weights;
  const synapseCount = built.synapseCount;

  report(0.85, 'Deriving neuron groups');
  const groupInfo = buildGroups(meta, neuronCount);
  const stimTargets = buildStimTargets(meta, groupInfo, neuronCount);
  const emptyGroups = GROUP_NAMES.filter((n) => groupInfo.groups[n].size === 0);
  if (emptyGroups.length > 0) {
    console.warn('[neuro_sim] empty group(s): ' + emptyGroups.join(', ') +
      ' (reported with size 0, nothing substituted)');
  }

  const dIdx = sanitizeDisplayIndices(
    displayIndices instanceof Int32Array ? Array.from(displayIndices) : (displayIndices || []),
    neuronCount);
  const rCount = Math.max(0, rasterCount | 0);
  const rIdx = buildRasterIndices(groupInfo, neuronCount, rCount);

  const ctx = {
    neuronCount: neuronCount,
    edgeCount: edgeCount,
    offsets: offsets,
    targets: targets,
    weights: weights,
    groupMask: groupInfo.mask,
    displayIndices: dIdx,
    rasterIndices: rIdx
  };

  // Rolling activity window, shared by both backends.
  const groupSizes = {};
  for (let k = 0; k < GROUP_NAMES.length; k++) {
    groupSizes[GROUP_NAMES[k]] = groupInfo.groups[GROUP_NAMES[k]].size;
  }
  let simTimeMs = 0;
  let lastDisplay = new Uint8Array(dIdx.length);
  let lastRaster = new Uint8Array(rIdx.length);
  const windowEntries = [];
  let windowMs = 0;
  let windowTotal = 0;
  const windowGroups = new Float64Array(5);

  function ingest(result) {
    if (result.display && result.display.length === lastDisplay.length) lastDisplay = result.display;
    if (result.raster && result.raster.length === lastRaster.length) lastRaster = result.raster;
    windowEntries.push({ ms: result.windowMs, total: result.totalSpikes, groups: result.groupSpikes });
    windowMs += result.windowMs;
    windowTotal += result.totalSpikes;
    for (let b = 0; b < 5; b++) windowGroups[b] += result.groupSpikes[b];
    while (windowEntries.length > 1 && (windowMs - windowEntries[0].ms) >= RATE_WINDOW_MS) {
      const old = windowEntries.shift();
      windowMs -= old.ms;
      windowTotal -= old.total;
      for (let b = 0; b < 5; b++) windowGroups[b] -= old.groups[b];
    }
  }

  report(0.88, 'Initialising backend');
  let impl = null;
  if (prefer !== 'cpu-worker') {
    impl = await createWebGpuBackend(ctx, report);
  }
  if (!impl && prefer !== 'webgpu') {
    impl = createCpuWorkerBackend(ctx, ingest);
  }
  if (!impl) {
    throw new NeuroSimError(
      'No usable backend: WebGPU is unavailable or refused a device, and Web ' +
      'Workers from Blob URLs are not supported in this context.');
  }

  // The large arrays are no longer needed on the main thread.
  ctx.targets = null;
  ctx.weights = null;
  if (impl.backend === 'webgpu') ctx.offsets = null;

  report(1.0, 'Ready (' + impl.backend + ')');

  const activeStim = [];
  const warnedTargets = Object.create(null);
  let disposed = false;

  function expireStims() {
    for (let k = activeStim.length - 1; k >= 0; k--) {
      if (simTimeMs >= activeStim[k].expiresAtMs) {
        impl.setStim(activeStim[k].indices, 0);
        activeStim.splice(k, 1);
      }
    }
  }

  return {
    backend: impl.backend,
    neuronCount: neuronCount,
    edgeCount: edgeCount,
    synapseCount: synapseCount,
    dataset: dataset,
    groups: groupInfo.groups,

    // Extra, non-contractual introspection.
    superclassCounts: groupInfo.superclassCounts,
    edgeBufferChunks: impl.chunkCount,
    stimulationTargets: {
      screen: { size: stimTargets.screen.indices.length, source: stimTargets.screen.source },
      reward: { size: stimTargets.reward.indices.length, source: stimTargets.reward.source },
      startle: { size: stimTargets.startle.indices.length, source: stimTargets.startle.source }
    },
    modelNote:
      'Adapted leaky integrate-and-fire on real MaleCNS v1.0 wiring. ' +
      'Shiu et al. 2024 parameters (tau_m 20 ms, V_th -45 mV, V_rest/reset -52 mV, ' +
      'refractory 2.2 ms, 0.275 mV per synapse) plus a 5 ms synaptic decay, a ' +
      'one-step synaptic delay, and a Poisson background drive added here. ' +
      'This is not a validated emulation.',

    /** Advance the simulation for one rendered frame. Never blocks. */
    step(frameDtSeconds) {
      if (disposed) return;
      if (impl.isLost()) return;
      const dtSec = Number.isFinite(frameDtSeconds) ? frameDtSeconds : 0.016;
      const wantMs = Math.min(Math.max(dtSec, 0) * 1000, MAX_STEPS_PER_FRAME * DT_MS);
      const steps = Math.max(1, Math.min(MAX_STEPS_PER_FRAME, Math.round(wantMs / DT_MS)));
      if (impl.backend === 'webgpu') {
        simTimeMs += steps * DT_MS;
        expireStims();
        impl.run(steps, ingest);
      } else {
        // The worker owns its own clock; track wall time for stimulus expiry.
        simTimeMs += Math.max(0, dtSec) * 1000;
        expireStims();
      }
    },

    /** Latest completed activity summary. Synchronous and cheap. */
    getSummary() {
      const winSec = Math.max(windowMs, DT_MS) / 1000;
      const groupRates = {};
      for (let k = 0; k < GROUP_NAMES.length; k++) {
        const name = GROUP_NAMES[k];
        const size = groupSizes[name];
        groupRates[name] = size > 0 ? windowGroups[GROUP_BIT[name]] / (size * winSec) : 0;
      }
      return {
        simTimeMs: simTimeMs,
        spikesPerSecond: windowTotal / winSec,
        groupRates: groupRates,
        displaySpikes: lastDisplay,
        raster: lastRaster,
        rasterIndices: rIdx
      };
    },

    /**
     * Inject Poisson input into a named target.
     * @param {'screen'|'reward'|'startle'} target
     * @param {number} rateHz     Poisson event rate per neuron
     * @param {number} durationMs simulated duration of the stimulus
     */
    stimulate(target, rateHz, durationMs) {
      if (disposed) return;
      const t = stimTargets[target];
      if (!t) {
        console.warn('[neuro_sim] unknown stimulation target: ' + target);
        return;
      }
      if (t.indices.length === 0) {
        if (!warnedTargets[target]) {
          warnedTargets[target] = true;
          console.warn('[neuro_sim] stimulation target ' + target +
            ' is empty in this dataset; stimulate() does nothing');
        }
        return;
      }
      const rate = Math.max(0, Number(rateHz) || 0);
      const dur = Math.max(DT_MS, Number(durationMs) || 0);
      impl.setStim(t.indices, rate);
      activeStim.push({ indices: t.indices, expiresAtMs: simTimeMs + dur });
    },

    /** Global background-drive multiplier, clamped to [0, 2]. */
    setDrive(scalar) {
      if (disposed) return;
      impl.setDrive(Math.max(0, Math.min(2, Number(scalar) || 0)));
    },

    dispose() {
      if (disposed) return;
      disposed = true;
      impl.dispose();
    }
  };
}

export default createConnectomeSim;
