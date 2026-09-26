/**
 * fly_rig.js
 * ----------------------------------------------------------------------------
 * Loads and rigs the fruit-fly body model published by the Hugging Face Space
 * "Xenova/fruit-fly-simulation" (public/body/assets/model.json + STL meshes),
 * proxied and cached locally by this project's Python backend at:
 *   GET /assets/body/model.json
 *   GET /assets/body/meshes/<name>.stl
 *
 * model.json shape (as published upstream):
 *   root:        name of the root segment ("c_thorax")
 *   segments:    flat array of ~69 segment names
 *   joints:      array of [parentSegment, childSegment] pairs
 *   rest:        { [segment]: { pos: [x,y,z], quat: [w,x,y,z] } }  (local, parent-relative)
 *   meshes:      { [segment]: { file: "name.stl", mirror: boolean } }
 *   meshScale:   scalar (1000) dividing raw STL vertex units down to rig units
 *   axisOrder:   e.g. ["pitch","roll","yaw"] - the order joint-angle axes compose in
 *   axisVector:  { pitch:[x,y,z], roll:[x,y,z], yaw:[x,y,z] } - local rotation axes
 *   dofs:        array of { name, parent, child, axis, defaultDeg, rangeDeg, limitDeg, ... }
 *
 * This module does NOT ship any fallback/placeholder anatomy. If the source
 * data is missing, malformed, or a mesh fails to parse, loadFly() throws
 * rather than silently drawing a dummy body. There are no independent
 * accessories in this file either: the rig is anatomy only.
 *
 * Coordinate conversion (+Z-up source -> +Y-up three.js):
 *   The entire rig is assembled in the source's native +Z-up convention
 *   (rest positions/quaternions are used exactly as published, including the
 *   root's rest position with z = 1.3). Conversion to three.js's +Y-up space
 *   happens exactly ONCE, as a single -90 degree rotation about X applied to
 *   the top-level group that contains the root joint. Nothing else in this
 *   file adds a second vertical offset or a second axis swap, so the root's
 *   z = 1.3 height is never double-counted. The simplified ragdoll below
 *   follows the same rule: its root sink/tilt targets already bake in the
 *   ragdoll blend weight, so they are applied once, not multiplied again.
 *
 * No primitive spheres are used for anatomy. Every anatomical mesh comes from
 * a real STL parsed into a non-indexed BufferGeometry (no vertex welding),
 * with normals recomputed after parsing (and after mirroring, so winding is
 * fixed first).
 *
 * Returned rig API:
 *   group                                       - THREE.Group, add this to your scene
 *   update(timeSeconds, walkingStrength, mood, options)
 *     - backward compatible with the original 3-argument call.
 *     - options: {
 *         pose: 'walk' | 'typing' | 'collapsed',   // default 'walk'
 *         typingRate: number,                      // keystrokes/sec, 0..12
 *         fatigue: number,                         // 0..1
 *         dt: number,                               // seconds; clamped to <= 1/30
 *       }
 *   parts                                        - { [segmentName]: THREE.Object3D }
 *   setRagdoll(weight)                           - 0 (fully animated) .. 1 (fully physics-driven)
 *   applyImpulse(worldDirection: THREE.Vector3, strength: number) - poke reaction
 *   isRagdolling()                               - true while ragdoll weight > 0
 *   floorY                                       - world-space floor height (get/set), default 0
 *
 * Ragdoll disclaimer: this is a deliberately simplified, single-file
 * approximation - independent per-DOF angular spring-dampers plus a small
 * rigid-body spring for the root's pitch/roll/sink, with a crude leg-tip
 * floor projection. It is NOT a real rigid-body/contact physics engine and
 * makes no such claim; it exists to give a believable "gone limp" look
 * without pulling in an external physics library.
 */

// ---------------------------------------------------------------------------
// Small utilities
// ---------------------------------------------------------------------------

function fail(message) {
  throw new Error(`fly_rig: ${message}`);
}

async function fetchJSON(url) {
  let response;
  try {
    response = await fetch(url);
  } catch (err) {
    fail(`network error fetching ${url}: ${err.message}`);
  }
  if (!response.ok) {
    fail(`failed to fetch ${url} (HTTP ${response.status})`);
  }
  try {
    return await response.json();
  } catch (err) {
    fail(`${url} did not return valid JSON: ${err.message}`);
  }
}

async function fetchArrayBuffer(url) {
  let response;
  try {
    response = await fetch(url);
  } catch (err) {
    fail(`network error fetching ${url}: ${err.message}`);
  }
  if (!response.ok) {
    fail(`failed to fetch ${url} (HTTP ${response.status})`);
  }
  return await response.arrayBuffer();
}

function hashOffset(name) {
  let h = 0;
  for (let i = 0; i < name.length; i++) {
    h = (h * 31 + name.charCodeAt(i)) >>> 0;
  }
  return ((h % 1000) / 1000) * Math.PI * 2;
}

function moodScalar(mood) {
  if (typeof mood === 'number' && Number.isFinite(mood)) {
    return Math.max(0, mood);
  }
  const table = {
    calm: 0.6,
    neutral: 1.0,
    curious: 1.2,
    excited: 1.5,
    agitated: 1.8,
    aggressive: 1.8,
  };
  if (typeof mood === 'string' && table[mood.toLowerCase()] !== undefined) {
    return table[mood.toLowerCase()];
  }
  return 1.0;
}

// ---------------------------------------------------------------------------
// model.json validation
// ---------------------------------------------------------------------------

function validateModel(model) {
  const required = ['root', 'segments', 'joints', 'rest', 'meshes', 'meshScale', 'axisOrder', 'axisVector', 'dofs'];
  for (const key of required) {
    if (!(key in model)) fail(`model.json is missing required field "${key}"`);
  }
  if (!Array.isArray(model.segments) || model.segments.length === 0) {
    fail('model.json "segments" must be a non-empty array');
  }
  if (!Array.isArray(model.joints)) {
    fail('model.json "joints" must be an array');
  }
  if (!model.segments.includes(model.root)) {
    fail(`root "${model.root}" is not present in segments`);
  }
  if (!Array.isArray(model.dofs)) {
    fail('model.json "dofs" must be an array');
  }
  if (!Array.isArray(model.axisOrder) || model.axisOrder.length === 0) {
    fail('model.json "axisOrder" must be a non-empty array');
  }
  if (!model.axisVector || typeof model.axisVector !== 'object') {
    fail('model.json "axisVector" must be an object');
  }
  if (typeof model.meshScale !== 'number' || !(model.meshScale > 0)) {
    fail('model.json "meshScale" must be a positive number');
  }
  if (!model.meshes || typeof model.meshes !== 'object' || Object.keys(model.meshes).length === 0) {
    fail('model.json "meshes" is missing or empty; refusing to build a dummy body');
  }
}

// ---------------------------------------------------------------------------
// STL parsing (binary + ASCII), no welding
// ---------------------------------------------------------------------------

function parseBinarySTL(view, triCount) {
  if (triCount === 0) fail('binary STL contains zero triangles');
  const positions = new Float32Array(triCount * 9);
  let offset = 84;
  for (let i = 0; i < triCount; i++) {
    offset += 12; // skip facet normal (recomputed later)
    for (let v = 0; v < 3; v++) {
      const base = i * 9 + v * 3;
      positions[base] = view.getFloat32(offset, true);
      positions[base + 1] = view.getFloat32(offset + 4, true);
      positions[base + 2] = view.getFloat32(offset + 8, true);
      offset += 12;
    }
    offset += 2; // skip attribute byte count
  }
  return positions;
}

function parseAsciiSTL(text) {
  const vertexRe = /vertex\s+([-\deE.+]+)\s+([-\deE.+]+)\s+([-\deE.+]+)/g;
  const values = [];
  let match;
  while ((match = vertexRe.exec(text)) !== null) {
    values.push(parseFloat(match[1]), parseFloat(match[2]), parseFloat(match[3]));
  }
  if (values.length === 0 || values.length % 9 !== 0) {
    fail('ASCII STL vertex data is malformed or empty');
  }
  return new Float32Array(values);
}

function parseSTL(buffer, sourceLabel) {
  if (buffer.byteLength >= 84) {
    const view = new DataView(buffer);
    const triCount = view.getUint32(80, true);
    if (triCount > 0 && 84 + triCount * 50 === buffer.byteLength) {
      return parseBinarySTL(view, triCount);
    }
  }
  const text = new TextDecoder('utf-8').decode(buffer);
  if (/facet\s+normal/i.test(text)) {
    return parseAsciiSTL(text);
  }
  fail(`${sourceLabel}: unrecognized STL data (neither valid binary nor ASCII facets found)`);
}

// Converts raw STL vertex positions into a non-indexed BufferGeometry.
// Scales by 1/meshScale, mirrors across Y when requested (fixing the
// resulting winding-order flip), and recomputes vertex normals from the
// unwelded triangle soup (intentionally no index buffer, so normals stay
// faceted per triangle rather than smoothed across shared vertices).
function buildGeometry(THREE, rawPositions, meshScale, mirror) {
  const positions = new Float32Array(rawPositions.length);
  const inv = 1 / meshScale;
  for (let i = 0; i < rawPositions.length; i += 3) {
    positions[i] = rawPositions[i] * inv;
    positions[i + 1] = rawPositions[i + 1] * inv * (mirror ? -1 : 1);
    positions[i + 2] = rawPositions[i + 2] * inv;
  }
  if (mirror) {
    // Negating one axis inverts triangle winding; swap the trailing two
    // vertices of every triangle to restore outward-facing normals.
    for (let t = 0; t < positions.length; t += 9) {
      for (let c = 0; c < 3; c++) {
        const a = t + 3 + c;
        const b = t + 6 + c;
        const tmp = positions[a];
        positions[a] = positions[b];
        positions[b] = tmp;
      }
    }
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geometry.computeVertexNormals();
  geometry.computeBoundingSphere();
  return geometry;
}

// ---------------------------------------------------------------------------
// Materials
// ---------------------------------------------------------------------------

function createBodyMaterial(THREE) {
  return new THREE.MeshStandardMaterial({
    color: 0x2b2320,
    roughness: 0.55,
    metalness: 0.08,
    name: 'fly_body_material',
  });
}

function createEyeMaterial(THREE) {
  const mat = new THREE.MeshPhysicalMaterial({
    color: 0x7a0e12,
    roughness: 0.12,
    metalness: 0.05,
    clearcoat: 1.0,
    clearcoatRoughness: 0.05,
    name: 'fly_ruby_eye_material',
  });
  if ('transmission' in mat) {
    mat.transmission = 0.25;
    mat.thickness = 0.02;
    mat.ior = 1.77; // approximate ruby index of refraction
  }
  return mat;
}

function createWingMaterial(THREE) {
  const mat = new THREE.MeshPhysicalMaterial({
    color: 0xdfe9f2,
    roughness: 0.25,
    metalness: 0.0,
    transparent: true,
    opacity: 0.22,
    side: THREE.DoubleSide,
    depthWrite: false,
    name: 'fly_wing_material',
  });
  if ('transmission' in mat) {
    mat.transmission = 0.55;
    mat.thickness = 0.01;
  }
  return mat;
}

function pickMaterial(segmentName, materials) {
  if (/eye/i.test(segmentName)) return materials.eye;
  if (/wing/i.test(segmentName)) return materials.wing;
  return materials.body;
}

// ---------------------------------------------------------------------------
// Leg / gait helpers
// ---------------------------------------------------------------------------

// Segment naming convention (NeuroMechFly-style): <side><position>_<part>
// side: l/r, position: f (front) / m (mid) / h (hind).
function legInfoFromChild(name) {
  const m = /^(l|r)(f|m|h)_/.exec(name);
  if (!m) return null;
  const key = m[1] + m[2];
  // Canonical alternating tripod gait groups.
  const groupA = new Set(['lf', 'rm', 'rh']);
  return { key, phase: groupA.has(key) ? 0 : Math.PI };
}

function findLegTip(prefix, model) {
  const candidates = model.segments.filter((s) => s.startsWith(prefix));
  if (candidates.length === 0) return null;
  const tarsus = candidates.filter((s) => /tarsus/i.test(s));
  const pool = tarsus.length ? tarsus : candidates;
  pool.sort((a, b) => {
    const na = parseInt((a.match(/(\d+)$/) || [0, '0'])[1], 10);
    const nb = parseInt((b.match(/(\d+)$/) || [0, '0'])[1], 10);
    return na - nb;
  });
  return pool[pool.length - 1];
}

// Computes a joint-angle delta (degrees) for a single DOF at time t, for the
// default 'walk' pose. Defaults to the DOF's rest/default angle; procedural
// drivers (tripod gait, antenna flick) layer motion on top of that default.
function driveDof(dof, t, walk, moodK) {
  const base = dof.defaultDeg || 0;
  const child = dof.child;

  if (/antenna/i.test(child)) {
    const off = hashOffset(child);
    if (dof.axis === 'pitch') {
      return base + 5 * moodK * Math.sin(t * 3.1 + off) * Math.sin(t * 7.7 + off * 1.3);
    }
    if (dof.axis === 'yaw') {
      return base + 4 * moodK * Math.sin(t * 2.3 + off * 0.7);
    }
    return base;
  }

  const leg = legInfoFromChild(child);
  if (leg && dof.axis === 'pitch') {
    const freq = 1.4 + walk * 3.0 * moodK;
    if (/trochanterfemur$/.test(child)) {
      return base + walk * 24 * Math.sin(2 * Math.PI * freq * t + leg.phase);
    }
    if (/tibia$/.test(child)) {
      return base + walk * 16 * Math.sin(2 * Math.PI * freq * t + leg.phase + Math.PI / 2);
    }
  }

  return base;
}

// Computes a joint-angle delta (degrees) for the 'typing' pose: the fly
// stands still and taps its front legs like fingers on a keyboard, while
// mid/hind legs hold a steady stance, the head nods, antennae twitch, and
// wings stay folded (no drive applied to them at all).
function driveTyping(dof, t, typingRate, fatigue, moodK) {
  const base = dof.defaultDeg || 0;
  const child = dof.child;

  if (/head/i.test(child) && dof.axis === 'pitch') {
    const droop = 15 * fatigue; // tired posture: head droops as fatigue rises
    const nod = 3 * Math.sin(2 * Math.PI * Math.max(0.5, typingRate) * t);
    return base + droop + nod;
  }

  if (/wing/i.test(child)) {
    return base; // wings folded: never driven while typing
  }

  if (/antenna/i.test(child)) {
    const off = hashOffset(child);
    const amp = dof.axis === 'pitch' ? 4 : dof.axis === 'yaw' ? 3 : 0;
    return base + amp * moodK * Math.sin(t * 5.0 + off) * Math.sin(t * 11.0 + off * 1.7);
  }

  const leg = legInfoFromChild(child);
  if (leg && dof.axis === 'pitch') {
    const amp = Math.max(0, 1 - fatigue); // tapping amplitude shrinks with fatigue
    const isFront = leg.key === 'lf' || leg.key === 'rf';
    if (isFront) {
      const rate = Math.max(0, typingRate);
      const tapPhase = leg.key === 'lf' ? 0 : Math.PI; // alternate hands
      const tapWave = rate > 0 ? Math.max(0, Math.sin(2 * Math.PI * rate * t + tapPhase)) : 0;
      if (/trochanterfemur$/.test(child)) return base - amp * 10 * tapWave; // reach forward/down
      if (/tibia$/.test(child)) return base + amp * 18 * tapWave; // tap down
      return base;
    }
    // Mid/hind legs: steady standing stance, no gait cycling while typing.
    if (/trochanterfemur$/.test(child)) return base + 4 * amp;
    return base;
  }

  return base;
}

function dofKeyOf(dof) {
  return `${dof.child}::${dof.axis}`;
}

function jointLimitsOf(dof) {
  if (Array.isArray(dof.limitDeg) && dof.limitDeg.length === 2) return dof.limitDeg;
  if (Array.isArray(dof.rangeDeg) && dof.rangeDeg.length === 2) return dof.rangeDeg;
  return [-90, 90]; // generic fallback clamp when the model doesn't specify one
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/**
 * Loads the fruit-fly body model and returns a rigged three.js group.
 *
 * @param {object} THREE - the three.js module/namespace (injected, not imported).
 * @param {(fraction:number)=>void} [onProgress] - optional load-progress callback (0..1).
 * @returns {Promise<{
 *   group:object,
 *   update:(timeSeconds:number, walkingStrength?:number, mood?:(string|number), options?:object)=>void,
 *   parts:Record<string,object>,
 *   setRagdoll:(weight:number)=>void,
 *   applyImpulse:(worldDirection:object, strength:number)=>void,
 *   isRagdolling:()=>boolean,
 *   floorY:number,
 * }>}
 */
export async function loadFly(THREE, onProgress) {
  if (!THREE || !THREE.Object3D || !THREE.BufferGeometry) {
    fail('loadFly requires a THREE namespace/module as its first argument');
  }
  const report = typeof onProgress === 'function' ? onProgress : () => {};

  report(0);
  const model = await fetchJSON('/assets/body/model.json');
  validateModel(model);
  report(0.05);

  // --- Joint hierarchy from rest pose + joints, in the source's native
  //     +Z-up space. No axis conversion happens here.
  const nodes = new Map();
  for (const name of model.segments) {
    const rest = model.rest[name];
    if (!rest || !Array.isArray(rest.pos) || rest.pos.length !== 3 || !Array.isArray(rest.quat) || rest.quat.length !== 4) {
      fail(`segment "${name}" has no usable rest pose in model.json`);
    }
    const node = new THREE.Object3D();
    node.name = name;
    node.position.set(rest.pos[0], rest.pos[1], rest.pos[2]);
    node.quaternion.set(rest.quat[1], rest.quat[2], rest.quat[3], rest.quat[0]); // [w,x,y,z] -> (x,y,z,w)
    node.userData.restQuaternion = node.quaternion.clone();
    node.userData.restPosition = node.position.clone();
    nodes.set(name, node);
  }

  const parentOf = new Map();
  for (const joint of model.joints) {
    if (!Array.isArray(joint) || joint.length !== 2) {
      fail('model.json "joints" entries must be [parent, child] pairs');
    }
    const [parent, child] = joint;
    if (!nodes.has(parent) || !nodes.has(child)) {
      fail(`joint references unknown segment(s) [${parent}, ${child}]`);
    }
    parentOf.set(child, parent);
  }

  for (const name of model.segments) {
    if (name === model.root) continue;
    const parentName = parentOf.get(name);
    if (!parentName) fail(`segment "${name}" has no joint connecting it to a parent`);
    nodes.get(parentName).add(nodes.get(name));
  }

  const rootNode = nodes.get(model.root);

  // Single, one-time coordinate conversion: +Z-up (source) -> +Y-up (three.js).
  // The root's rest position (including z = 1.3) is used as-is above and
  // converted only by this rotation - it is never separately offset again.
  const group = new THREE.Group();
  group.name = 'fly';
  group.rotation.x = -Math.PI / 2;
  group.add(rootNode);

  report(0.1);

  // --- Materials -----------------------------------------------------------
  const materials = {
    body: createBodyMaterial(THREE),
    eye: createEyeMaterial(THREE),
    wing: createWingMaterial(THREE),
  };

  // --- Meshes: fetch + parse STL, mirror/wind-fix, attach ------------------
  const meshEntries = Object.entries(model.meshes);
  let loaded = 0;
  for (const [segmentName, meshInfo] of meshEntries) {
    const node = nodes.get(segmentName);
    if (!node) fail(`mesh "${segmentName}" has no matching segment/joint node`);
    if (!meshInfo || typeof meshInfo.file !== 'string' || !meshInfo.file) {
      fail(`mesh entry for "${segmentName}" is missing a file name`);
    }

    const url = `/assets/body/meshes/${meshInfo.file}`;
    const buffer = await fetchArrayBuffer(url);
    const rawPositions = parseSTL(buffer, url);
    const geometry = buildGeometry(THREE, rawPositions, model.meshScale, !!meshInfo.mirror);

    const mesh = new THREE.Mesh(geometry, pickMaterial(segmentName, materials));
    mesh.name = `${segmentName}_mesh`;
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    node.add(mesh);
    node.userData.mesh = mesh;

    loaded += 1;
    report(0.1 + 0.8 * (loaded / meshEntries.length));
  }

  // --- Degrees of freedom, grouped by the joint (child segment) they drive -
  const dofsByChild = new Map();
  for (const dof of model.dofs) {
    if (!dof || typeof dof.child !== 'string' || typeof dof.axis !== 'string') {
      fail('model.json "dofs" entries must include child and axis');
    }
    if (!nodes.has(dof.child)) {
      fail(`dof "${dof.name || ''}" targets unknown segment "${dof.child}"`);
    }
    if (!model.axisVector[dof.axis]) {
      fail(`dof "${dof.name || ''}" uses undefined axis "${dof.axis}"`);
    }
    if (!dofsByChild.has(dof.child)) dofsByChild.set(dof.child, []);
    dofsByChild.get(dof.child).push(dof);
  }

  // --- Simplified ragdoll state --------------------------------------------
  // Only leg joints and head/abdomen joints get a physics state; everything
  // else (eyes, wings, thorax internals) stays fully animated regardless of
  // ragdoll weight, since they have no meaningful "go limp" behaviour.
  const physicsDofIndex = new Map(); // dofKey -> dof
  for (const dof of model.dofs) {
    const isLeg = !!legInfoFromChild(dof.child);
    const isHead = /head/i.test(dof.child);
    const isAbdomen = /abdomen/i.test(dof.child);
    if (isLeg || isHead || isAbdomen) {
      physicsDofIndex.set(dofKeyOf(dof), dof);
    }
  }
  const physState = new Map(); // dofKey -> { angleDeg, angularVel }
  for (const [key, dof] of physicsDofIndex) {
    physState.set(key, { angleDeg: dof.defaultDeg || 0, angularVel: 0 });
  }

  // Leg tips used for the floor-contact approximation: last tarsus segment
  // per leg, paired with that leg's main swing DOF (coxa -> trochanterfemur
  // pitch) when one exists in the physics index.
  const legTips = [];
  for (const prefix of ['lf_', 'rf_', 'lm_', 'rm_', 'lh_', 'rh_']) {
    const tipName = findLegTip(prefix, model);
    if (!tipName || !nodes.has(tipName)) continue;
    const swingChildName = model.segments.find((s) => s.startsWith(prefix) && /trochanterfemur$/.test(s));
    const swingDof = swingChildName ? dofsByChild.get(swingChildName)?.find((d) => d.axis === 'pitch') : null;
    legTips.push({
      node: nodes.get(tipName),
      swingDof: swingDof || null,
      swingDofKey: swingDof ? dofKeyOf(swingDof) : null,
    });
  }

  // Simplified rigid-body state for the root (thorax): a small pitch/roll
  // spring plus a vertical "sink" spring, both driven toward targets that
  // scale with the current ragdoll weight (see stepRootPhysics below).
  const rootPhysics = { pitchDeg: 0, pitchVel: 0, rollDeg: 0, rollVel: 0, sinkZ: 0, sinkVel: 0, fallSign: 1 };

  const state = { ragdollWeight: 0, floorY: 0 };
  let lastUpdateT = null;
  const _floorProbe = new THREE.Vector3();

  report(1);

  function composeDelta(angles) {
    let q = new THREE.Quaternion();
    for (const axisName of model.axisOrder) {
      const deg = angles[axisName];
      if (!deg) continue;
      const vec = model.axisVector[axisName];
      const dq = new THREE.Quaternion().setFromAxisAngle(
        new THREE.Vector3(vec[0], vec[1], vec[2]).normalize(),
        THREE.MathUtils.degToRad(deg)
      );
      q = dq.multiply(q);
    }
    return q;
  }

  // Advances every physics-enabled joint by one semi-implicit Euler step:
  // velocity is integrated first from spring + damping + a crude gravity-like
  // torque, then position is integrated from the updated velocity. This is a
  // simplified per-DOF approximation, not a coupled rigid-body simulation.
  function stepPhysics(dt) {
    if (dt <= 0) return;
    const SPRING_K = 14;
    const DAMPING_C = 5;
    const GRAVITY_K = 22;
    for (const [key, dof] of physicsDofIndex) {
      const s = physState.get(key);
      const restAngle = dof.defaultDeg || 0;
      const rel = s.angleDeg - restAngle;
      const gravityTorque = GRAVITY_K * Math.sin(THREE.MathUtils.degToRad(rel));
      const springTorque = -SPRING_K * rel;
      const dampTorque = -DAMPING_C * s.angularVel;
      const accel = springTorque + dampTorque + gravityTorque;
      s.angularVel += accel * dt;
      s.angleDeg += s.angularVel * dt;
      const [lo, hi] = jointLimitsOf(dof);
      if (s.angleDeg < lo) {
        s.angleDeg = lo;
        s.angularVel *= -0.15;
      } else if (s.angleDeg > hi) {
        s.angleDeg = hi;
        s.angularVel *= -0.15;
      }
    }
  }

  // Crude floor contact: if a leg tip's world position sinks below floorY,
  // nudge that leg's main swing joint back toward its rest angle (a rough
  // position projection) and damp its velocity (approximate friction). This
  // is not a real contact solver; it only prevents the most obvious visual
  // floor penetration during the ragdoll blend.
  function resolveFloorContacts(dt) {
    if (dt <= 0) return;
    for (const tip of legTips) {
      if (!tip.swingDof) continue;
      tip.node.getWorldPosition(_floorProbe);
      const penetration = state.floorY - _floorProbe.y;
      if (penetration <= 0) continue;
      const s = physState.get(tip.swingDofKey);
      if (!s) continue;
      const restAngle = tip.swingDof.defaultDeg || 0;
      const dir = Math.sign(restAngle - s.angleDeg) || 1;
      const correction = Math.min(60, penetration * 400) * dt;
      s.angleDeg += dir * correction;
      s.angularVel *= 0.55; // friction: bleed velocity while in contact
    }
  }

  // Simplified rigid-body spring for the root: pitch/roll tilt over toward a
  // "fallen on its side" pose and sinks toward the floor as ragdoll weight
  // rises, and springs back upright as weight returns to 0. Targets already
  // scale by the current weight, so callers must not multiply by weight again.
  function stepRootPhysics(dt) {
    if (dt <= 0) return;
    const w = state.ragdollWeight;
    const targetPitch = 8 * w;
    const targetRoll = 70 * w * rootPhysics.fallSign;
    const targetSink = -0.9 * w;
    const K = 30;
    const C = 9;

    rootPhysics.pitchVel += (-K * (rootPhysics.pitchDeg - targetPitch) - C * rootPhysics.pitchVel) * dt;
    rootPhysics.pitchDeg += rootPhysics.pitchVel * dt;

    rootPhysics.rollVel += (-K * (rootPhysics.rollDeg - targetRoll) - C * rootPhysics.rollVel) * dt;
    rootPhysics.rollDeg += rootPhysics.rollVel * dt;

    rootPhysics.sinkVel += (-K * (rootPhysics.sinkZ - targetSink) - C * rootPhysics.sinkVel) * dt;
    rootPhysics.sinkZ += rootPhysics.sinkVel * dt;
  }

  /**
   * Advances the rig's pose. Safe to call every frame. Backward compatible:
   * calling update(t, walkingStrength, mood) with no options behaves exactly
   * as before.
   * @param {number} timeSeconds - monotonic animation clock, in seconds.
   * @param {number} [walkingStrength] - 0 (stationary) .. 1 (full tripod gait). Used in the 'walk' pose.
   * @param {string|number} [mood] - 'calm'|'neutral'|'curious'|'excited'|'agitated', or a numeric scalar.
   * @param {object} [options]
   * @param {('walk'|'typing'|'collapsed')} [options.pose] - defaults to 'walk'.
   * @param {number} [options.typingRate] - keystrokes/sec, 0..12, used in the 'typing' pose.
   * @param {number} [options.fatigue] - 0..1, used in the 'typing' pose.
   * @param {number} [options.dt] - explicit frame delta in seconds; clamped to <= 1/30.
   */
  function update(timeSeconds, walkingStrength, mood, options) {
    const opts = options || {};
    const t = typeof timeSeconds === 'number' && Number.isFinite(timeSeconds) ? timeSeconds : 0;

    let dt = typeof opts.dt === 'number' && Number.isFinite(opts.dt)
      ? opts.dt
      : (lastUpdateT === null ? 1 / 60 : t - lastUpdateT);
    dt = Math.max(0, Math.min(dt, 1 / 30)); // keep the simplified physics stable
    lastUpdateT = t;

    const walk = Math.min(1, Math.max(0, walkingStrength || 0));
    const moodK = moodScalar(mood !== undefined ? mood : 'neutral');
    const pose = opts.pose === 'typing' || opts.pose === 'collapsed' ? opts.pose : 'walk';
    const typingRate = Math.min(12, Math.max(0, opts.typingRate || 0));
    const fatigue = Math.min(1, Math.max(0, opts.fatigue || 0));

    stepPhysics(dt);
    resolveFloorContacts(dt);
    stepRootPhysics(dt);

    const weight = state.ragdollWeight;

    for (const [child, dofList] of dofsByChild) {
      const node = nodes.get(child);
      const angles = {};
      for (const dof of dofList) {
        let animated;
        if (pose === 'typing') animated = driveTyping(dof, t, typingRate, fatigue, moodK);
        else if (pose === 'collapsed') animated = dof.defaultDeg || 0;
        else animated = driveDof(dof, t, walk, moodK);

        const key = dofKeyOf(dof);
        const physicsState = physState.get(key);
        angles[dof.axis] = physicsState ? animated + (physicsState.angleDeg - animated) * weight : animated;
      }
      node.quaternion.copy(node.userData.restQuaternion).multiply(composeDelta(angles));
    }

    // Root: tired posture sinks the body while typing (fades out as the
    // ragdoll takes over), plus the ragdoll's own settle/tilt on top.
    const restPos = rootNode.userData.restPosition;
    const restQuat = rootNode.userData.restQuaternion;
    const tiredSinkZ = pose === 'typing' ? -0.05 * fatigue * (1 - weight) : 0;
    rootNode.position.set(restPos.x, restPos.y, restPos.z + tiredSinkZ + rootPhysics.sinkZ);

    const ragdollTilt = new THREE.Quaternion()
      .setFromAxisAngle(new THREE.Vector3(0, 1, 0), THREE.MathUtils.degToRad(rootPhysics.pitchDeg))
      .multiply(new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), THREE.MathUtils.degToRad(rootPhysics.rollDeg)));
    rootNode.quaternion.copy(restQuat).multiply(ragdollTilt);
  }

  /**
   * Blends the rig from fully animated (0) to fully physics-driven (1).
   * @param {number} weight
   */
  function setRagdoll(weight) {
    const w = Math.min(1, Math.max(0, typeof weight === 'number' && Number.isFinite(weight) ? weight : 0));
    if (state.ragdollWeight <= 0 && w > 0) {
      rootPhysics.fallSign = Math.random() < 0.5 ? -1 : 1;
    }
    state.ragdollWeight = w;
  }

  /**
   * Kicks the ragdoll's angular velocities, e.g. in response to the fly
   * being poked. Has no visible effect unless the ragdoll weight is > 0.
   * @param {object} worldDirection - THREE.Vector3-like {x,y,z} in world space.
   * @param {number} strength
   */
  function applyImpulse(worldDirection, strength) {
    if (!worldDirection || typeof strength !== 'number' || !Number.isFinite(strength)) return;
    const dir = new THREE.Vector3(worldDirection.x || 0, worldDirection.y || 0, worldDirection.z || 0);
    if (dir.lengthSq() === 0) return;
    dir.normalize();
    // Undo the single group-level +Z-up -> +Y-up rotation to bring the poke
    // direction back into the rig's native local space.
    dir.applyQuaternion(group.quaternion.clone().invert());

    const kick = Math.max(0, strength) * 40;
    for (const key of physicsDofIndex.keys()) {
      const s = physState.get(key);
      if (!s) continue;
      s.angularVel += kick * (Math.random() * 0.6 + 0.4) * (dir.z >= 0 ? 1 : -1);
    }
    rootPhysics.pitchVel += dir.z * Math.max(0, strength) * 25;
    rootPhysics.rollVel += -dir.x * Math.max(0, strength) * 25;
  }

  const parts = {};
  for (const [name, node] of nodes) parts[name] = node;

  return {
    group,
    update,
    parts,
    setRagdoll,
    applyImpulse,
    isRagdolling: () => state.ragdollWeight > 0,
    get floorY() {
      return state.floorY;
    },
    set floorY(value) {
      state.floorY = typeof value === 'number' && Number.isFinite(value) ? value : 0;
    },
  };
}
