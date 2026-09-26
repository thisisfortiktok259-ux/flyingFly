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
 *   dofs:        array of { name, parent, child, axis, defaultDeg, rangeDeg, ... }
 *
 * This module does NOT ship any fallback/placeholder anatomy. If the source
 * data is missing, malformed, or a mesh fails to parse, loadFly() throws
 * rather than silently drawing a dummy body.
 *
 * Coordinate conversion (+Z-up source -> +Y-up three.js):
 *   The entire rig is assembled in the source's native +Z-up convention
 *   (rest positions/quaternions are used exactly as published, including the
 *   root's rest position with z = 1.3). Conversion to three.js's +Y-up space
 *   happens exactly ONCE, as a single -90 degree rotation about X applied to
 *   the top-level group that contains the root joint. Nothing else in this
 *   file adds a second vertical offset or a second axis swap, so the root's
 *   z = 1.3 height is never double-counted.
 *
 * No primitive spheres are used for anatomy. Every anatomical mesh comes from
 * a real STL parsed into a non-indexed BufferGeometry (no vertex welding),
 * with normals recomputed after parsing (and after mirroring, so winding is
 * fixed first). The only primitive geometry in this file builds the
 * right-foreleg vape prop, which is an independent accessory, not anatomy.
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

// Computes a joint-angle delta (degrees) for a single DOF at time t.
// Defaults to the DOF's rest/default angle; procedural drivers (tripod gait,
// antenna flick) layer motion on top of that default.
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

// ---------------------------------------------------------------------------
// Right-foreleg vape accessory (independent prop, not anatomy - no spheres)
// ---------------------------------------------------------------------------

function buildVapeAccessory(THREE, tipNode) {
  const group = new THREE.Group();
  group.name = 'accessory_vape';

  const bodyMat = new THREE.MeshStandardMaterial({ color: 0x1c1c22, metalness: 0.6, roughness: 0.35 });
  const tipMat = new THREE.MeshStandardMaterial({
    color: 0x3a7bd5,
    metalness: 0.2,
    roughness: 0.2,
    emissive: 0x0a1a33,
    emissiveIntensity: 0.4,
  });

  const body = new THREE.Mesh(new THREE.CylinderGeometry(0.006, 0.007, 0.03, 12), bodyMat);
  body.rotation.z = Math.PI / 2;
  body.position.set(0.02, 0, 0);
  group.add(body);

  const mouth = new THREE.Mesh(new THREE.CylinderGeometry(0.0025, 0.004, 0.008, 10), tipMat);
  mouth.rotation.z = Math.PI / 2;
  mouth.position.set(0.038, 0, 0);
  group.add(mouth);

  const puffs = [];
  for (let i = 0; i < 3; i++) {
    const puffMat = new THREE.MeshStandardMaterial({
      color: 0xe8f0f8,
      transparent: true,
      opacity: 0,
      roughness: 1,
      metalness: 0,
    });
    const puff = new THREE.Mesh(new THREE.ConeGeometry(0.004 + i * 0.002, 0.01 + i * 0.006, 6, 1, true), puffMat);
    puff.position.set(0.045 + i * 0.006, 0, 0);
    puff.rotation.z = -Math.PI / 2;
    group.add(puff);
    puffs.push(puff);
  }

  group.position.set(0.01, -0.004, 0.002);
  tipNode.add(group);
  return { group, puffs };
}

function animateVape(vape, t, moodK) {
  const cycle = 4.0 / Math.max(0.3, moodK);
  const phase = (t % cycle) / cycle;
  vape.puffs.forEach((puff, i) => {
    const local = Math.min(1, Math.max(0, phase * 3 - i));
    const grow = Math.sin(Math.min(1, local) * Math.PI);
    puff.material.opacity = 0.35 * grow;
    const s = 0.6 + grow * 0.8;
    puff.scale.set(s, s, s);
  });
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/**
 * Loads the fruit-fly body model and returns a rigged three.js group.
 *
 * @param {object} THREE - the three.js module/namespace (injected, not imported).
 * @param {(fraction:number)=>void} [onProgress] - optional load-progress callback (0..1).
 * @returns {Promise<{group:object, update:(timeSeconds:number, walkingStrength?:number, mood?:(string|number))=>void, parts:Record<string,object>}>}
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

  // --- Degrees of freedom ---------------------------------------------------
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

  // --- Right-foreleg vape accessory (independent prop, optional) -----------
  const vapeTipName = findLegTip('rf_', model);
  let vape = null;
  if (vapeTipName && nodes.has(vapeTipName)) {
    vape = buildVapeAccessory(THREE, nodes.get(vapeTipName));
  } else {
    console.warn('fly_rig: could not locate a right-foreleg tip segment for the vape accessory; skipping it');
  }

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

  /**
   * Advances the rig's pose. Safe to call every frame.
   * @param {number} timeSeconds - monotonic animation clock, in seconds.
   * @param {number} [walkingStrength] - 0 (stationary) .. 1 (full tripod gait).
   * @param {string|number} [mood] - 'calm'|'neutral'|'curious'|'excited'|'agitated', or a numeric scalar.
   */
  function update(timeSeconds, walkingStrength, mood) {
    const t = typeof timeSeconds === 'number' && Number.isFinite(timeSeconds) ? timeSeconds : 0;
    const walk = Math.min(1, Math.max(0, walkingStrength || 0));
    const moodK = moodScalar(mood !== undefined ? mood : 'neutral');

    for (const [child, dofList] of dofsByChild) {
      const node = nodes.get(child);
      const angles = {};
      for (const dof of dofList) {
        angles[dof.axis] = driveDof(dof, t, walk, moodK);
      }
      node.quaternion.copy(node.userData.restQuaternion).multiply(composeDelta(angles));
    }

    if (vape) animateVape(vape, t, moodK);
  }

  const parts = {};
  for (const [name, node] of nodes) parts[name] = node;
  if (vape) parts.accessory_vape = vape.group;

  return { group, update, parts };
}
