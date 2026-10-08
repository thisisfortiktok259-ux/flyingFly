const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { test } = require('node:test');
const vm = require('node:vm');

const source = readFileSync('game.js', 'utf8');
function functionSource(name, end) {
  const start = source.indexOf(`  ${name === 'init' ? 'async ' : ''}function ${name}(`);
  return source.slice(start, source.indexOf(end, start));
}

test('sound off mutes the continuous ambient pad; on restores volume', () => {
  let click;
  const audio = { enabled: false, master: { gain: { value: 0.55 } } };
  const button = {
    addEventListener: (_, handler) => { click = handler; },
    setAttribute: (_, value) => { button.pressed = value; },
    querySelector: () => button.label,
    label: {},
  };
  const context = vm.createContext({
    audio, dom: { soundToggle: button },
    ensureAudioContext: () => ({ state: 'running' }),
    startAmbientPad() {}, logEvent() {}, showErrorBanner() {},
  });
  vm.runInContext(functionSource('setupSoundToggle', '\n  // -----') + '\nsetupSoundToggle();', context);
  click();
  assert.equal(button.pressed, 'true');
  assert.equal(audio.master.gain.value, 0.55);
  click();
  assert.equal(button.pressed, 'false');
  assert.equal(audio.master.gain.value, 0);
  assert.equal(button.label.textContent, 'Включить звук');
  click();
  assert.equal(audio.master.gain.value, 0.55);
});

test('animation starts before the connectome resolves and uses it when ready', async () => {
  let resolveSim;
  let frames = 0;
  let comments = 0;
  const pendingSim = new Promise(resolve => { resolveSim = resolve; });
  const context = vm.createContext({
    setup2dCanvasSizing() {}, buildEqualizerBars() {}, setupSoundToggle() {},
    createBrainScene() {}, createWorldScene() {}, buildBeetle() {}, setStatus() {},
    initFlyRig: async () => true,
    fetchNeurons: async () => ({ points: [{ index: 7 }] }),
    buildBrainPointCloud() {}, logEvent() {}, hideLoadingScreen() {},
    loadNeuroSim: () => pendingSim, SKIP_NEURO_SIM: false,
    neuroSim: null, neuroSimFailed: false, dom: {},
    startNewComment: () => { comments++; },
    frame() {}, requestAnimationFrame: () => { frames++; },
  });
  vm.runInContext(functionSource('init', '\n  init().catch'), context);
  const initialization = context.init();
  // Advance the rig and neuron promises, leaving the sim deliberately pending.
  for (let i = 0; i < 6; i++) await Promise.resolve();
  assert.equal(frames, 1, 'connectome download must not block the first frame');
  assert.equal(comments, 1);
  assert.equal(context.neuroSim, null);
  const sim = { backend: 'cpu' };
  resolveSim(sim);
  await initialization;
  await Promise.resolve();
  assert.equal(context.neuroSim, sim);
  assert.equal(frames, 1, 'simulation readiness must not start a second loop');
});
