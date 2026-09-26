# Flying Fly — гневный комментатор

A fruit fly sits at a tiny laptop and writes angry comments online, on its
own, until it gets tired. When it does, жук-надзиратель (the beetle
supervisor) shows up and forces it back to work. The fly's animated body and
its neural activity view are driven by a real published fly connectome, not
by scripted keyframes.

This README covers only the backend (`server.py`). Frontend files
(`index.html`, `style.css`, `game.js`, `fly_rig.js`, `neuro_sim.js`) are
described here only in terms of what the backend serves for them; their
contents are out of scope for this document and are not changed by it.

## What runs where

- **The backend (`server.py`)** is a small stdlib-only Python HTTP server.
  It serves the app shell, proxies and caches a few datasets from a public
  Hugging Face Space, and does nothing else. It does not simulate anything
  and does not call any third-party AI API.
- **The simulation runs entirely in the browser.** `neuro_sim.js` (owned by
  a separate, concurrent change, not by this backend) is expected to run a
  full 166,700-neuron leaky-integrate-and-fire (LIF) simulation using WebGPU
  where available, falling back to a CPU worker otherwise. The simulation is
  built from the real MaleCNS v1.0 connectivity (which neuron connects to
  which, and with how many synapses) fetched via the `/data/*` routes below.
  This is an adapted LIF model for an interactive visualization, not a
  validated scientific emulation of fly brain activity.
- **Leg movement** uses simplified ragdoll physics, not a biomechanical
  muscle model.

## Important disclaimers

- **The neuron simulation is a real-connectome-driven approximation, not a
  validated brain emulation.** It uses the actual MaleCNS synaptic wiring
  (who connects to whom, and roughly how strongly) to drive a simplified LIF
  model in real time in the browser. It does not claim to reproduce measured
  fly neural activity.
- **First run downloads a meaningful amount of data.** The full connectome
  (metadata plus the CSR-style offsets/sources/counts arrays, gzip-chunked
  into multiple part files) is on the order of ~80 MB compressed. The
  backend downloads and caches each file once, under `.cache/`, the first
  time it's requested; later runs reuse the cache.
- **The simulation can use a meaningful amount of GPU memory** (roughly
  ~250 MB) when running on WebGPU for a network this size. The CPU worker
  fallback exists for browsers/devices without usable WebGPU, and will be
  slower.
- **Leg and body motion is simplified ragdoll physics**, not an anatomically
  accurate biomechanical simulation.

## Data sources and credits

All datasets are mirrored from the public Hugging Face Space
**`Xenova/fruit-fly-simulation`** (`public/body/assets/` and `public/data/`):

- **MaleCNS v1.0** connectome: 166,700 neurons spanning brain and ventral
  nerve cord, plus ~25.6 million synaptic edges. Attribution: FlyEM / HHMI
  Janelia, University of Cambridge, MRC LMB, Google Research; released
  under **CC BY 4.0**.
- **NeuroMechFly** body meshes (used for the 3D fly body rig): released
  under **Apache-2.0**, mirrored via the same Space.

This project uses MaleCNS (not the smaller, brain-only FlyWire dataset)
because it is the larger, more complete connectome of the two and includes
the nerve cord.

## Requirements

- Python 3.9 or later. No third-party packages are required; the server uses
  only the Python standard library.
- Outbound internet access on first run, to fetch body assets and the
  connectome dataset from Hugging Face.
- A browser with WebGPU for the full-speed simulation; the CPU worker
  fallback is slower but does not require WebGPU.

## Local setup

1. (Optional) Copy the environment template if you want to override the
   port:

   ```bash
   cp .env.example .env
   ```

   `.env` only supports `PORT` (defaults to 8085 if unset or missing). There
   is no API key to configure; this backend does not call any third-party AI
   service.

2. Run the server:

   ```bash
   python3 server.py
   ```

   It listens on `http://0.0.0.0:8085` by default (override with `PORT`).

3. Open `http://localhost:8085/` in a browser. The server serves
   `index.html` at the app root (and at `/index.html`), plus `/style.css`,
   `/game.js`, `/fly_rig.js`, and `/neuro_sim.js` for the page to load.

4. (Optional) Run the tests:

   ```bash
   python3 -m unittest test_server.py -v
   ```

## Routes

### Frontend (fixed allowlist, not a generic static-file server)

| Route | File served | Content-Type |
|---|---|---|
| `GET /` | `index.html` | `text/html; charset=utf-8` |
| `GET /index.html` | `index.html` | `text/html; charset=utf-8` |
| `GET /style.css` | `style.css` | `text/css; charset=utf-8` |
| `GET /game.js` | `game.js` | `application/javascript; charset=utf-8` |
| `GET /fly_rig.js` | `fly_rig.js` | `application/javascript; charset=utf-8` |
| `GET /neuro_sim.js` | `neuro_sim.js` | `application/javascript; charset=utf-8` |

Each entry maps to exactly one fixed file in the project root; there is no
directory listing and no path is built from request input. A query string
(e.g. `/?v=123`) is accepted and ignored. `neuro_sim.js` is on this list even
before the file exists in the repo (it may be added by a separate, ongoing
change); until then, requesting it is an ordinary 404. Any other path,
including `.env`, `.env.example`, `server.py`, and `test_server.py`, is not
on this list and always returns a 404 JSON error; those files are never
served over HTTP. All non-GET methods (including `POST`, e.g. to any
previously-existing `/api/chat`-style route) also return a 404 JSON error:
this backend no longer has any POST routes.

### API and asset routes

#### `GET /api/health`

Liveness check. Returns `{"status": "ok", "time": <unix timestamp>}`.

#### `GET /assets/body/model.json`

Proxies and caches `public/body/assets/model.json` from the
`Xenova/fruit-fly-simulation` Hugging Face Space (the fly body rig: segments,
joints, rest pose, and per-segment mesh references).

#### `GET /assets/body/meshes/<filename>.stl`

Proxies and caches one mesh file from `public/body/assets/meshes/` on the
same Space. `<filename>` is strictly validated (letters, digits, underscore,
hyphen, single `.stl` extension only); any other value, including path
traversal attempts, is rejected with a 400 error before any file access.

#### `GET /api/neurons`

Returns a deterministic sample of about 16,000 real neuron soma locations out
of the full 166,700-neuron MaleCNS dataset (every Nth neuron, in dataset
order, so the same request always returns the same sample):

```json
{
  "dataset": "MaleCNS v1.0",
  "total": 166700,
  "source": "...attribution string...",
  "points": [
    {"x": 0.12, "y": -0.4, "z": 0.03, "index": 4821, "type": "...", "superclass": "...", "side": "...", "nt": "..."}
  ]
}
```

- `x`, `y`, `z` are the real soma coordinates, centered and uniformly scaled
  (same scale on all three axes, to preserve shape) so the whole sample fits
  roughly within `[-1, 1]` for easy rendering.
- `index` is the point's row position in the source neuron order (the same
  order used to build the connectome arrays under `/data/*`), so a
  client-side simulation can map computed per-neuron activity onto the
  displayed points.
- `type`, `superclass`, `side`, and `nt` (neurotransmitter) are included only
  when present in the source metadata for that neuron; they are never
  invented.
- No synthetic or placeholder points are ever generated. If the source data
  cannot be fetched or parsed, the endpoint returns a JSON error (502)
  instead of fabricated data.

#### `GET /data/manifest.json`

Proxies and caches the full-connectome manifest (`public/data/manifest.json`)
unchanged, as `application/json`. Its shape (from the live source) is
approximately:

```json
{
  "dataset": "MaleCNS v1.0",
  "neurons": 166700,
  "edges": 25582938,
  "metadata": "neurons.json.gz",
  "metadataColumns": ["bodyId", "...", "fastSign", "somaLocation8nm"],
  "arrays": [
    {"name": "offsets", "length": 166701, "parts": [{"file": "offsets-000.bin.gz", "bytes": 0, "sha256": "..."}]},
    {"name": "sources", "parts": [/* ~13 parts */]},
    {"name": "counts", "parts": [/* ~13 parts */]}
  ]
}
```

This is a CSR (compressed sparse row) representation of the connectome, keyed
by source neuron: for neuron `i`, `offsets[i]..offsets[i+1]` is the index
range of its outgoing edges; `sources` holds the target neuron id for each
edge in that range, and `counts` holds the synapse count for that edge. The
backend does not interpret or assume any dtype for these arrays; it only
proxies the declared files unchanged and lets the browser-side simulation
decide how to parse them.

#### `GET /data/<file>`

Proxies one connectome part file: either the neuron metadata file named by
the manifest's `metadata` field, or one of the part files named by
`arrays[*].parts[*].file`. Any other filename is a 404, and filenames are
additionally restricted to a strict allowlisted character set with no path
separators, so this can never be used to read an arbitrary file.

Bytes are served exactly as downloaded (still gzip-compressed) with
`Content-Type: application/octet-stream` and no `Content-Encoding` header;
decompression is the browser's job (via `DecompressionStream`), not this
server's. When the manifest declares a `sha256` for a part, a freshly
downloaded file is hashed and verified before it is written to the local
cache; on a mismatch the response is a 502 and the bad data is never cached.
Each download is also bounded by a size cap derived from the manifest's
declared byte size (with a small tolerance) or a generous default, and by an
absolute hard ceiling regardless of what the manifest claims, since the
manifest itself is untrusted upstream data.

## Caching and safety notes

- All Hugging Face assets are fetched once and cached under `.cache/` next to
  `server.py`. Delete that directory to force a re-download. `.cache/` is
  excluded from git via `.gitignore`.
- Mesh and data-part filenames, and all cache paths, are validated against a
  strict allowlist before touching the filesystem, to prevent directory
  traversal.
- The server exposes only the fixed routes listed above (six static frontend
  routes plus the API/asset routes). There is no generic static-file or
  directory-listing handler, so `.env`, `server.py`, `test_server.py`, and any
  other project file are never served over HTTP. There are no POST routes.
- The server uses a threaded HTTP server with a socket timeout and a fixed
  timeout on every outbound request to Hugging Face, so a slow or
  unresponsive upstream cannot hang the server indefinitely.
- **No downloaded data is trusted just because the request succeeded.**
  `manifest.json`, `neurons.json.gz`, and its decompressed contents are each
  size-capped before being parsed (decompression-bomb protection lives in
  `_safe_gunzip`), and each `/data/<file>` part is size-capped and, when the
  manifest supplies one, sha256-verified before it is ever written to the
  local cache.

## Tests and CI

`test_server.py` covers, without requiring live internet access:

- The static frontend routes and that `.env`, `server.py`, `test_server.py`,
  etc. are never served; `/neuro_sim.js` is tested for both the
  file-not-yet-present (plain 404) and file-present (200) cases.
- That every POST route, including the now-removed `/api/chat`, returns a
  plain 404 JSON error.
- `/data/manifest.json` and `/data/<file>` with Hugging Face fetches mocked:
  the metadata file and an array part being served correctly, an unlisted
  filename returning 404, a traversal attempt being rejected before any
  fetch is attempted, and a sha256 mismatch or oversized download returning
  502 without writing anything to the cache.
- `_build_neuron_sample()` with the Hugging Face fetch mocked: both a
  list-of-lists and a list-of-dicts row shape, that every point carries its
  source row `index`, that `superclass` is carried through when present,
  that rows without usable coordinates are skipped (never fabricated), and
  that malformed source data raises an explicit error.
- `_safe_gunzip()`: normal round-trip decompression, and that decompression
  is refused once it would exceed an explicit output cap.

Run locally with `python3 -m unittest test_server.py -v`.

A GitHub Actions workflow (`.github/workflows/test.yml`) runs this same test
suite on push/PR to `main`, plus a syntax-only check of `game.js`,
`fly_rig.js`, and `neuro_sim.js` with `node --check` (files that don't exist
yet, such as `neuro_sim.js` before a concurrent change adds it, are skipped
rather than failing the job; existing files are copied to a scratch `.mjs`
path for the check only, since they use ES module syntax without a
`package.json`, and are never executed or modified). This workflow has been
added/updated but its results should be checked in the repository's Actions
tab; nothing here should be read as a claim that a specific run has passed
unless that run was actually observed.

## Scope of this change

This backend change touches only `server.py`, `test_server.py`,
`.github/workflows/test.yml`, `.env.example`, and this README. The frontend
files (`index.html`, `style.css`, `game.js`, `fly_rig.js`, `neuro_sim.js`) are
served by the allowlisted routes above and syntax-checked (never executed) by
CI, but their contents are owned and changed by separate, concurrent work,
not by this change.
