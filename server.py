#!/usr/bin/env python3
"""Flying Fly backend server.

Pure Python 3 standard library HTTP server (no third-party dependencies).
Runs on port 8085 by default and exposes:

GET /
GET /index.html
GET /style.css
GET /game.js
GET /fly_rig.js
GET /neuro_sim.js
Serves the app shell and its frontend assets from a small, fixed
allowlist (each URL maps to exactly one file in the project root, with
the correct MIME type). This is not a generic static-file or
directory-listing handler: any path not on this list, and not one of
the API/asset routes below, is a 404. In particular, dotfiles such as
.env are never served, and the server's own source files (e.g.
server.py, test_server.py) are never served. /neuro_sim.js is served
the same way once that file exists in the project root; until then a
request for it is a plain 404 (this server does not require the file
to exist to start or to serve any other route).

GET /api/health
Liveness check.

GET /assets/body/model.json
GET /assets/body/meshes/<filename>.stl
Proxies and locally caches the fruit-fly body model assets published by
the Hugging Face Space "Xenova/fruit-fly-simulation"
(public/body/assets/model.json and public/body/assets/meshes/*.stl).
Files are fetched once and served from a local cache directory after
that. Mesh filenames are strictly validated to prevent directory
traversal or arbitrary upstream paths.

GET /api/neurons
Returns a deterministic ~16,000-point sample of REAL soma coordinates
from the same Hugging Face Space's public/data/manifest.json and
public/data/neurons.json.gz (MaleCNS v1.0 dataset, 166,700 neurons,
brain + nerve cord). Each point includes its row index in the source
neuron order ('index') so a client-side simulation driven by the full
connectome (see /data below) can map computed activity back onto the
displayed points. No synthetic or fabricated points are ever returned;
if the source data cannot be fetched or parsed, the endpoint returns an
explicit JSON error instead of a fallback. Downloaded and decompressed
data is size-capped (see MAX_* constants below) so a corrupted or
hostile upstream response cannot exhaust memory.

GET /data/manifest.json
GET /data/<file>
Proxies the full-connectome dataset (the same Space's public/data/):
the neuron metadata file and the binary connectivity arrays (CSR-style
offsets/sources/counts, chunked into multiple gzip'd part files) that a
browser-side simulation needs to run the real MaleCNS network. Every
<file> request is checked against an allowlist built from the live
manifest (the file must equal manifest['metadata'] or appear as some
manifest['arrays'][*]['parts'][*]['file']); anything else is a 404.
Filenames are further restricted to a strict character allowlist with
no path separators, so this can never read an arbitrary file. Bytes are
proxied through unchanged (the files stay gzip-compressed on disk and
on the wire; the server sets Content-Type: application/octet-stream
and no Content-Encoding, since decompression is the browser's job via
DecompressionStream, not this server's). When the manifest declares a
sha256 for a part, a freshly downloaded file is hashed and verified
before it is written to the local cache; a mismatch is a 502 and the
bad data is never cached. Every downloaded part is also bounded by a
size cap derived from the manifest's declared byte size (with a small
tolerance) or a generous default, and by an absolute hard ceiling
regardless of what the manifest claims.

Configuration (read from a local .env file or the real process environment,
process environment wins if both are set):

PORT Defaults to 8085.

The .env file itself, and any dotfile, is never served over HTTP by this
server: only the fixed set of routes above exists (a small static allowlist
plus the API/asset routes), there is no generic static-file or
directory-listing handler.

Note on scope: this backend does not run the neural simulation itself.
The full 166,700-neuron leaky-integrate-and-fire simulation runs entirely
in the browser (WebGPU with a CPU worker fallback); this server's only
job for that feature is to serve the real connectome bytes it's built
from, unchanged and integrity-checked.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / ".cache"
ENV_PATH = BASE_DIR / ".env"


def _load_dotenv(path: Path) -> dict:
    """Minimal .env parser: KEY=VALUE lines, '#' comments, no interpolation."""
    values: dict = {}
    if not path.exists():
        return values
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return values
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key:
            values[key] = val
    return values


_DOTENV = _load_dotenv(ENV_PATH)


def get_env(name: str, default: str = "") -> str:
    """Real process environment takes precedence over the local .env file."""
    if name in os.environ and os.environ[name] != "":
        return os.environ[name]
    return _DOTENV.get(name, default)


try:
    PORT = int(get_env("PORT", "8085") or "8085")
except ValueError:
    PORT = 8085

HF_SPACE_ROOT = (
    "https://huggingface.co/spaces/Xenova/fruit-fly-simulation/resolve/main"
)
HF_BODY_ASSETS = f"{HF_SPACE_ROOT}/public/body/assets"
HF_DATA = f"{HF_SPACE_ROOT}/public/data"
HF_ATTRIBUTION_SOURCE = (
    "Xenova/fruit-fly-simulation Hugging Face Space "
    "(public/data/manifest.json, public/data/neurons.json.gz); "
    "MaleCNS v1.0 connectome, FlyEM/HHMI Janelia, University of Cambridge, "
    "MRC LMB, Google Research; CC BY 4.0"
)

HTTP_TIMEOUT = 20  # seconds, applies to every outbound fetch to Hugging Face
NEURON_SAMPLE_SIZE = 16000

# Sanity caps on untrusted upstream data. These are deliberately generous for
# the real MaleCNS v1.0 dataset (manifest.json is a few KB, neurons.json.gz
# is a few MB compressed) but still bound worst-case memory/CPU use if the
# upstream ever returns something corrupted, truncated, or hostile. We never
# trust the content of a downloaded file just because the request succeeded.
MAX_MANIFEST_BYTES = 5 * 1024 * 1024  # 5 MiB
MAX_NEURON_GZ_BYTES = 50 * 1024 * 1024  # 50 MiB compressed
MAX_NEURON_DECOMPRESSED_BYTES = 200 * 1024 * 1024  # 200 MiB decompressed
MAX_NEURON_ROWS = 5_000_000  # sanity bound on the parsed row count

# Per-part size caps for the raw connectome arrays served under /data/<file>.
# When the manifest declares a byte size for a part we use that (plus a
# small tolerance for minor packaging differences) as the cap, but we never
# exceed the absolute hard ceiling below regardless of what the manifest
# claims -- the manifest itself is untrusted upstream data.
DEFAULT_DATA_PART_MAX_BYTES = 16 * 1024 * 1024  # 16 MiB
DATA_PART_HARD_CAP_BYTES = 32 * 1024 * 1024  # 32 MiB
DATA_PART_SIZE_TOLERANCE_BYTES = 4096

MESH_NAME_RE = re.compile(r"^[A-Za-z0-9_\-]{1,80}\.stl$")
DATA_FILENAME_RE = re.compile(r"^[A-Za-z0-9_.\-]{1,80}$")

DEFAULT_NEURON_METADATA_COLUMNS = [
    "bodyId",
    "type",
    "superclass",
    "side",
    "consensusNT",
    "fastSign",
    "somaLocation8nm",
]

# ---------------------------------------------------------------------------
# Static frontend allowlist.
#
# This is intentionally NOT a generic static-file server. Each entry maps one
# exact request path to one fixed file (by basename, in BASE_DIR) and its
# correct MIME type. Requests for anything else -- including .env, .env.*,
# server.py, test_server.py, or any path with ".." -- never match this table
# and fall through to a plain 404 from the router. Frontend file *contents*
# (index.html, style.css, game.js, fly_rig.js, neuro_sim.js) are out of scope
# for this server change and are not modified here; neuro_sim.js is listed
# even before it exists in the repo, since other files may land it later and
# this route simply 404s until the file is present.
# ---------------------------------------------------------------------------

STATIC_FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/style.css": ("style.css", "text/css; charset=utf-8"),
    "/game.js": ("game.js", "application/javascript; charset=utf-8"),
    "/fly_rig.js": ("fly_rig.js", "application/javascript; charset=utf-8"),
    "/neuro_sim.js": ("neuro_sim.js", "application/javascript; charset=utf-8"),
}

# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

_cache_locks_guard = threading.Lock()
_cache_locks: dict = {}


def _lock_for(key: str) -> threading.Lock:
    with _cache_locks_guard:
        lock = _cache_locks.get(key)
        if lock is None:
            lock = threading.Lock()
            _cache_locks[key] = lock
        return lock


def _http_get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "flyingFly-server/1.0"})
    with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
        return resp.read()


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp-{os.getpid()}-{threading.get_ident()}")
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def _fetch_cached(url: str, cache_path: Path) -> bytes:
    """Return cached bytes for url, downloading and caching on first use.

    Raises urllib.error.URLError / OSError on failure; callers must turn
    this into an explicit JSON error response, never a synthetic fallback.
    No integrity check is performed here -- use _fetch_verify_cache when the
    manifest supplies a sha256 to check against before caching.
    """
    if cache_path.exists():
        return cache_path.read_bytes()
    lock = _lock_for(str(cache_path))
    with lock:
        if cache_path.exists():
            return cache_path.read_bytes()
        data = _http_get(url)
        _atomic_write(cache_path, data)
        return data


class DataIntegrityError(Exception):
    """Raised when a downloaded /data part fails sha256 verification."""


def _fetch_verify_cache(
    url: str,
    cache_path: Path,
    expected_bytes,
    expected_sha256,
    max_bytes: int,
) -> bytes:
    """Like _fetch_cached, but for manifest-described connectome parts.

    Before writing anything to the cache: enforces max_bytes on the
    downloaded size, and if expected_sha256 is provided, verifies it and
    raises DataIntegrityError (without caching) on any mismatch. Already-
    cached files are trusted (they were verified when first written) and
    returned as-is without re-hashing on every request.
    """
    if cache_path.exists():
        return cache_path.read_bytes()
    lock = _lock_for(str(cache_path))
    with lock:
        if cache_path.exists():
            return cache_path.read_bytes()
        data = _http_get(url)
        if len(data) > max_bytes:
            raise ValueError(f"{url} exceeded {max_bytes} byte safety cap")
        if expected_sha256:
            digest = hashlib.sha256(data).hexdigest()
            if digest.lower() != expected_sha256.lower():
                raise DataIntegrityError(
                    f"sha256 mismatch for downloaded part (expected {expected_sha256})"
                )
        _atomic_write(cache_path, data)
        return data


def _size_cap_for_part(declared_bytes) -> int:
    """Pick a per-part download size cap from the manifest's declared size.

    Prefers the manifest's declared byte size (plus a small tolerance for
    minor packaging differences) when present, but never exceeds the
    absolute hard ceiling -- the manifest is untrusted upstream data, so a
    manifest that lies about a huge size cannot be used to bypass the cap.
    """
    if isinstance(declared_bytes, int) and declared_bytes > 0:
        return min(declared_bytes + DATA_PART_SIZE_TOLERANCE_BYTES, DATA_PART_HARD_CAP_BYTES)
    return DEFAULT_DATA_PART_MAX_BYTES


def _safe_gunzip(data: bytes, max_output_bytes: int) -> bytes:
    """Decompress gzip bytes with a hard cap on decompressed size.

    Protects against decompression bombs: a small compressed payload that
    would expand into an enormous buffer. Raises ValueError instead of
    materializing unbounded output if the cap would be exceeded. We never
    trust that a successfully-downloaded file is well-behaved just because
    the HTTP request succeeded.
    """
    decompressor = zlib.decompressobj(16 + zlib.MAX_WBITS)
    chunks = []
    total = 0
    chunk_size = 1 << 20  # 1 MiB input chunks

    def _consume(piece: bytes) -> None:
        nonlocal total
        if not piece:
            return
        total += len(piece)
        if total > max_output_bytes:
            raise ValueError(f"decompressed data exceeds {max_output_bytes} byte safety cap")
        chunks.append(piece)

    for start in range(0, len(data), chunk_size):
        remaining = max_output_bytes - total + 1
        _consume(decompressor.decompress(data[start : start + chunk_size], remaining))
    while decompressor.unconsumed_tail:
        remaining = max_output_bytes - total + 1
        piece = decompressor.decompress(decompressor.unconsumed_tail, remaining)
        if not piece:
            break
        _consume(piece)
    _consume(decompressor.flush())
    return b"".join(chunks)


# ---------------------------------------------------------------------------
# Shared manifest loading (used by /api/neurons and the /data/* proxy)
# ---------------------------------------------------------------------------


def _load_manifest() -> dict:
    """Fetch (or reuse the cache of) manifest.json and return it parsed.

    Raises urllib.error.URLError / OSError on fetch failure, and ValueError /
    json.JSONDecodeError if the manifest is oversized or not a JSON object.
    Never returns a fabricated manifest.
    """
    manifest_bytes = _fetch_cached(f"{HF_DATA}/manifest.json", CACHE_DIR / "data" / "manifest.json")
    if len(manifest_bytes) > MAX_MANIFEST_BYTES:
        raise ValueError(f"manifest.json exceeds {MAX_MANIFEST_BYTES} byte safety cap")
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("manifest.json is not a JSON object")
    return manifest


def _manifest_allowed_data_files(manifest: dict) -> dict:
    """Build the /data/<file> allowlist from a parsed manifest.

    Returns {filename: {"bytes": int|None, "sha256": str|None}} for the
    neuron metadata file (manifest["metadata"]) and every connectivity array
    part file (manifest["arrays"][*]["parts"][*]["file"]). Only files that
    actually appear in the live manifest are ever allowed; nothing here is
    hardcoded to a specific dataset size or part count.
    """
    allowed: dict = {}

    metadata_file = manifest.get("metadata")
    if isinstance(metadata_file, str) and metadata_file:
        allowed[metadata_file] = {"bytes": None, "sha256": None}

    arrays = manifest.get("arrays")
    if isinstance(arrays, list):
        for arr in arrays:
            if not isinstance(arr, dict):
                continue
            parts = arr.get("parts")
            if not isinstance(parts, list):
                continue
            for part in parts:
                if not isinstance(part, dict):
                    continue
                fname = part.get("file")
                if isinstance(fname, str) and fname:
                    declared_bytes = part.get("bytes")
                    declared_sha256 = part.get("sha256")
                    allowed[fname] = {
                        "bytes": declared_bytes if isinstance(declared_bytes, int) else None,
                        "sha256": declared_sha256 if isinstance(declared_sha256, str) else None,
                    }
    return allowed


# ---------------------------------------------------------------------------
# Neuron soma sample (MaleCNS v1.0, brain + nerve cord, 166,700 neurons)
# ---------------------------------------------------------------------------

_neuron_sample_guard = threading.Lock()
_neuron_sample_cache = None  # populated lazily, kept in memory once built


def _extract_xyz(value):
    """Best-effort extraction of (x, y, z) floats from a soma-location field.

    Returns None if the value cannot be interpreted as three coordinates.
    Never invents coordinates; unparsable rows are simply excluded.
    """
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        try:
            return float(value[0]), float(value[1]), float(value[2])
        except (TypeError, ValueError):
            return None
    if isinstance(value, dict):
        for keys in (("x", "y", "z"), ("X", "Y", "Z")):
            if all(k in value for k in keys):
                try:
                    return (
                        float(value[keys[0]]),
                        float(value[keys[1]]),
                        float(value[keys[2]]),
                    )
                except (TypeError, ValueError):
                    return None
        return None
    if isinstance(value, str):
        # NOTE: this character class must stay properly closed ("]" before
        # the "+" quantifier). An earlier revision had an unterminated class
        # here (missing the closing "]"), which raised re.error on every
        # plain-string soma-location value instead of returning None.
        parts = re.split(r"[\[\],\s]+", value.strip("[]() "))
        parts = [p for p in parts if p]
        if len(parts) >= 3:
            try:
                return float(parts[0]), float(parts[1]), float(parts[2])
            except ValueError:
                return None
        return None
    return None


def _row_to_fields(row, columns):
    """Map one neuron row (list/tuple or dict) to a column-name dict."""
    if isinstance(row, dict):
        return row
    if isinstance(row, (list, tuple)):
        return {columns[i]: row[i] for i in range(min(len(columns), len(row)))}
    return {}


def _build_neuron_sample():
    """Download (or reuse cache of) manifest + neuron metadata, then build a
    deterministic ~16k-point normalized sample of real soma coordinates.
    Raises on any failure; there is no synthetic-data fallback.

    Note: the published manifest documents metadataColumns (an ordered list
    of field names) but does not itself state whether each row in
    neurons.json.gz is a JSON array in that order or a JSON object keyed by
    those names -- the two descriptions of this found while researching the
    source were contradictory. This function handles both shapes (see
    _row_to_fields) and only emits a point when real x/y/z coordinates were
    actually parsed from the source; it never fabricates a point. All
    downloaded/decompressed data is validated and size-capped before use; a
    successful download is never assumed to be well-formed.

    Each point also carries its row index in the source neuron order
    ('index'), so a client-side simulation driven by the full connectome
    (built from the same row order via /data/*) can map computed per-neuron
    activity back onto the displayed points.
    """
    manifest = _load_manifest()

    metadata_file = manifest.get("metadata", "neurons.json.gz")
    if not isinstance(metadata_file, str) or not metadata_file:
        raise ValueError("manifest.json has no usable 'metadata' filename")
    columns = manifest.get("metadataColumns") or DEFAULT_NEURON_METADATA_COLUMNS
    if not isinstance(columns, list) or not all(isinstance(c, str) for c in columns):
        columns = DEFAULT_NEURON_METADATA_COLUMNS
    dataset_name = manifest.get("dataset", "MaleCNS v1.0")
    total_neurons = manifest.get("neurons")

    gz_bytes = _fetch_cached(
        f"{HF_DATA}/{metadata_file}",
        CACHE_DIR / "data" / metadata_file,
    )
    if len(gz_bytes) > MAX_NEURON_GZ_BYTES:
        raise ValueError(f"{metadata_file} exceeds {MAX_NEURON_GZ_BYTES} byte safety cap")
    raw = _safe_gunzip(gz_bytes, MAX_NEURON_DECOMPRESSED_BYTES)
    rows = json.loads(raw.decode("utf-8"))
    if not isinstance(rows, list):
        raise ValueError("neuron metadata is not a JSON array")
    if len(rows) > MAX_NEURON_ROWS:
        raise ValueError(f"neuron metadata exceeds {MAX_NEURON_ROWS} row safety cap")

    total_rows = len(rows)
    if not isinstance(total_neurons, int) or total_neurons <= 0:
        total_neurons = total_rows

    target = min(NEURON_SAMPLE_SIZE, total_rows)
    stride = max(1, total_rows // target) if target else 1

    soma_key = "somaLocation8nm" if "somaLocation8nm" in columns else columns[-1]

    candidates = []
    idx = 0
    while idx < total_rows and len(candidates) < target:
        fields = _row_to_fields(rows[idx], columns)
        xyz = _extract_xyz(fields.get(soma_key))
        if xyz is not None:
            candidates.append((xyz, fields, idx))
        idx += stride

    if not candidates:
        raise ValueError("no usable soma coordinates found in source data")

    xs = [c[0][0] for c in candidates]
    ys = [c[0][1] for c in candidates]
    zs = [c[0][2] for c in candidates]
    cx = (min(xs) + max(xs)) / 2.0
    cy = (min(ys) + max(ys)) / 2.0
    cz = (min(zs) + max(zs)) / 2.0
    half_range = max(
        (max(xs) - min(xs)) / 2.0,
        (max(ys) - min(ys)) / 2.0,
        (max(zs) - min(zs)) / 2.0,
        1e-9,
    )

    points = []
    for (x, y, z), fields, row_index in candidates:
        point = {
            "x": (x - cx) / half_range,
            "y": (y - cy) / half_range,
            "z": (z - cz) / half_range,
            "index": row_index,
        }
        if fields.get("type") is not None:
            point["type"] = fields.get("type")
        if fields.get("superclass") is not None:
            point["superclass"] = fields.get("superclass")
        if fields.get("side") is not None:
            point["side"] = fields.get("side")
        if fields.get("consensusNT") is not None:
            point["nt"] = fields.get("consensusNT")
        points.append(point)

    return {
        "dataset": dataset_name,
        "total": total_neurons,
        "source": HF_ATTRIBUTION_SOURCE,
        "points": points,
    }


def get_neuron_sample():
    global _neuron_sample_cache
    if _neuron_sample_cache is not None:
        return _neuron_sample_cache
    with _neuron_sample_guard:
        if _neuron_sample_cache is None:
            _neuron_sample_cache = _build_neuron_sample()
        return _neuron_sample_cache


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    server_version = "FlyingFlyServer/1.0"
    protocol_version = "HTTP/1.1"
    timeout = 30  # socket read timeout, seconds

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _send_error_json(self, status: int, message: str) -> None:
        self._send_json(status, {"error": message})

    def _send_bytes(self, status: int, content_type: str, data: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, fmt, *args):  # quieter default logging
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    # -- routing -------------------------------------------------------------
    def do_GET(self):
        path = self.path.split("?", 1)[0]
        try:
            if path in STATIC_FILES:
                return self._handle_static(*STATIC_FILES[path])
            if path == "/api/health":
                return self._handle_health()
            if path == "/api/neurons":
                return self._handle_neurons()
            if path == "/assets/body/model.json":
                return self._handle_body_model()
            if path.startswith("/assets/body/meshes/"):
                return self._handle_body_mesh(path[len("/assets/body/meshes/"):])
            if path == "/data/manifest.json":
                return self._handle_data_manifest()
            if path.startswith("/data/"):
                return self._handle_data_file(path[len("/data/"):])
            return self._send_error_json(404, "Not found.")
        except Exception as exc:  # last-resort guard, never leak internals
            self._send_error_json(500, f"Internal server error: {exc.__class__.__name__}")

    def do_POST(self):
        try:
            return self._send_error_json(404, "Not found.")
        except Exception as exc:
            self._send_error_json(500, f"Internal server error: {exc.__class__.__name__}")

    # -- handlers --------------------------------------------------------------
    def _handle_static(self, filename: str, content_type: str):
        """Serve one fixed, allowlisted frontend file from BASE_DIR."""
        file_path = BASE_DIR / filename
        try:
            data = file_path.read_bytes()
        except OSError:
            return self._send_error_json(404, f"{filename} not found.")
        self._send_bytes(200, content_type, data)

    def _handle_health(self):
        self._send_json(200, {"status": "ok", "time": time.time()})

    def _handle_body_model(self):
        try:
            data = _fetch_cached(f"{HF_BODY_ASSETS}/model.json", CACHE_DIR / "body" / "model.json")
        except (urllib.error.URLError, OSError):
            return self._send_error_json(502, "Could not fetch body model.")
        self._send_bytes(200, "application/json; charset=utf-8", data)

    def _handle_body_mesh(self, filename: str):
        if not filename or not MESH_NAME_RE.match(filename) or ".." in filename or "/" in filename or "\\" in filename:
            return self._send_error_json(400, "Invalid mesh filename.")
        cache_root = (CACHE_DIR / "body" / "meshes").resolve()
        cache_path = (cache_root / filename).resolve()
        if cache_root != cache_path.parent:
            return self._send_error_json(400, "Invalid mesh filename.")
        try:
            data = _fetch_cached(f"{HF_BODY_ASSETS}/meshes/{filename}", cache_path)
        except (urllib.error.URLError, OSError):
            return self._send_error_json(502, "Could not fetch mesh file.")
        self._send_bytes(200, "model/stl", data)

    def _handle_data_manifest(self):
        try:
            data = _fetch_cached(f"{HF_DATA}/manifest.json", CACHE_DIR / "data" / "manifest.json")
        except (urllib.error.URLError, OSError):
            return self._send_error_json(502, "Could not fetch manifest.")
        if len(data) > MAX_MANIFEST_BYTES:
            return self._send_error_json(502, "Manifest exceeds safety cap.")
        self._send_bytes(200, "application/json; charset=utf-8", data)

    def _handle_data_file(self, filename: str):
        if (
            not filename
            or not DATA_FILENAME_RE.match(filename)
            or ".." in filename
            or "/" in filename
            or "\\" in filename
        ):
            return self._send_error_json(400, "Invalid data file name.")

        try:
            manifest = _load_manifest()
        except (urllib.error.URLError, OSError):
            return self._send_error_json(502, "Could not reach data source.")
        except (ValueError, json.JSONDecodeError):
            return self._send_error_json(502, "Data source manifest is invalid.")

        allowed = _manifest_allowed_data_files(manifest)
        if filename not in allowed:
            return self._send_error_json(404, "Not found.")

        part = allowed[filename]
        cache_root = (CACHE_DIR / "data").resolve()
        cache_path = (cache_root / filename).resolve()
        if cache_root != cache_path.parent:
            return self._send_error_json(400, "Invalid data file name.")
        cap = _size_cap_for_part(part.get("bytes"))

        try:
            data = _fetch_verify_cache(
                f"{HF_DATA}/{filename}", cache_path, part.get("bytes"), part.get("sha256"), cap
            )
        except DataIntegrityError:
            return self._send_error_json(502, "Downloaded data file failed integrity verification.")
        except ValueError:
            return self._send_error_json(502, "Downloaded data file exceeded the safety size cap.")
        except (urllib.error.URLError, OSError):
            return self._send_error_json(502, "Could not fetch data file.")

        # Bytes are proxied unchanged: files stay gzip-compressed on disk and
        # on the wire. No Content-Encoding header is set, since the browser
        # decompresses these itself via DecompressionStream; setting
        # Content-Encoding: gzip here would make urllib/browsers that do
        # transparent transfer decoding unwrap it unexpectedly.
        self._send_bytes(200, "application/octet-stream", data)

    def _handle_neurons(self):
        try:
            sample = get_neuron_sample()
        except (urllib.error.URLError, OSError):
            return self._send_error_json(502, "Could not fetch neuron data.")
        except (ValueError, json.JSONDecodeError, zlib.error):
            return self._send_error_json(502, "Neuron data source returned malformed data.")
        self._send_json(200, sample)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"Flying Fly server listening on http://0.0.0.0:{PORT}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
