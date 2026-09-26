#!/usr/bin/env python3
"""Flying Fly backend server.

Pure Python 3 standard library HTTP server (no third-party dependencies).
Runs on port 8085 by default and exposes:

  GET  /
  GET  /index.html
  GET  /style.css
  GET  /game.js
  GET  /fly_rig.js
      Serves the app shell and its frontend assets from a small, fixed
      allowlist (each URL maps to exactly one file in the project root,
      with the correct MIME type). This is not a generic static-file or
      directory-listing handler: any path not on this list, and not one of
      the API/asset routes below, is a 404. In particular, dotfiles such as
      as .env are never served, and the server's own source files (e.g.
      server.py, test_server.py) are never served.

  GET  /api/health
      Liveness check.

  POST /api/chat
      Body: {"npc": "<one of the known character names>", "message": "<text>"}
      Sends a persona-flavored prompt to Google Gemini (generateContent) and
      returns {"npc": ..., "reply": ...}. All chat output is model-generated
      text; it is not scripted dialogue and is not validated game content.
      Errors from Gemini are surfaced as sanitized JSON; the API key (sent as
      a URL query parameter to Gemini) is never included in any response,
      error message, or log line.

  GET  /assets/body/model.json
  GET  /assets/body/meshes/<filename>.stl
      Proxies and locally caches the fruit-fly body model assets published by
      the Hugging Face Space "Xenova/fruit-fly-simulation"
      (public/body/assets/model.json and public/body/assets/meshes/*.stl).
      Files are fetched once and served from a local cache directory after
      that. Mesh filenames are strictly validated to prevent directory
      traversal or arbitrary upstream paths.

  GET  /api/neurons
      Returns a deterministic ~16,000-point sample of REAL soma coordinates
      from the same Hugging Face Space's public/data/manifest.json and
      public/data/neurons.json.gz (MaleCNS v1.0 dataset, 166,700 neurons,
      brain + nerve cord). No synthetic or fabricated points are ever
      returned; if the source data cannot be fetched or parsed, the endpoint
      returns an explicit JSON error instead of a fallback. Note: the exact
      per-row shape of neurons.json.gz (list-of-lists vs list-of-objects) is
      not fully confirmed from the published manifest alone; this code
      handles both shapes defensively and only emits points it can actually
      parse from the real source data, never fabricated ones. Downloaded and
      decompressed data is size-capped (see MAX_* constants below) so a
      corrupted or hostile upstream response cannot exhaust memory.

Configuration (read from a local .env file or the real process environment,
process environment wins if both are set):

  GEMINI_API_KEY   Required for /api/chat. Never logged or echoed back.
  GEMINI_MODEL     Defaults to "gemini-3.5-flash-lite".
  PORT             Defaults to 8085.

The .env file itself, and any dotfile, is never served over HTTP by this
server: only the fixed set of routes above exists (a small static allowlist
plus the API/asset routes), there is no generic static-file or
directory-listing handler.
"""
from __future__ import annotations

import gzip
import json
import os
import re
import socket
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


GEMINI_API_KEY = get_env("GEMINI_API_KEY", "")
GEMINI_MODEL = get_env("GEMINI_MODEL", "gemini-3.5-flash-lite")
try:
    PORT = int(get_env("PORT", "8085") or "8085")
except ValueError:
    PORT = 8085

HF_SPACE_ROOT = "https://huggingface.co/spaces/Xenova/fruit-fly-simulation/resolve/main"
HF_BODY_ASSETS = f"{HF_SPACE_ROOT}/public/body/assets"
HF_DATA = f"{HF_SPACE_ROOT}/public/data"
HF_ATTRIBUTION_SOURCE = (
    "Xenova/fruit-fly-simulation Hugging Face Space "
    "(public/data/manifest.json, public/data/neurons.json.gz); "
    "MaleCNS v1.0 connectome, FlyEM/HHMI Janelia, University of Cambridge, "
    "MRC LMB, Google Research; CC BY 4.0"
)

HTTP_TIMEOUT = 20  # seconds, applies to every outbound fetch (HF + Gemini)
NEURON_SAMPLE_SIZE = 16000
MAX_CHAT_BODY_BYTES = 4096
MAX_MESSAGE_LEN = 500

# Sanity caps on untrusted upstream data. These are deliberately generous for
# the real MaleCNS v1.0 dataset (manifest.json is a few KB, neurons.json.gz
# is a few MB compressed) but still bound worst-case memory/CPU use if the
# upstream ever returns something corrupted, truncated, or hostile. We never
# trust the content of a downloaded file just because the request succeeded.
MAX_MANIFEST_BYTES = 5 * 1024 * 1024  # 5 MiB
MAX_NEURON_GZ_BYTES = 50 * 1024 * 1024  # 50 MiB compressed
MAX_NEURON_DECOMPRESSED_BYTES = 200 * 1024 * 1024  # 200 MiB decompressed
MAX_NEURON_ROWS = 5_000_000  # sanity bound on the parsed row count

MESH_NAME_RE = re.compile(r"^[A-Za-z0-9_\-]{1,80}\.stl$")

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
# (index.html, style.css, game.js, fly_rig.js) are out of scope for this
# server change and are not modified here.
# ---------------------------------------------------------------------------

STATIC_FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/style.css": ("style.css", "text/css; charset=utf-8"),
    "/game.js": ("game.js", "application/javascript; charset=utf-8"),
    "/fly_rig.js": ("fly_rig.js", "application/javascript; charset=utf-8"),
}

# ---------------------------------------------------------------------------
# NPC personas (lightweight; no additional game lore is documented in this
# project, so each persona is a short, self-consistent character sketch tied
# only to its given name). Instructions are in Russian since replies must be
# in Russian.
# ---------------------------------------------------------------------------

_PERSONA_SUFFIX = (
    " Не выходи из роли, не упоминай, что ты искусственный интеллект или "
    "языковая модель. Отвечай только по-русски, живо и коротко "
    "(1-2 предложения)."
)

NPC_PERSONAS = {
    "Зина": (
        "Ты — Зина, приветливая пожилая жительница деревни. Ты любишь "
        "готовить, ухаживать за огородом и делиться деревенскими новостями. "
        "Говоришь тепло и по-простому, обращаешься к собеседнику "
        "по-доброму." + _PERSONA_SUFFIX
    ),
    "Артем": (
        "Ты — Артем, молодой энергичный механик, который вечно что-то "
        "ремонтирует и мастерит. Говоришь быстро, с энтузиазмом, иногда "
        "упоминаешь инструменты и детали." + _PERSONA_SUFFIX
    ),
    "Григорий": (
        "Ты — Григорий, пожилой и немного суровый деревенский старейшина. "
        "Говоришь мало, но по делу, с житейской мудростью." + _PERSONA_SUFFIX
    ),
    "Петрович": (
        "Ты — Петрович, практичный деревенский мастер на все руки. "
        "Говоришь прямо, без лишних слов, иногда называешь собеседника "
        "«дружище»." + _PERSONA_SUFFIX
    ),
    "Барсик": (
        "Ты — Барсик, деревенский кот. Ты игривый, хитрый и немного "
        "капризный, воспринимаешь всё по-кошачьи, снисходишь до разговора "
        "с человеком нехотя." + _PERSONA_SUFFIX
    ),
    "Даня": (
        "Ты — Даня, любопытный деревенский мальчишка. Тебе всё интересно, "
        "ты задаёшь вопросы и легко увлекаешься, говоришь живо и "
        "по-детски." + _PERSONA_SUFFIX
    ),
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
        parts = re.split(r"[,\s]+", value.strip("[]() "))
        parts = [p for p in parts if p]
        if len(parts) >= 3:
            try:
                return float(parts[0]), float(parts[1]), float(parts[2])
            except ValueError:
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
    downloaded/decompressed data is validated and size-capped before use;
    a successful download is never assumed to be well-formed.
    """
    manifest_bytes = _fetch_cached(f"{HF_DATA}/manifest.json", CACHE_DIR / "data" / "manifest.json")
    if len(manifest_bytes) > MAX_MANIFEST_BYTES:
        raise ValueError(f"manifest.json exceeds {MAX_MANIFEST_BYTES} byte safety cap")
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("manifest.json is not a JSON object")

    metadata_file = manifest.get("metadata", "neurons.json.gz")
    if not isinstance(metadata_file, str) or not metadata_file:
        raise ValueError("manifest.json has no usable 'metadata' filename")
    columns = manifest.get("metadataColumns") or DEFAULT_NEURON_METADATA_COLUMNS
    if not isinstance(columns, list) or not all(isinstance(c, str) for c in columns):
        columns = DEFAULT_NEURON_METADATA_COLUMNS
    dataset_name = manifest.get("dataset", "MaleCNS v1.0")
    total_neurons = manifest.get("neurons")

    gz_bytes = _fetch_cached(f"{HF_DATA}/{metadata_file}", CACHE_DIR / "data" / metadata_file)
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
            candidates.append((xyz, fields))
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
    for (x, y, z), fields in candidates:
        point = {
            "x": (x - cx) / half_range,
            "y": (y - cy) / half_range,
            "z": (z - cz) / half_range,
        }
        if fields.get("type") is not None:
            point["type"] = fields.get("type")
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
# Gemini chat
# ---------------------------------------------------------------------------


class ChatError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


def call_gemini(persona_prompt: str, message: str) -> str:
    """Call Gemini's generateContent REST endpoint and return the reply text.

    Security note: GEMINI_API_KEY is sent to Google as a URL query parameter
    (Google's documented auth mechanism for this endpoint). Every error path
    below builds its message only from the HTTP status code and a bounded,
    sanitized detail string -- never from the request URL or from str(exc) on
    the raised exception -- so the key can never leak into a client-visible
    error message or into anything this function returns.
    """
    if not GEMINI_API_KEY:
        raise ChatError(500, "Server is not configured with a Gemini API key.")

    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    )
    # NOTE: 'url' (which contains the API key) must never be interpolated
    # into a ChatError message or logged below.
    payload = {
        "systemInstruction": {"parts": [{"text": persona_prompt}]},
        "contents": [{"role": "user", "parts": [{"text": message}]}],
        "generationConfig": {"temperature": 0.9, "maxOutputTokens": 220},
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:300]
        raise ChatError(502, f"Gemini API returned an error (HTTP {exc.code}): {detail}") from None
    except urllib.error.URLError as exc:
        raise ChatError(502, f"Could not reach Gemini API: {exc.reason}") from None

    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise ChatError(502, "Gemini API returned a response that was not valid JSON.") from None

    if not isinstance(data, dict):
        raise ChatError(502, "Gemini API returned an unexpected response shape.")

    candidates = data.get("candidates")
    if not candidates or not isinstance(candidates, list):
        feedback = data.get("promptFeedback")
        block_reason = feedback.get("blockReason") if isinstance(feedback, dict) else None
        if block_reason:
            raise ChatError(502, f"Gemini API returned no content (blocked: {block_reason}).")
        raise ChatError(502, "Gemini API returned no candidates.")

    try:
        text = candidates[0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError, TypeError):
        raise ChatError(502, "Gemini API returned an unexpected response shape.") from None

    if not isinstance(text, str) or not text.strip():
        raise ChatError(502, "Gemini API returned an empty reply.")

    return text.strip()


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

    # -- routing -----------------------------------------------------------

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
                return self._handle_body_mesh(path[len("/assets/body/meshes/") :])
            return self._send_error_json(404, "Not found.")
        except Exception as exc:  # last-resort guard, never leak internals
            self._send_error_json(500, f"Internal server error: {exc.__class__.__name__}")

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        try:
            if path == "/api/chat":
                return self._handle_chat()
            return self._send_error_json(404, "Not found.")
        except Exception as exc:
            self._send_error_json(500, f"Internal server error: {exc.__class__.__name__}")

    # -- handlers ------------------------------------------------------------

    def _handle_static(self, filename: str, content_type: str):
        """Serve one fixed, allowlisted frontend file from BASE_DIR.

        `filename` always comes from the hardcoded STATIC_FILES table above,
        never from request input, so there is no path-traversal surface
        here. If the file is missing from the repository this returns a 404
        JSON error rather than guessing at content.
        """
        file_path = BASE_DIR / filename
        try:
            data = file_path.read_bytes()
        except OSError:
            return self._send_error_json(404, f"{filename} not found.")
        self._send_bytes(200, content_type, data)

    def _handle_health(self):
        self._send_json(200, {"status": "ok", "time": time.time()})

    def _handle_chat(self):
        length_header = self.headers.get("Content-Length")
        if length_header is None:
            return self._send_error_json(400, "Missing Content-Length header.")
        try:
            length = int(length_header)
        except ValueError:
            return self._send_error_json(400, "Invalid Content-Length header.")
        if length <= 0:
            return self._send_error_json(400, "Empty request body.")
        if length > MAX_CHAT_BODY_BYTES:
            return self._send_error_json(413, "Request body too large.")

        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            return self._send_error_json(400, "Request body must be valid JSON.")
        if not isinstance(payload, dict):
            return self._send_error_json(400, "Request body must be a JSON object.")

        npc = payload.get("npc")
        message = payload.get("message")

        if not isinstance(npc, str) or npc not in NPC_PERSONAS:
            return self._send_error_json(
                400,
                "Field 'npc' must be one of: " + ", ".join(sorted(NPC_PERSONAS)),
            )
        if not isinstance(message, str):
            return self._send_error_json(400, "Field 'message' must be a string.")
        message = message.strip()
        if not message:
            return self._send_error_json(400, "Field 'message' must not be empty.")
        if len(message) > MAX_MESSAGE_LEN:
            return self._send_error_json(
                400, f"Field 'message' must be at most {MAX_MESSAGE_LEN} characters."
            )

        try:
            reply = call_gemini(NPC_PERSONAS[npc], message)
        except ChatError as exc:
            return self._send_error_json(exc.status, exc.message)

        self._send_json(200, {"npc": npc, "reply": reply})

    def _handle_body_model(self):
        cache_path = CACHE_DIR / "body" / "model.json"
        try:
            data = _fetch_cached(f"{HF_BODY_ASSETS}/model.json", cache_path)
        except (urllib.error.URLError, OSError) as exc:
            return self._send_error_json(502, f"Could not fetch model.json: {exc}")
        self._send_bytes(200, "application/json; charset=utf-8", data)

    def _handle_body_mesh(self, filename: str):
        if not MESH_NAME_RE.match(filename) or ".." in filename or "/" in filename:
            return self._send_error_json(400, "Invalid mesh filename.")
        cache_path = CACHE_DIR / "body" / "meshes" / filename
        # Defense in depth: resolved path must stay inside the cache dir.
        resolved = cache_path.resolve()
        if not str(resolved).startswith(str((CACHE_DIR / "body" / "meshes").resolve())):
            return self._send_error_json(400, "Invalid mesh path.")
        try:
            data = _fetch_cached(f"{HF_BODY_ASSETS}/meshes/{filename}", cache_path)
        except (urllib.error.URLError, OSError) as exc:
            return self._send_error_json(502, f"Could not fetch mesh '{filename}': {exc}")
        self._send_bytes(200, "application/vnd.ms-pki.stl", data)

    def _handle_neurons(self):
        try:
            sample = get_neuron_sample()
        except (urllib.error.URLError, OSError) as exc:
            return self._send_error_json(502, f"Could not fetch neuron data: {exc}")
        except (ValueError, json.JSONDecodeError, zlib.error) as exc:
            return self._send_error_json(502, f"Could not parse neuron data: {exc}")
        self._send_json(200, sample)


def main():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    server.daemon_threads = True
    print(f"Flying Fly backend listening on http://0.0.0.0:{PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
