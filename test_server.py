#!/usr/bin/env python3
"""Minimal stdlib tests for server.py.

These tests spin up the real server on an ephemeral port and exercise input
validation, static-file routing, and safety guarantees. They do not require
network access to Gemini or Hugging Face: Gemini and Hugging Face calls are
mocked where behavior depends on them (see GeminiMockedTestCase and
NeuronSampleMockedTestCase); everything else only exercises code paths that
run before any outbound call, or checks correct error handling when an
upstream is genuinely unreachable (no assertion requires live internet
access to pass).

Run with: python3 -m unittest test_server.py -v
"""
import gzip
import json
import threading
import unittest
import urllib.error
import urllib.request
from unittest import mock

import server as srv


class ServerTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.httpd = srv.ThreadingHTTPServer(("127.0.0.1", 0), srv.Handler)
        cls.port = cls.httpd.server_address[1]
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def _url(self, path):
        return f"http://127.0.0.1:{self.port}{path}"

    def _get(self, path):
        try:
            with urllib.request.urlopen(self._url(path), timeout=10) as resp:
                return resp.status, resp.getheader("Content-Type"), resp.read()
        except urllib.error.HTTPError as exc:
            return exc.code, exc.headers.get("Content-Type"), exc.read()

    def _get_json(self, path):
        status, _, body = self._get(path)
        return status, json.loads(body.decode("utf-8"))

    def _post_raw(self, path, body_bytes, content_length=None):
        """POST raw bytes with an explicit (possibly lying) Content-Length."""
        length = len(body_bytes) if content_length is None else content_length
        req = urllib.request.Request(
            self._url(path),
            data=body_bytes,
            headers={"Content-Type": "application/json", "Content-Length": str(length)},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.status, json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read().decode("utf-8"))

    def _post_json(self, path, payload):
        body = json.dumps(payload).encode("utf-8")
        return self._post_raw(path, body)

    # -- static frontend routes ---------------------------------------------

    def test_root_serves_index_html(self):
        status, content_type, body = self._get("/")
        self.assertEqual(status, 200)
        self.assertIn("text/html", content_type or "")
        text = body.decode("utf-8", "replace").lower()
        self.assertTrue(text.startswith("<!doctype html") or "<html" in text)

    def test_index_html_route_serves_same_file(self):
        status, content_type, body = self._get("/index.html")
        self.assertEqual(status, 200)
        self.assertIn("text/html", content_type or "")
        self.assertTrue(len(body) > 0)

    def test_style_css_serves(self):
        status, content_type, body = self._get("/style.css")
        self.assertEqual(status, 200)
        self.assertIn("text/css", content_type or "")
        self.assertTrue(len(body) > 0)

    def test_game_js_serves(self):
        status, content_type, body = self._get("/game.js")
        self.assertEqual(status, 200)
        self.assertIn("javascript", content_type or "")
        self.assertTrue(len(body) > 0)

    def test_fly_rig_js_serves(self):
        status, content_type, body = self._get("/fly_rig.js")
        self.assertEqual(status, 200)
        self.assertIn("javascript", content_type or "")
        self.assertTrue(len(body) > 0)

    def test_root_with_querystring_still_serves(self):
        status, content_type, body = self._get("/?v=123")
        self.assertEqual(status, 200)
        self.assertIn("text/html", content_type or "")

    # -- API routes -----------------------------------------------------------

    def test_health(self):
        status, payload = self._get_json("/api/health")
        self.assertEqual(status, 200)
        self.assertEqual(payload.get("status"), "ok")

    def test_unknown_route_404(self):
        status, payload = self._get_json("/nope")
        self.assertEqual(status, 404)
        self.assertIn("error", payload)

    def test_chat_rejects_unknown_npc(self):
        status, payload = self._post_json("/api/chat", {"npc": "Неизвестный", "message": "Привет"})
        self.assertEqual(status, 400)
        self.assertIn("error", payload)

    def test_chat_rejects_empty_message(self):
        status, payload = self._post_json("/api/chat", {"npc": "Зина", "message": "   "})
        self.assertEqual(status, 400)
        self.assertIn("error", payload)

    def test_chat_rejects_oversized_message(self):
        status, payload = self._post_json(
            "/api/chat", {"npc": "Зина", "message": "a" * (srv.MAX_MESSAGE_LEN + 1)}
        )
        self.assertEqual(status, 400)
        self.assertIn("error", payload)

    def test_chat_rejects_missing_fields(self):
        status, payload = self._post_json("/api/chat", {})
        self.assertEqual(status, 400)
        self.assertIn("error", payload)

    def test_chat_rejects_oversized_body_413(self):
        """Regression test: a JSON body larger than MAX_CHAT_BODY_BYTES must
        be rejected with 413 before any JSON parsing or Gemini call is
        attempted (Content-Length is checked first in _handle_chat).
        """
        huge_message = "a" * (srv.MAX_CHAT_BODY_BYTES + 500)
        body = json.dumps({"npc": "Зина", "message": huge_message}).encode("utf-8")
        self.assertGreater(len(body), srv.MAX_CHAT_BODY_BYTES)
        status, payload = self._post_raw("/api/chat", body)
        self.assertEqual(status, 413)
        self.assertIn("error", payload)

    def test_mesh_path_rejects_traversal(self):
        status, _, _ = self._get("/assets/body/meshes/..%2f..%2fserver.py")
        self.assertIn(status, (400, 404))

    def test_mesh_path_rejects_bad_extension(self):
        status, payload = self._get_json("/assets/body/meshes/evil.py")
        self.assertEqual(status, 400)
        self.assertIn("error", payload)

    def test_personas_cover_required_npcs(self):
        expected = {"Зина", "Артем", "Григорий", "Петрович", "Барсик", "Даня"}
        self.assertEqual(set(srv.NPC_PERSONAS.keys()), expected)

    # -- safety: never serve project/config files ---------------------------

    def test_env_and_source_files_not_served(self):
        for path in (
            "/.env",
            "/.env.example",
            "/server.py",
            "/test_server.py",
            "/.gitignore",
            "/README.md",
        ):
            status, _, _ = self._get(path)
            self.assertEqual(status, 404, f"expected 404 for {path}")


class _FakeHTTPResponse:
    """Minimal context-manager stand-in for urllib.request.urlopen()'s return
    value, used to mock successful Gemini responses without any network
    access."""

    def __init__(self, payload_bytes):
        self._payload = payload_bytes

    def read(self):
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


class GeminiMockedTestCase(unittest.TestCase):
    """call_gemini() tests with urllib.request.urlopen mocked out: no real
    network call to Google is made."""

    def setUp(self):
        self._orig_key = srv.GEMINI_API_KEY
        srv.GEMINI_API_KEY = "test-secret-key-should-never-leak"

    def tearDown(self):
        srv.GEMINI_API_KEY = self._orig_key

    def test_request_payload_uses_camelcase_system_instruction(self):
        captured = {}

        def fake_urlopen(req, timeout=None):
            captured["body"] = json.loads(req.data.decode("utf-8"))
            captured["url"] = req.full_url
            reply = {"candidates": [{"content": {"parts": [{"text": "Привет!"}]}}]}
            return _FakeHTTPResponse(json.dumps(reply).encode("utf-8"))

        with mock.patch.object(srv.urllib.request, "urlopen", side_effect=fake_urlopen):
            text = srv.call_gemini("persona", "hi")

        self.assertEqual(text, "Привет!")
        self.assertIn("systemInstruction", captured["body"])
        self.assertNotIn("system_instruction", captured["body"])
        self.assertEqual(
            captured["body"]["systemInstruction"]["parts"][0]["text"], "persona"
        )
        self.assertEqual(
            captured["body"]["contents"][0]["parts"][0]["text"], "hi"
        )

    def test_missing_api_key_rejected_before_any_network_call(self):
        srv.GEMINI_API_KEY = ""
        with mock.patch.object(srv.urllib.request, "urlopen") as mocked:
            with self.assertRaises(srv.ChatError) as ctx:
                srv.call_gemini("persona", "hi")
        mocked.assert_not_called()
        self.assertEqual(ctx.exception.status, 500)

    def test_http_error_from_gemini_never_leaks_api_key(self):
        def fake_urlopen(req, timeout=None):
            err = urllib.error.HTTPError(
                req.full_url, 503, "Service Unavailable", {}, None
            )
            err.read = lambda: b'{"error": "upstream unavailable"}'
            raise err

        with mock.patch.object(srv.urllib.request, "urlopen", side_effect=fake_urlopen):
            with self.assertRaises(srv.ChatError) as ctx:
                srv.call_gemini("persona", "hi")

        self.assertEqual(ctx.exception.status, 502)
        self.assertNotIn("key=", ctx.exception.message)
        self.assertNotIn(srv.GEMINI_API_KEY, ctx.exception.message)

    def test_url_error_from_gemini_never_leaks_api_key(self):
        def fake_urlopen(req, timeout=None):
            raise urllib.error.URLError("timed out")

        with mock.patch.object(srv.urllib.request, "urlopen", side_effect=fake_urlopen):
            with self.assertRaises(srv.ChatError) as ctx:
                srv.call_gemini("persona", "hi")

        self.assertEqual(ctx.exception.status, 502)
        self.assertNotIn("key=", ctx.exception.message)
        self.assertNotIn(srv.GEMINI_API_KEY, ctx.exception.message)

    def test_blocked_response_reports_block_reason_not_shape_error(self):
        def fake_urlopen(req, timeout=None):
            reply = {"candidates": [], "promptFeedback": {"blockReason": "SAFETY"}}
            return _FakeHTTPResponse(json.dumps(reply).encode("utf-8"))

        with mock.patch.object(srv.urllib.request, "urlopen", side_effect=fake_urlopen):
            with self.assertRaises(srv.ChatError) as ctx:
                srv.call_gemini("persona", "hi")

        self.assertEqual(ctx.exception.status, 502)
        self.assertIn("SAFETY", ctx.exception.message)

    def test_malformed_response_shape_raises_chat_error(self):
        def fake_urlopen(req, timeout=None):
            reply = {"candidates": [{"content": {}}]}
            return _FakeHTTPResponse(json.dumps(reply).encode("utf-8"))

        with mock.patch.object(srv.urllib.request, "urlopen", side_effect=fake_urlopen):
            with self.assertRaises(srv.ChatError) as ctx:
                srv.call_gemini("persona", "hi")

        self.assertEqual(ctx.exception.status, 502)


class NeuronSampleMockedTestCase(unittest.TestCase):
    """_build_neuron_sample() tests with _fetch_cached mocked out: no real
    network access to Hugging Face is made."""

    def setUp(self):
        srv._neuron_sample_cache = None

    def tearDown(self):
        srv._neuron_sample_cache = None

    def _patched_fetch(self, manifest: dict, rows):
        gz_bytes = gzip.compress(json.dumps(rows).encode("utf-8"))
        manifest_bytes = json.dumps(manifest).encode("utf-8")

        def fake_fetch_cached(url, cache_path):
            if url.endswith("manifest.json"):
                return manifest_bytes
            return gz_bytes

        return mock.patch.object(srv, "_fetch_cached", side_effect=fake_fetch_cached)

    def test_list_of_lists_rows_are_parsed(self):
        manifest = {
            "dataset": "MaleCNS v1.0",
            "neurons": 3,
            "metadata": "neurons.json.gz",
            "metadataColumns": [
                "bodyId", "type", "superclass", "side", "consensusNT", "fastSign", "somaLocation8nm",
            ],
        }
        rows = [
            [1, "KCg", "Kenyon", "L", "acetylcholine", 1, [100, 200, 300]],
            [2, "KCg", "Kenyon", "R", "acetylcholine", 1, [110, 210, 310]],
            [3, "PN", "olfactory", "L", "glutamate", -1, [-100, -200, -300]],
        ]
        with self._patched_fetch(manifest, rows):
            sample = srv._build_neuron_sample()

        self.assertEqual(sample["dataset"], "MaleCNS v1.0")
        self.assertEqual(sample["total"], 3)
        self.assertEqual(len(sample["points"]), 3)
        for point in sample["points"]:
            self.assertIn("x", point)
            self.assertIn("y", point)
            self.assertIn("z", point)
            self.assertIn("type", point)
            self.assertIn("side", point)
            self.assertIn("nt", point)

    def test_list_of_dicts_rows_are_parsed(self):
        manifest = {
            "dataset": "MaleCNS v1.0",
            "neurons": 2,
            "metadata": "neurons.json.gz",
            "metadataColumns": [
                "bodyId", "type", "superclass", "side", "consensusNT", "fastSign", "somaLocation8nm",
            ],
        }
        rows = [
            {
                "bodyId": 1,
                "type": "KCg",
                "side": "L",
                "consensusNT": "acetylcholine",
                "somaLocation8nm": {"x": 10, "y": 20, "z": 30},
            },
            {
                "bodyId": 2,
                "type": "PN",
                "side": "R",
                "consensusNT": "gaba",
                "somaLocation8nm": {"x": -10, "y": -20, "z": -30},
            },
        ]
        with self._patched_fetch(manifest, rows):
            sample = srv._build_neuron_sample()

        self.assertEqual(len(sample["points"]), 2)
        xs = {round(p["x"], 3) for p in sample["points"]}
        self.assertEqual(xs, {-1.0, 1.0})

    def test_rows_missing_soma_location_are_skipped_not_fabricated(self):
        manifest = {
            "dataset": "MaleCNS v1.0",
            "neurons": 2,
            "metadata": "neurons.json.gz",
            "metadataColumns": [
                "bodyId", "type", "superclass", "side", "consensusNT", "fastSign", "somaLocation8nm",
            ],
        }
        rows = [
            [1, "KCg", "Kenyon", "L", "acetylcholine", 1, [1, 2, 3]],
            [2, "PN", "olfactory", "R", "glutamate", -1, None],
        ]
        with self._patched_fetch(manifest, rows):
            sample = srv._build_neuron_sample()

        self.assertEqual(len(sample["points"]), 1)

    def test_non_array_metadata_raises_value_error(self):
        manifest = {"dataset": "MaleCNS v1.0", "neurons": 1, "metadata": "neurons.json.gz"}
        with self._patched_fetch(manifest, {"not": "a list"}):
            with self.assertRaises(ValueError):
                srv._build_neuron_sample()

    def test_all_rows_missing_coordinates_raises_value_error(self):
        manifest = {
            "dataset": "MaleCNS v1.0",
            "neurons": 1,
            "metadata": "neurons.json.gz",
            "metadataColumns": ["bodyId", "somaLocation8nm"],
        }
        rows = [[1, None], [2, "not-coordinates"]]
        with self._patched_fetch(manifest, rows):
            with self.assertRaises(ValueError):
                srv._build_neuron_sample()


class SafeGunzipTestCase(unittest.TestCase):
    """_safe_gunzip() must decompress normal gzip data correctly and must
    refuse to materialize output beyond an explicit cap (decompression-bomb
    protection)."""

    def test_round_trips_normal_data(self):
        original = json.dumps({"hello": "world", "n": list(range(1000))}).encode("utf-8")
        compressed = gzip.compress(original)
        restored = srv._safe_gunzip(compressed, max_output_bytes=10 * 1024 * 1024)
        self.assertEqual(restored, original)

    def test_raises_when_output_exceeds_cap(self):
        # A highly compressible payload that decompresses far beyond a tiny cap.
        original = b"0" * (5 * 1024 * 1024)  # 5 MiB of a single repeated byte
        compressed = gzip.compress(original)
        with self.assertRaises(ValueError):
            srv._safe_gunzip(compressed, max_output_bytes=1024)  # 1 KiB cap


if __name__ == "__main__":
    unittest.main()
