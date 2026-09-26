#!/usr/bin/env python3
"""Minimal stdlib tests for server.py.

These tests spin up the real server on an ephemeral port and exercise input
validation, static-file routing, and safety guarantees. They do not require
network access to Hugging Face: all outbound fetches are mocked where
behavior depends on them (see DataProxyTestCase and
NeuronSampleMockedTestCase); everything else only exercises code paths that
run before any outbound call, or checks correct error handling when an
upstream is genuinely unreachable (no assertion requires live internet
access to pass).

Run with: python3 -m unittest test_server.py -v
"""
import gzip
import hashlib
import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path
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

    @unittest.skipUnless(
        (srv.BASE_DIR / "neuro_sim.js").exists(),
        "neuro_sim.js not present yet (owned by another in-progress change)",
    )
    def test_neuro_sim_js_serves_when_present(self):
        status, content_type, body = self._get("/neuro_sim.js")
        self.assertEqual(status, 200)
        self.assertIn("javascript", content_type or "")
        self.assertTrue(len(body) > 0)

    def test_neuro_sim_js_route_is_a_plain_404_when_file_missing(self):
        """neuro_sim.js is on the static allowlist even before the file
        exists in the repo (another change may add it); until it exists,
        requesting it must be an ordinary 404, not a crash.
        """
        if (srv.BASE_DIR / "neuro_sim.js").exists():
            self.skipTest("neuro_sim.js already exists; covered by the serves-when-present test")
        status, payload = self._get_json("/neuro_sim.js")
        self.assertEqual(status, 404)
        self.assertIn("error", payload)

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

    def test_chat_route_removed_returns_404(self):
        """Gemini chat has been removed entirely; POST /api/chat (and any
        other POST route) must now be a plain 404, not a crash and not a
        500 (do_POST's fallback previously called _send_error_json without
        self, which raised NameError and was masked by an outer except
        turning it into a 500; that bug is fixed as part of this removal).
        """
        status, payload = self._post_json("/api/chat", {"npc": "x", "message": "hi"})
        self.assertEqual(status, 404)
        self.assertIn("error", payload)

    def test_any_post_route_is_404(self):
        status, payload = self._post_json("/anything", {})
        self.assertEqual(status, 404)
        self.assertIn("error", payload)

    def test_mesh_path_rejects_traversal(self):
        status, _, _ = self._get("/assets/body/meshes/..%2f..%2fserver.py")
        self.assertIn(status, (400, 404))

    def test_mesh_path_rejects_bad_extension(self):
        status, payload = self._get_json("/assets/body/meshes/evil.py")
        self.assertEqual(status, 400)
        self.assertIn("error", payload)

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


class DataProxyTestCase(ServerTestCase):
    """Tests for /data/manifest.json and /data/<file>, with all Hugging Face
    fetches mocked via srv._http_get -- no real network access is made, and
    the on-disk cache is redirected to a temp directory for isolation."""

    def setUp(self):
        self._tmp_cache = tempfile.TemporaryDirectory()
        self._orig_cache_dir = srv.CACHE_DIR
        srv.CACHE_DIR = Path(self._tmp_cache.name)

    def tearDown(self):
        srv.CACHE_DIR = self._orig_cache_dir
        self._tmp_cache.cleanup()

    @staticmethod
    def _manifest(part_file="offsets-000.bin.gz", part_bytes=None, part_sha256=None):
        return {
            "dataset": "MaleCNS v1.0",
            "neurons": 166700,
            "edges": 25582938,
            "metadata": "neurons.json.gz",
            "metadataColumns": [
                "bodyId", "type", "superclass", "side", "consensusNT", "fastSign",
                "somaLocation8nm",
            ],
            "arrays": [
                {
                    "name": "offsets",
                    "length": 166701,
                    "parts": [
                        {"file": part_file, "bytes": part_bytes, "sha256": part_sha256}
                    ],
                },
            ],
        }

    def test_manifest_route_serves_bytes_as_is(self):
        manifest = self._manifest()
        manifest_bytes = json.dumps(manifest).encode("utf-8")
        with mock.patch.object(srv, "_http_get", return_value=manifest_bytes):
            status, content_type, body = self._get("/data/manifest.json")
            self.assertEqual(status, 200)
            self.assertIn("json", (content_type or "").lower())
            self.assertEqual(json.loads(body), manifest)

    def test_allowlisted_metadata_file_is_served(self):
        manifest = self._manifest()
        manifest_bytes = json.dumps(manifest).encode("utf-8")
        gz_bytes = gzip.compress(b"[]")

        def fake_http_get(url):
            if url.endswith("manifest.json"):
                return manifest_bytes
            return gz_bytes

        with mock.patch.object(srv, "_http_get", side_effect=fake_http_get):
            status, content_type, body = self._get("/data/neurons.json.gz")
            self.assertEqual(status, 200)
            self.assertEqual(content_type, "application/octet-stream")
            self.assertEqual(body, gz_bytes)

    def test_allowlisted_array_part_is_verified_and_served(self):
        part_bytes = b"fake-offsets-part-data"
        digest = hashlib.sha256(part_bytes).hexdigest()
        manifest = self._manifest(
            part_file="offsets-000.bin.gz", part_bytes=len(part_bytes), part_sha256=digest
        )
        manifest_bytes = json.dumps(manifest).encode("utf-8")

        def fake_http_get(url):
            if url.endswith("manifest.json"):
                return manifest_bytes
            return part_bytes

        with mock.patch.object(srv, "_http_get", side_effect=fake_http_get):
            status, content_type, body = self._get("/data/offsets-000.bin.gz")
            self.assertEqual(status, 200)
            self.assertEqual(content_type, "application/octet-stream")
            self.assertEqual(body, part_bytes)
            cached_path = srv.CACHE_DIR / "data" / "offsets-000.bin.gz"
            self.assertTrue(cached_path.exists())
            self.assertEqual(cached_path.read_bytes(), part_bytes)

    def test_non_manifest_filename_is_404(self):
        manifest = self._manifest()
        manifest_bytes = json.dumps(manifest).encode("utf-8")
        with mock.patch.object(srv, "_http_get", return_value=manifest_bytes):
            status, payload = self._get_json("/data/not-a-real-part.bin.gz")
            self.assertEqual(status, 404)
            self.assertIn("error", payload)

    def test_traversal_filename_rejected_before_any_fetch(self):
        with mock.patch.object(srv, "_http_get") as mocked:
            status, _, _ = self._get("/data/..%2f..%2fserver.py")
            self.assertIn(status, (400, 404))
            mocked.assert_not_called()

    def test_sha256_mismatch_returns_502_and_does_not_cache(self):
        part_bytes = b"fake-offsets-part-data"
        wrong_digest = "0" * 64
        manifest = self._manifest(
            part_file="offsets-000.bin.gz", part_bytes=len(part_bytes), part_sha256=wrong_digest
        )
        manifest_bytes = json.dumps(manifest).encode("utf-8")

        def fake_http_get(url):
            if url.endswith("manifest.json"):
                return manifest_bytes
            return part_bytes

        with mock.patch.object(srv, "_http_get", side_effect=fake_http_get):
            status, payload = self._get_json("/data/offsets-000.bin.gz")
            self.assertEqual(status, 502)
            self.assertIn("error", payload)
            cached_path = srv.CACHE_DIR / "data" / "offsets-000.bin.gz"
            self.assertFalse(cached_path.exists())

    def test_oversized_part_returns_502_and_does_not_cache(self):
        # Declared size is tiny (10 bytes); _size_cap_for_part pads that by
        # DATA_PART_SIZE_TOLERANCE_BYTES (4096), so the effective cap is 4106
        # bytes. The downloaded payload must clearly exceed that padded cap
        # (not just the tiny declared size) for this test to actually
        # exercise the oversized-part rejection path.
        part_bytes = b"x" * 8192
        manifest = self._manifest(part_file="offsets-000.bin.gz", part_bytes=10, part_sha256=None)
        manifest_bytes = json.dumps(manifest).encode("utf-8")

        def fake_http_get(url):
            if url.endswith("manifest.json"):
                return manifest_bytes
            return part_bytes

        with mock.patch.object(srv, "_http_get", side_effect=fake_http_get):
            status, payload = self._get_json("/data/offsets-000.bin.gz")
            self.assertEqual(status, 502)
            self.assertIn("error", payload)
            cached_path = srv.CACHE_DIR / "data" / "offsets-000.bin.gz"
            self.assertFalse(cached_path.exists())


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
                "bodyId", "type", "superclass", "side", "consensusNT", "fastSign",
                "somaLocation8nm",
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
            indices = {p["index"] for p in sample["points"]}
            self.assertEqual(indices, {0, 1, 2})
            for point in sample["points"]:
                self.assertIn("x", point)
                self.assertIn("y", point)
                self.assertIn("z", point)
                self.assertIn("index", point)
                self.assertIn("type", point)
                self.assertIn("superclass", point)
                self.assertIn("side", point)
                self.assertIn("nt", point)

    def test_list_of_dicts_rows_are_parsed(self):
        """_build_neuron_sample uniformly scales all three axes by the same
        divisor (the largest half-range among x/y/z, so shape/aspect ratio
        is preserved rather than each axis being independently stretched to
        fill [-1, 1]). With x-range [-10, 10] but z-range [-30, 30], the
        largest half-range is 30 (from z), so normalized x is +/-10/30, not
        +/-1.
        """
        manifest = {
            "dataset": "MaleCNS v1.0",
            "neurons": 2,
            "metadata": "neurons.json.gz",
            "metadataColumns": [
                "bodyId", "type", "superclass", "side", "consensusNT", "fastSign",
                "somaLocation8nm",
            ],
        }
        rows = [
            {
                "bodyId": 1,
                "type": "KCg",
                "superclass": "Kenyon",
                "side": "L",
                "consensusNT": "acetylcholine",
                "somaLocation8nm": {"x": 10, "y": 20, "z": 30},
            },
            {
                "bodyId": 2,
                "type": "PN",
                "superclass": "olfactory",
                "side": "R",
                "consensusNT": "gaba",
                "somaLocation8nm": {"x": -10, "y": -20, "z": -30},
            },
        ]
        with self._patched_fetch(manifest, rows):
            sample = srv._build_neuron_sample()

            self.assertEqual(len(sample["points"]), 2)
            xs = {round(p["x"], 3) for p in sample["points"]}
            expected = {round(-10 / 30, 3), round(10 / 30, 3)}
            self.assertEqual(xs, expected)
            indices = {p["index"] for p in sample["points"]}
            self.assertEqual(indices, {0, 1})
            superclasses = {p["superclass"] for p in sample["points"]}
            self.assertEqual(superclasses, {"Kenyon", "olfactory"})

    def test_rows_missing_soma_location_are_skipped_not_fabricated(self):
        manifest = {
            "dataset": "MaleCNS v1.0",
            "neurons": 2,
            "metadata": "neurons.json.gz",
            "metadataColumns": [
                "bodyId", "type", "superclass", "side", "consensusNT", "fastSign",
                "somaLocation8nm",
            ],
        }
        rows = [
            [1, "KCg", "Kenyon", "L", "acetylcholine", 1, [1, 2, 3]],
            [2, "PN", "olfactory", "R", "glutamate", -1, None],
        ]
        with self._patched_fetch(manifest, rows):
            sample = srv._build_neuron_sample()

            self.assertEqual(len(sample["points"]), 1)
            self.assertEqual(sample["points"][0]["index"], 0)

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
        original = b"0" * (5 * 1024 * 1024)  # 5 MiB of single repeated byte
        compressed = gzip.compress(original)
        with self.assertRaises(ValueError):
            srv._safe_gunzip(compressed, max_output_bytes=1 * 1024)  # 1 KiB cap


if __name__ == "__main__":
    unittest.main()
