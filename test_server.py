#!/usr/bin/env python3
"""Minimal stdlib tests for server.py.

These tests spin up the real server on an ephemeral port and exercise input
validation and routing. They do not require network access to Gemini or
Hugging Face: Gemini-dependent behavior is only smoke-tested for the
validation errors that happen before any outbound call, and asset/neuron
endpoints are checked only for correct error handling when unreachable is
acceptable (no assertions require live internet access to pass).

Run with: python3 -m unittest test_server.py -v
"""
import json
import threading
import unittest
import urllib.error
import urllib.request

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
                return resp.status, json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read().decode("utf-8"))

    def _post_json(self, path, payload):
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self._url(path), data=body, headers={"Content-Type": "application/json"}, method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.status, json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read().decode("utf-8"))

    def test_health(self):
        status, payload = self._get("/api/health")
        self.assertEqual(status, 200)
        self.assertEqual(payload.get("status"), "ok")

    def test_unknown_route_404(self):
        status, payload = self._get("/nope")
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

    def test_mesh_path_rejects_traversal(self):
        status, payload = self._get("/assets/body/meshes/..%2f..%2fserver.py")
        self.assertIn(status, (400, 404))

    def test_mesh_path_rejects_bad_extension(self):
        status, payload = self._get("/assets/body/meshes/evil.py")
        self.assertEqual(status, 400)
        self.assertIn("error", payload)

    def test_env_file_not_served(self):
        for path in ("/.env", "/.env.example", "/server.py"):
            status, _ = self._get(path)
            self.assertEqual(status, 404)

    def test_personas_cover_required_npcs(self):
        expected = {"Зина", "Артем", "Григорий", "Петрович", "Барсик", "Даня"}
        self.assertEqual(set(srv.NPC_PERSONAS.keys()), expected)


if __name__ == "__main__":
    unittest.main()
