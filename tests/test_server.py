import unittest

try:
    from fastapi.testclient import TestClient
    from voxflow import server
    from voxflow.config import Settings
    from voxflow.pipeline import VoicePipeline
    from voxflow.store import Store
    from voxflow.testing import MockLLM, MockSTT, MockTTS
    app = server.app
    server.pipeline = VoicePipeline(Settings(redis_url='', database_url=''), stt=MockSTT(), llm=MockLLM(), tts=MockTTS(), store=Store())
except ImportError:  # fastapi/httpx not installed
    app = None


@unittest.skipIf(app is None, "fastapi + httpx not installed")
class ServerTests(unittest.TestCase):
    def test_health(self):
        self.assertTrue(TestClient(app).get("/health").json()["ok"])  # llm_up is False without a server

    def test_websocket_turn(self):
        with TestClient(app).websocket_connect("/ws?session=ws1") as ws:
            ws.send_json({"type": "text", "text": "what is 6 * 7"})
            evs = []
            while not evs or evs[-1]["type"] != "metrics":
                evs.append(ws.receive_json())
        self.assertIn("42", " ".join(e.get("text", "") for e in evs if e["type"] == "token"))

    def test_metrics_endpoint(self):
        self.assertIn("turns", TestClient(app).get("/api/metrics").json())


if __name__ == "__main__":
    unittest.main()
