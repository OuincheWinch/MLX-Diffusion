import sys
import unittest
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import state
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

class TestQueueRecovery(unittest.TestCase):
    def setUp(self):
        state.clear_recoverable_jobs()

    def tearDown(self):
        state.clear_recoverable_jobs()

    def test_record_and_get_recoverable(self):
        req_data = {
            "prompt": "a beautiful stormtrooper in the alps",
            "model": "flux2-klein-4b",
            "width": 1024,
            "height": 1024,
            "steps": 4,
            "seed": 42,
        }
        state.record_recoverable_job("test-job-1", req_data, reason="cancelled")
        items = state.get_recoverable_jobs()
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["id"], "test-job-1")
        self.assertEqual(items[0]["reason"], "cancelled")
        self.assertEqual(items[0]["request"]["prompt"], "a beautiful stormtrooper in the alps")

    def test_api_queue_recovery_list(self):
        req_data = {
            "prompt": "cyberpunk city neon lights",
            "model": "z-image-turbo",
            "width": 512,
            "height": 512,
            "steps": 8,
            "seed": 1234,
        }
        state.record_recoverable_job("test-job-api", req_data, reason="interrupted")
        res = client.get("/api/queue/recovery")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("items", data)
        self.assertTrue(any(x["id"] == "test-job-api" for x in data["items"]))

    def test_api_queue_recovery_restore(self):
        req_data = {
            "prompt": "restored prompt test",
            "model": "flux2-klein-4b",
            "width": 512,
            "height": 512,
            "steps": 4,
            "seed": 999,
        }
        state.record_recoverable_job("restore-test-id", req_data, reason="cancelled")
        res = client.post("/api/queue/recovery/restore", json={"job_ids": ["restore-test-id"]})
        self.assertEqual(res.status_code, 200)
        result = res.json()
        self.assertEqual(result["count"], 1)
        # Verify it was removed from recovery
        items = state.get_recoverable_jobs()
        self.assertFalse(any(x["id"] == "restore-test-id" for x in items))

    def test_delete_recovery(self):
        req_data = {"prompt": "to delete", "model": "flux2-klein-4b"}
        state.record_recoverable_job("del-1", req_data, reason="cancelled")
        state.record_recoverable_job("del-2", req_data, reason="cancelled")
        # Delete specific
        client.delete("/api/queue/recovery?job_id=del-1")
        items = state.get_recoverable_jobs()
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["id"], "del-2")
        # Delete all
        client.delete("/api/queue/recovery")
        self.assertEqual(len(state.get_recoverable_jobs()), 0)

if __name__ == "__main__":
    unittest.main()
