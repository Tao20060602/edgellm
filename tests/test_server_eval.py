import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import server_eval


class StreamingTests(unittest.TestCase):
    def test_changed_model_dirty_or_wrong_upstream_rejected(self):
        metadata = {"source_sha": "upstream", "source_dirty": False, "model_sha256": "model"}
        server_eval.validate_identities(metadata, {"commit": "upstream"}, {"sha256": "model"})
        for change in ({"source_sha": "other"}, {"source_dirty": True}, {"model_sha256": "other"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                server_eval.validate_identities(dict(metadata, **change), {"commit": "upstream"}, {"sha256": "model"})

    def test_ttft_ignores_keepalive_and_empty_events(self):
        rows = [b": ping\n", b"data: {\"content\":\"\"}\n",
                b'data: {"content":"731942"}\n',
                b'data: {"stop":true,"timings":{"prompt_n":2048,"cache_n":0}}\n']
        now = iter([10.25, 10.50])
        result = server_eval.consume_sse(rows, 10.0, now=lambda: next(now))
        self.assertEqual(result["ttft_seconds"], 0.25)
        self.assertEqual(result["content"], "731942")

    def test_incomplete_cached_and_truncated_streams_rejected(self):
        for final in ({}, {"stop": True, "timings": {"prompt_n": 10, "cache_n": 9}},
                      {"stop": True, "truncated": True, "timings": {"prompt_n": 10, "cache_n": 0}}):
            rows = [b'data: {"content":"hello"}\n', ("data: " + json.dumps(final)).encode()]
            with self.subTest(final=final), self.assertRaises(ValueError):
                server_eval.consume_sse(rows, 0, now=lambda: 1)

    def test_gate_requires_adequate_baseline_and_identical_inputs(self):
        spec = {"baseline_min_correct": 8, "allowed_extra_failures": 1}
        def rows(n):
            return [{"correct": i < n, "prompt_sha256": str(i)} for i in range(9)]
        results = {"f16": {"retrieval": rows(8)}, "q4_0": {"retrieval": rows(7)}}
        self.assertTrue(server_eval.evaluate_gate(results, spec)["pass"])
        results["q4_0"]["retrieval"] = rows(6)
        self.assertFalse(server_eval.evaluate_gate(results, spec)["pass"])
        results["f16"]["retrieval"] = rows(7)
        self.assertFalse(server_eval.evaluate_gate(results, spec)["baseline_sufficient"])
        results["q4_0"]["retrieval"][0]["prompt_sha256"] = "different"
        with self.assertRaises(ValueError):
            server_eval.evaluate_gate(results, spec)


if __name__ == "__main__":
    unittest.main()
