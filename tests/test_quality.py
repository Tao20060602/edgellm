import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import quality


FAKE_PERPLEXITY = r'''#!/usr/bin/env python3
import os
import sys
import time

args = sys.argv[1:]
def option(name):
    return args[args.index(name) + 1]

mode = os.environ.get("FAKE_QUALITY_MODE", "success")
kv = option("-ctk")
ppl = {"f16": 10.0, "q8_0": 10.1, "q4_0": 10.5}[kv]
if mode == "threshold":
    ppl = 10.3 if kv == "q8_0" else ppl
if mode == "timeout":
    print("partial log", file=sys.stderr, flush=True)
    time.sleep(10)
if mode == "badparse":
    print("perplexity: output was truncated", file=sys.stderr)
    raise SystemExit(0)
if mode == "nonfinite":
    ppl_text = "nan"
else:
    ppl_text = "%.4f" % ppl
if mode == "wrong_chunks":
    chunks = 7
else:
    chunks = int(option("--chunks"))
actual_kv = "q8_0" if mode == "wrong_kv" else kv
probe_offload = 28 if mode == "inconsistent_offload" else 29
actual_buffer = 0.0 if mode == "zero_actual_kv" else 224.0
print("llama_model_loader: loaded model", file=sys.stderr)
print("llama_model_load: offloaded %d/29 layers to GPU" % probe_offload, file=sys.stderr)
print("llama_kv_cache:      CUDA0 KV buffer size =     0.00 MiB", file=sys.stderr)
print("llama_kv_cache: size = 224.00 MiB, K (f16): 112.00 MiB, V (f16): 112.00 MiB", file=sys.stderr)
print("llama_model_load: offloaded 29/29 layers to GPU", file=sys.stderr)
print("llama_kv_cache:      CUDA0 KV buffer size = %8.2f MiB" % actual_buffer, file=sys.stderr)
print("llama_kv_cache: size = 224.00 MiB, K (%s): 112.00 MiB, V (%s): 112.00 MiB" % (actual_kv, actual_kv), file=sys.stderr)
print("perplexity: calculating perplexity over %d chunks, n_ctx=2048, batch_size=512, n_seq=1" % chunks, file=sys.stderr)
print("Final estimate: PPL = %s +/- 0.02000" % ppl_text, file=sys.stderr)
'''


class QualityHarnessTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.upstream = self.root / "upstream"
        self.upstream.mkdir()
        self.binary = self.root / ("fake-perplexity.exe" if os.name == "nt" else "fake-perplexity")
        self.binary.write_text(FAKE_PERPLEXITY, encoding="utf-8", newline="\n")
        if os.name == "posix":
            self.binary.chmod(self.binary.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        self.model = self.root / "fixture.gguf"
        self.model.write_bytes(b"test model placeholder")
        self.dataset = self.root / "wiki.test.raw"
        dataset_lock = quality.load_json(quality.QUALITY_LOCK)["dataset"]
        self.dataset.write_bytes(b"x" * dataset_lock["member_bytes"])
        self.expected_model_sha = quality.load_json(quality.MODEL_LOCK)["sha256"]

    def tearDown(self):
        self.temp.cleanup()

    def mock_process(self, command, **kwargs):
        mode = os.environ.get("FAKE_QUALITY_MODE", "success")
        if mode == "timeout":
            raise subprocess.TimeoutExpired(command, kwargs["timeout"], output=b"partial stdout", stderr=b"partial stderr")
        if mode == "badparse":
            return subprocess.CompletedProcess(command, 0, b"", b"truncated output\n")

        kv = command[command.index("-ctk") + 1]
        ppl = {"f16": 10.0, "q8_0": 10.1, "q4_0": 10.5}[kv]
        if mode == "threshold" and kv == "q8_0":
            ppl = 10.3
        chunks = 7 if mode == "wrong_chunks" else 8
        actual_kv = "q8_0" if mode == "wrong_kv" else kv
        probe_offload = 28 if mode == "inconsistent_offload" else 29
        actual_buffer = 0.0 if mode == "zero_actual_kv" else 224.0
        ppl_text = "nan" if mode == "nonfinite" else f"{ppl:.4f}"
        logs = (
            f"llama_model_load: offloaded {probe_offload}/29 layers to GPU\n"
            "llama_kv_cache:      CUDA0 KV buffer size = 0.00 MiB\n"
            "llama_kv_cache: size = 224.00 MiB, K (f16): 112.00 MiB, V (f16): 112.00 MiB\n"
            "llama_model_load: offloaded 29/29 layers to GPU\n"
            f"llama_kv_cache:      CUDA0 KV buffer size = {actual_buffer:.2f} MiB\n"
            f"llama_kv_cache: size = 224.00 MiB, K ({actual_kv}): 112.00 MiB, V ({actual_kv}): 112.00 MiB\n"
            f"perplexity: calculating perplexity over {chunks} chunks, n_ctx=2048, batch_size=512, n_seq=1\n"
            f"Final estimate: PPL = {ppl_text} +/- 0.02000\n"
        )
        return subprocess.CompletedProcess(command, 0, b"", logs.encode())

    def run_harness(self, output, mode="success", timeout=3):
        argv = [
            "--binary", str(self.binary),
            "--model", str(self.model),
            "--upstream", str(self.upstream),
            "--dataset", str(self.dataset),
            "--out", str(output),
            "--timeout", str(timeout),
        ]
        real_sha256 = quality.bench.sha256_file

        def sha256_for_fixture(path):
            resolved = Path(path).resolve()
            if resolved == self.model.resolve():
                return self.expected_model_sha
            if resolved == self.dataset.resolve():
                return quality.load_json(quality.QUALITY_LOCK)["dataset"]["member_sha256"]
            return real_sha256(path)

        def run_git(_upstream, *args):
            if args == ("rev-parse", "HEAD"):
                return quality.load_json(quality.UPSTREAM_LOCK)["commit"]
            if args == ("status", "--porcelain"):
                return ""
            raise AssertionError(f"unexpected git args: {args}")

        with patch.dict(os.environ, {"FAKE_QUALITY_MODE": mode}), \
             patch.object(quality.bench, "sha256_file", side_effect=sha256_for_fixture), \
             patch.object(quality.bench, "run_git", side_effect=run_git):
            if os.name == "nt":
                with patch.object(quality.subprocess, "run", side_effect=self.mock_process):
                    return quality.main(argv)
            if mode == "timeout":
                with patch.object(quality.subprocess, "run", side_effect=self.mock_process):
                    return quality.main(argv)
            return quality.main(argv)

    def test_success_retains_exact_plan_raw_stream_hashes_and_runtime_evidence(self):
        output = self.root / "success"
        self.assertEqual(self.run_harness(output), 0)
        plan = json.loads((output / "plan.json").read_text(encoding="utf-8"))
        summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["status"], "success")
        self.assertEqual([case["kv_type"] for case in plan["cases"]], ["f16", "q8_0", "q4_0"])
        first = plan["cases"][0]
        self.assertEqual(first["argv"][first["argv"].index("--chunks") + 1], "8")
        self.assertEqual(first["argv"][first["argv"].index("-fa") + 1], "on")
        self.assertEqual(first["argv"][first["argv"].index("-b") + 1], "512")
        self.assertIn("-v", first["argv"])
        self.assertEqual(summary["measurement_scope"]["scored_next_token_targets"], 8184)
        self.assertAlmostEqual(summary["gate"]["q8_0"]["relative_ppl_increase_percent"], 1.0)
        case = json.loads((output / "cases" / "case-01-f16.json").read_text(encoding="utf-8"))
        self.assertIsNone(case["parsed"]["input_token_count"])
        self.assertEqual(case["parsed"]["input_token_count_status"], "not_reported_by_pinned_upstream")
        self.assertEqual(case["parsed"]["chunk_count"], 8)
        self.assertEqual(case["parsed"]["gpu_layer_offload"]["offloaded"], 29)
        self.assertEqual(len(case["parsed"]["gpu_layer_offload"]["summaries"]), 2)
        self.assertEqual(case["parsed"]["gpu_layer_offload"]["summaries"][-1]["phase"], "ppl_context")
        self.assertEqual(len(case["parsed"]["runtime_kv_types"]), 2)
        self.assertEqual(case["parsed"]["runtime_kv_types"][0]["phase"], "fit_probe")
        self.assertEqual(case["parsed"]["actual_ppl_context_kv_types"]["key"], "f16")
        self.assertEqual(case["parsed"]["runtime_kv_buffer_sizes"][0]["mib"], 0.0)
        self.assertEqual(case["parsed"]["actual_cuda_kv_buffers"][0]["mib"], 224.0)
        self.assertEqual(case["parsed"]["cuda_device_count_logs"], [])
        self.assertEqual(case["parsed"]["cuda_runtime_evidence"], ["all_requested_model_layers_offloaded", "nonzero_materialized_cuda_kv_buffer"])
        self.assertEqual(case["parsed"]["runtime_kv_types"][0]["key"], "f16")
        self.assertEqual(case["parsed"]["ppl_source_stream"], "stderr")
        self.assertEqual(case["stdout_sha256"], quality.bench.sha256_file(output / case["stdout_path"]))
        self.assertTrue(case["stderr_sha256"])

    def test_threshold_miss_is_a_complete_negative_evaluation_not_invalid_execution(self):
        output = self.root / "threshold"
        self.assertEqual(self.run_harness(output, mode="threshold"), 1)
        metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
        summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        failure = json.loads((output / "failure.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["status"], "evaluation_failed")
        self.assertEqual(summary["status"], "evaluation_failed")
        self.assertFalse(summary["gate"]["q8_0"]["passed"])
        self.assertEqual(failure["status"], "evaluation_failed")
        self.assertEqual(len(summary["cases"]), 3)

    def test_missing_result_or_chunk_mismatch_fails_closed_and_preserves_case_logs(self):
        for mode in ("badparse", "wrong_chunks", "nonfinite", "wrong_kv", "inconsistent_offload", "zero_actual_kv"):
            with self.subTest(mode=mode):
                output = self.root / mode
                self.assertEqual(self.run_harness(output, mode=mode), 1)
                metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
                failure = json.loads((output / "failure.json").read_text(encoding="utf-8"))
                case = json.loads((output / "cases" / "case-01-f16.json").read_text(encoding="utf-8"))
                self.assertEqual(metadata["status"], "invalid_execution")
                self.assertEqual(failure["status"], "invalid_execution")
                self.assertEqual(case["status"], "invalid_execution")
                self.assertTrue(case["stderr_sha256"])
                if mode == "nonfinite":
                    self.assertIn("non_finite_or_invalid_ppl", case["failure_reason"])
                if mode == "wrong_kv":
                    self.assertIn("PPL-context K/V type", case["failure_reason"])
                if mode == "inconsistent_offload":
                    self.assertIn("offload_mismatch", case["failure_reason"])
                if mode == "zero_actual_kv":
                    self.assertIn("actual_cuda_kv_buffer_missing_or_empty", case["failure_reason"])
                self.assertFalse((output / "summary.json").exists())

    def test_timeout_keeps_partial_stdout_and_stderr_and_stops_after_first_case(self):
        output = self.root / "timeout"
        self.assertEqual(self.run_harness(output, mode="timeout", timeout=1), 1)
        case = json.loads((output / "cases" / "case-01-f16.json").read_text(encoding="utf-8"))
        self.assertEqual(case["status"], "timeout")
        self.assertEqual((output / case["stdout_path"]).read_bytes(), b"partial stdout")
        self.assertEqual((output / case["stderr_path"]).read_bytes(), b"partial stderr")
        self.assertEqual(len(list((output / "cases").glob("*.json"))), 1)

    def test_existing_output_path_is_rejected_without_modification(self):
        output = self.root / "existing"
        output.mkdir()
        sentinel = output / "keep.txt"
        sentinel.write_text("keep", encoding="utf-8")
        self.assertEqual(self.run_harness(output), 2)
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")


if __name__ == "__main__":
    unittest.main()
