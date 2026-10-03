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
import bench


FAKE_BENCH = r'''#!/usr/bin/env python3
import json
import os
import sys
import time

args = sys.argv[1:]
def option(name):
    return args[args.index(name) + 1]

mode = os.environ.get("FAKE_BENCH_MODE", "success")
if mode == "timeout":
    print("partial", flush=True)
    time.sleep(10)
    raise SystemExit(0)
if mode == "invalid_json":
    print("this is not json")
    print("llama_kv_cache: CPU KV buffer size = 12.50 MiB", file=sys.stderr)
    raise SystemExit(0)

row = {
    "n_prompt": int(option("-p")),
    "n_gen": int(option("-n")),
    "n_depth": int(option("-d")),
    "n_batch": int(option("-b")),
    "n_ubatch": int(option("-ub")),
    "type_k": option("-ctk"),
    "type_v": option("-ctv"),
    "n_gpu_layers": int(option("-ngl")),
    "n_threads": int(option("-t")),
    "flash_attn": 1,
    "backends": "CPU",
    "avg_ts": 42.5,
    "stddev_ts": 0.25,
    "samples_ts": [42.25, 42.75][:int(option("-r"))],
    "samples_ns": [1000000, 1100000][:int(option("-r"))],
}
if mode == "wrong_depth":
    row["n_depth"] += 1
elif mode == "wrong_flash_attn":
    row["flash_attn"] = -1
elif mode == "zero_avg":
    row["avg_ts"] = 0
elif mode == "wrong_prefill":
    row.update(n_prompt=int(option("-d")), n_gen=0, n_depth=0)
elif mode == "wrong_decode":
    row.update(n_prompt=0, n_gen=8, n_depth=int(option("-p")))
elif mode == "short_samples":
    row["samples_ts"] = row["samples_ts"][:-1]
elif mode == "cuda_without_evidence":
    row["backends"] = "CUDA"

print(json.dumps([row]))
print("llama_kv_cache: CPU KV buffer size = 12.50 MiB", file=sys.stderr)
'''


@unittest.skipUnless(shutil.which("git"), "git is required for provenance tests")
class BenchHarnessTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.upstream = self.root / "upstream"
        self.upstream.mkdir()
        self.git(["init", "-q"])
        self.git(["config", "user.name", "Bench Test"])
        self.git(["config", "user.email", "bench@example.invalid"])
        (self.upstream / "README.md").write_text("fixture\n", encoding="utf-8")
        self.git(["add", "README.md"])
        self.git(["commit", "-q", "-m", "fixture"])
        self.binary = self.root / "fake-llama-bench"
        self.binary.write_text(FAKE_BENCH, encoding="utf-8", newline="\n")
        if os.name == "posix":
            self.binary.chmod(self.binary.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        self.model = self.root / "fixture.gguf"
        self.model.write_bytes(b"fixture model bytes")

    def tearDown(self):
        self.temp.cleanup()

    def git(self, args):
        return subprocess.run(["git", "-C", str(self.upstream), *args], check=True, capture_output=True).stdout.decode().strip()

    def fake_windows_run(self, command, **kwargs):
        if command[0] == "git":
            return self.real_subprocess_run(command, **kwargs)
        mode = os.environ.get("FAKE_BENCH_MODE", "success")
        if mode == "timeout":
            raise subprocess.TimeoutExpired(command, kwargs["timeout"], output=b"partial\n", stderr=b"")
        if mode == "invalid_json":
            return subprocess.CompletedProcess(command, 0, b"this is not json\n", b"llama_kv_cache: CPU KV buffer size = 12.50 MiB\n")

        def option(name):
            return command[command.index(name) + 1]

        row = {
            "n_prompt": int(option("-p")),
            "n_gen": int(option("-n")),
            "n_depth": int(option("-d")),
            "n_batch": int(option("-b")),
            "n_ubatch": int(option("-ub")),
            "type_k": option("-ctk"),
            "type_v": option("-ctv"),
            "n_gpu_layers": int(option("-ngl")),
            "n_threads": int(option("-t")),
            "flash_attn": 1,
            "backends": "CPU",
            "avg_ts": 42.5,
            "stddev_ts": 0.25,
            "samples_ts": [42.25, 42.75][:int(option("-r"))],
            "samples_ns": [1000000, 1100000][:int(option("-r"))],
        }
        if mode == "wrong_depth":
            row["n_depth"] += 1
        elif mode == "wrong_flash_attn":
            row["flash_attn"] = -1
        elif mode == "zero_avg":
            row["avg_ts"] = 0
        elif mode == "wrong_prefill":
            row.update(n_prompt=int(option("-d")), n_gen=0, n_depth=0)
        elif mode == "wrong_decode":
            row.update(n_prompt=0, n_gen=8, n_depth=int(option("-p")))
        elif mode == "short_samples":
            row["samples_ts"] = row["samples_ts"][:-1]
        elif mode == "cuda_without_evidence":
            row["backends"] = "CUDA"
        return subprocess.CompletedProcess(
            command,
            0,
            (json.dumps([row]) + "\n").encode(),
            b"llama_kv_cache: CPU KV buffer size = 12.50 MiB\n",
        )

    def run_harness(
        self,
        out,
        mode="success",
        *,
        dry_run=False,
        depths="0",
        kv_types="f16",
        timeout="2",
        benchmark_mode="decode",
        require_backend="any",
        monitor_memory=False,
    ):
        argv = [
            "--binary", str(self.binary),
            "--model", str(self.model),
            "--upstream", str(self.upstream),
            "--out", str(out),
            "--depths", depths,
            "--kv-types", kv_types,
            "--gpu-layers", "0",
            "--threads", "4",
            "--repetitions", "2",
            "--gen", "8",
            "--timeout", timeout,
            "--seed", "123",
            "--mode", benchmark_mode,
            "--batch", "512",
            "--ubatch", "512",
            "--require-backend", require_backend,
        ]
        if monitor_memory:
            argv.append("--monitor-memory")
        if dry_run:
            argv.append("--dry-run")
        with patch.dict(os.environ, {"FAKE_BENCH_MODE": mode}):
            if os.name == "nt":
                self.real_subprocess_run = subprocess.run
                with patch.object(bench.subprocess, "run", side_effect=self.fake_windows_run):
                    return bench.main(argv)
            return bench.main(argv)

    def test_success_preserves_raw_output_provenance_and_exact_command(self):
        out = self.root / "run"
        self.assertEqual(self.run_harness(out, depths="0,16", kv_types="f16,q8_0"), 0)

        metadata = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
        plan = json.loads((out / "plan.json").read_text(encoding="utf-8"))
        summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["status"], "success")
        self.assertEqual(metadata["provenance"]["binary"]["sha256"], bench.sha256_file(self.binary))
        self.assertEqual(metadata["provenance"]["model"]["sha256"], bench.sha256_file(self.model))
        self.assertEqual(metadata["provenance"]["upstream"]["git_sha"], self.git(["rev-parse", "HEAD"]))
        self.assertFalse(metadata["provenance"]["upstream"]["dirty"])
        self.assertEqual(summary["measurement_count"], 4)
        self.assertEqual(len(plan["randomized_order"]), 4)

        command = plan["randomized_order"][0]["argv"]
        self.assertEqual(command[0], str(self.binary.resolve()))
        self.assertEqual(command[command.index("-p") + 1], "0")
        self.assertEqual(command[command.index("-b") + 1], "512")
        self.assertEqual(command[command.index("-ub") + 1], "512")
        self.assertEqual(command[command.index("-fa") + 1], "on")
        self.assertIn("-v", command)
        self.assertEqual(command[command.index("-o") + 1], "json")

        first_case = plan["randomized_order"][0]["case_id"]
        raw_stdout = (out / "cases" / f"{first_case}.stdout.bin").read_bytes()
        raw_stderr = (out / "cases" / f"{first_case}.stderr.bin").read_bytes()
        case = json.loads((out / "cases" / f"{first_case}.json").read_text(encoding="utf-8"))
        self.assertEqual(json.loads(raw_stdout)[0]["backends"], "CPU")
        self.assertEqual(json.loads(raw_stdout)[0]["flash_attn"], 1)
        self.assertEqual(raw_stderr, b"llama_kv_cache: CPU KV buffer size = 12.50 MiB\n")
        self.assertEqual(case["kv_runtime_mib"][0]["mib"], 12.5)
        self.assertEqual(case["kv_runtime_mib"][0]["source"], "stderr")
        self.assertEqual(case["status"], "success")

    def test_dry_run_does_not_require_binary_or_model_and_marks_plan_only(self):
        out = self.root / "plan-only"
        self.binary.unlink()
        self.model.unlink()
        self.assertEqual(self.run_harness(out, dry_run=True, depths="0,512", kv_types="f16,q8_0,q4_0"), 0)

        metadata = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
        plan = json.loads((out / "plan.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["status"], "planning_only")
        self.assertTrue(metadata["planning_only"])
        self.assertIsNone(metadata["provenance"]["binary"]["sha256"])
        self.assertIsNone(metadata["provenance"]["model"]["sha256"])
        self.assertEqual(len(plan["randomized_order"]), 6)
        self.assertFalse((out / "summary.json").exists())

    def test_timeout_fails_closed_and_keeps_partial_raw_output(self):
        out = self.root / "timeout"
        self.assertEqual(self.run_harness(out, mode="timeout", timeout="1"), 1)

        failure = json.loads((out / "failure.json").read_text(encoding="utf-8"))
        plan = json.loads((out / "plan.json").read_text(encoding="utf-8"))
        case_id = plan["randomized_order"][0]["case_id"]
        case = json.loads((out / "cases" / f"{case_id}.json").read_text(encoding="utf-8"))
        self.assertEqual(failure["stage"], "case")
        self.assertIn("timeout_after_seconds", failure["reason"])
        self.assertEqual(case["status"], "timeout")
        self.assertIn(b"partial", (out / "cases" / f"{case_id}.stdout.bin").read_bytes())
        self.assertFalse((out / "summary.json").exists())

    def test_invalid_json_fails_closed_and_preserves_stderr(self):
        out = self.root / "invalid-json"
        self.assertEqual(self.run_harness(out, mode="invalid_json"), 1)

        failure = json.loads((out / "failure.json").read_text(encoding="utf-8"))
        plan = json.loads((out / "plan.json").read_text(encoding="utf-8"))
        case_id = plan["randomized_order"][0]["case_id"]
        self.assertIn("invalid_json", failure["reason"])
        self.assertIn(b"KV buffer size", (out / "cases" / f"{case_id}.stderr.bin").read_bytes())
        self.assertFalse((out / "summary.json").exists())

    def test_wrong_result_fields_and_nonpositive_rate_fail_closed(self):
        for mode in ("wrong_depth", "wrong_flash_attn", "zero_avg", "short_samples"):
            with self.subTest(mode=mode):
                out = self.root / mode
                self.assertEqual(self.run_harness(out, mode=mode), 1)
                failure = json.loads((out / "failure.json").read_text(encoding="utf-8"))
                self.assertIn("mismatch", failure["reason"])
                self.assertFalse((out / "summary.json").exists())

    def test_prefill_command_and_result_use_prompt_tokens(self):
        out = self.root / "prefill"
        self.assertEqual(self.run_harness(out, depths="16", benchmark_mode="prefill"), 0)
        plan = json.loads((out / "plan.json").read_text(encoding="utf-8"))
        summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
        command = plan["randomized_order"][0]["argv"]
        row = json.loads((out / "cases" / "case-0001.stdout.bin").read_bytes())[0]
        self.assertEqual(command[command.index("-p") + 1], "16")
        self.assertEqual(command[command.index("-n") + 1], "0")
        self.assertEqual(command[command.index("-d") + 1], "0")
        self.assertEqual((row["n_prompt"], row["n_gen"], row["n_depth"]), (16, 0, 0))
        self.assertEqual(summary["measurements"][0]["prompt_tokens"], 16)

    def test_opposite_mode_fields_and_cpu_backend_fallback_are_rejected(self):
        cases = (
            ("decode-prefill-shaped", "wrong_prefill", "decode", "any", "field_mismatch"),
            ("prefill-decode-shaped", "wrong_decode", "prefill", "any", "field_mismatch"),
            ("cuda-cpu-fallback", "success", "decode", "cuda", "backend_mismatch"),
            ("cuda-without-offload-proof", "cuda_without_evidence", "decode", "cuda", "cuda_evidence_missing"),
        )
        for name, fake_mode, mode, backend, expected in cases:
            with self.subTest(name=name):
                out = self.root / name
                depths = "16" if mode == "prefill" or fake_mode == "wrong_prefill" else "0"
                self.assertEqual(
                    self.run_harness(out, mode=fake_mode, depths=depths, benchmark_mode=mode, require_backend=backend),
                    1,
                )
                failure = json.loads((out / "failure.json").read_text(encoding="utf-8"))
                self.assertIn(expected, failure["reason"])
                self.assertFalse((out / "summary.json").exists())

    def test_proc_memory_parser_preserves_missing_values_and_scope(self):
        parsed = bench.parse_proc_memory_status(
            "Name:\tllama-bench\nVmHWM:\t4096 kB\nVmRSS:\t3072 kB\nThreads:\t4\n"
        )
        self.assertEqual(parsed, {"rss_kib": 3072, "vmhwm_kib": 4096})
        missing = bench.parse_proc_memory_status("Name:\tfinished\n")
        self.assertEqual(missing, {"rss_kib": None, "vmhwm_kib": None})
        monitor = bench.summarize_memory_monitor(
            type("Args", (), {"sample_interval_ms": 100})(),
            [{"rss_kib": None, "vmhwm_kib": None, "status": "process_disappeared"}],
            [],
        )
        self.assertIsNone(monitor["sampled_peak_vmrss_kib"])
        self.assertIsNone(monitor["max_observed_vmhwm_kib"])
        self.assertIn("not a process-tree peak", monitor["process_scope"])
        self.assertIn("not KV-specific", monitor["interpretation"])

    def test_cuda_stderr_parsers_capture_device_kv_and_full_offload(self):
        stderr = (
            b"ggml_cuda_init: Device 0: NVIDIA GeForce RTX 3080 Laptop GPU, compute capability 8.6, VMM: yes, VRAM: 8192 MiB\n"
            b"llama_model_load: offloaded 29/29 layers to GPU\n"
            b"llama_kv_cache: CUDA0 KV buffer size = 128.00 MiB\n"
        )
        self.assertEqual(bench.parse_cuda_device_info(stderr)["compute_capability"], "8.6")
        self.assertEqual(bench.parse_cuda_kv_allocation(stderr), 128.0)
        self.assertEqual(bench.parse_offloaded_layers(stderr), [{"offloaded": 29, "total": 29}])

    def test_existing_output_path_is_rejected_without_modifying_it(self):
        out = self.root / "already-there"
        out.mkdir()
        sentinel = out / "keep.txt"
        sentinel.write_text("preserve me", encoding="utf-8")

        self.assertEqual(self.run_harness(out, dry_run=True), 2)
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "preserve me")
        self.assertEqual(sorted(path.name for path in out.iterdir()), ["keep.txt"])


if __name__ == "__main__":
    unittest.main()
