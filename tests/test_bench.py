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
    "n_prompt": 0,
    "n_gen": int(option("-n")),
    "n_depth": int(option("-d")),
    "type_k": option("-ctk"),
    "type_v": option("-ctv"),
    "n_gpu_layers": int(option("-ngl")),
    "n_threads": int(option("-t")),
    "flash_attn": 1,
    "backends": "CPU",
    "avg_ts": 42.5,
    "stddev_ts": 0.25,
    "samples_ts": [42.25, 42.75],
}
if mode == "wrong_depth":
    row["n_depth"] += 1
elif mode == "wrong_flash_attn":
    row["flash_attn"] = -1
elif mode == "zero_avg":
    row["avg_ts"] = 0

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
            "n_prompt": 0,
            "n_gen": int(option("-n")),
            "n_depth": int(option("-d")),
            "type_k": option("-ctk"),
            "type_v": option("-ctv"),
            "n_gpu_layers": int(option("-ngl")),
            "n_threads": int(option("-t")),
            "flash_attn": 1,
            "backends": "CPU",
            "avg_ts": 42.5,
            "stddev_ts": 0.25,
            "samples_ts": [42.25, 42.75],
        }
        if mode == "wrong_depth":
            row["n_depth"] += 1
        elif mode == "wrong_flash_attn":
            row["flash_attn"] = -1
        elif mode == "zero_avg":
            row["avg_ts"] = 0
        return subprocess.CompletedProcess(
            command,
            0,
            (json.dumps([row]) + "\n").encode(),
            b"llama_kv_cache: CPU KV buffer size = 12.50 MiB\n",
        )

    def run_harness(self, out, mode="success", *, dry_run=False, depths="0", kv_types="f16", timeout="2"):
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
        ]
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
        for mode in ("wrong_depth", "wrong_flash_attn", "zero_avg"):
            with self.subTest(mode=mode):
                out = self.root / mode
                self.assertEqual(self.run_harness(out, mode=mode), 1)
                failure = json.loads((out / "failure.json").read_text(encoding="utf-8"))
                self.assertIn("field_mismatch", failure["reason"])
                self.assertFalse((out / "summary.json").exists())

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
