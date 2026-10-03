"""Warm localhost TTFT and a fixed synthetic retrieval screen; no external service."""

import argparse
import gzip
import hashlib
import json
import math
import random
import re
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from bench import sha256_file, run_git, cmake_cache_info, shared_library_info, utc_now, write_json

ROOT = Path(__file__).resolve().parents[1]


def http_json(base, route, payload=None, timeout=60):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(base + route, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def consume_sse(lines, started, now=time.perf_counter):
    content, first, final = [], None, None
    for raw in lines:
        line = raw.decode("utf-8").strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            continue
        event = json.loads(data)
        if "error" in event:
            raise ValueError(f"server error: {event['error']}")
        text = event.get("content", "")
        if not isinstance(text, str):
            raise ValueError("non-string streamed content")
        if text:
            if first is None:
                first = now() - started
            content.append(text)
        if event.get("stop") is True:
            final = event
    elapsed = now() - started
    if first is None or final is None:
        raise ValueError("stream missing content or terminal stop event")
    timings = final.get("timings")
    if not isinstance(timings, dict) or timings.get("prompt_n", 0) <= 0:
        raise ValueError("stream missing prompt timing evidence")
    if timings.get("cache_n") != 0:
        raise ValueError(f"unexpected prompt reuse: cache_n={timings.get('cache_n')}")
    if final.get("truncated"):
        raise ValueError("prompt was truncated")
    if not math.isfinite(first) or first < 0:
        raise ValueError("invalid TTFT")
    return {"content": "".join(content), "ttft_seconds": first,
            "request_seconds": elapsed, "server_timings": timings, "final_event": final}


def stream_request(base, prompt, out, case_id, timeout, spec):
    prompt_bytes = prompt.encode("utf-8")
    prompt_file = case_id + ".prompt.txt.gz"
    (out / prompt_file).write_bytes(gzip.compress(prompt_bytes, mtime=0))
    params = {"n_predict": spec["max_generation_tokens"], "temperature": 0,
              "seed": spec["seed"], "cache_prompt": False, "stream": True,
              "return_tokens": True, "id_slot": 0}
    record = {"case_id": case_id, "prompt_sha256": hashlib.sha256(prompt_bytes).hexdigest(),
              "prompt_file": prompt_file, "request_parameters": params, "started_at": utc_now()}
    write_json(out / (case_id + ".request.json"), record)
    payload = dict(params, prompt=prompt)
    request = urllib.request.Request(base + "/completion", data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
    lines = []
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            def recorded_lines():
                for line in response:
                    lines.append(line)
                    yield line
            result = consume_sse(recorded_lines(), started)
    finally:
        (out / (case_id + ".sse.bin")).write_bytes(b"".join(lines))
    result.update(record)
    write_json(out / (case_id + ".result.json"), result)
    return result


def prompt_for(base, spec, target, position, code=None):
    def tokens(text):
        return http_json(base, "/tokenize", {"content": text, "parse_special": True})["tokens"]
    filler_ids = tokens(spec["filler"])
    needle = spec["needle"].format(code=code) if code else ""
    budget = max(1, target - len(tokens(needle)))
    repeated = (filler_ids * (budget // len(filler_ids) + 1))[:budget]
    split = int(budget * position)
    before = http_json(base, "/detokenize", {"tokens": repeated[:split]})["content"]
    after = http_json(base, "/detokenize", {"tokens": repeated[split:]})["content"]
    question = spec["question"] if code else spec["ttft_question"]
    text = "Read this archive and answer the final question.\n" + before + needle + after + "\n" + question
    formatted = http_json(base, "/apply-template", {
        "messages": [{"role": "user", "content": text}],
        "chat_template_kwargs": {"enable_thinking": False}})["prompt"]
    return formatted, len(tokens(formatted))


def process_memory(pid):
    path = Path("/proc") / str(pid) / "status"
    if not path.is_file():
        return None
    values = {}
    for line in path.read_text().splitlines():
        if line.startswith(("VmRSS:", "VmHWM:")):
            values[line.split(":")[0] + "_kib"] = int(line.split()[1])
    return {"scope": "owned server process since startup, observed after requests", **values}


def device_snapshot():
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,temperature.gpu,power.draw,utilization.gpu",
                                 "--format=csv,noheader"], capture_output=True, timeout=10)
        return {"scope": "whole device snapshot, not process peak", "at": utc_now(),
                "returncode": result.returncode, "stdout": result.stdout.decode(errors="replace")}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"scope": "whole device snapshot", "unavailable": str(exc)}


def evaluate_gate(results, spec):
    base = results["f16"]["retrieval"]
    baseline = sum(row["correct"] for row in base)
    gates = {}
    for kv, data in results.items():
        if [row["prompt_sha256"] for row in data["retrieval"]] != [row["prompt_sha256"] for row in base]:
            raise ValueError("retrieval prompts differ between KV formats")
        correct = sum(row["correct"] for row in data["retrieval"])
        gates[kv] = {"correct": correct, "total": len(base),
                     "extra_failures_vs_f16": baseline - correct,
                     "pass": baseline >= spec["baseline_min_correct"] and
                     baseline - correct <= spec["allowed_extra_failures"]}
    return {"baseline_correct": baseline, "baseline_sufficient": baseline >= spec["baseline_min_correct"],
            "formats": gates, "pass": all(item["pass"] for item in gates.values())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("binary", "model", "upstream", "out"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--suite", type=Path, default=ROOT / "eval/server_suite.json")
    parser.add_argument("--ctx", type=int, default=33792)
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    args.binary, args.model, args.upstream, args.out = [p.resolve() for p in (args.binary, args.model, args.upstream, args.out)]
    try:
        args.out.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        raise SystemExit("Output directory already exists; refusing to overwrite")
    spec = json.loads(args.suite.read_text())
    metadata = {"schema_version": 1, "status": "starting", "started_at": utc_now(),
                "source_sha": run_git(args.upstream, "rev-parse", "HEAD"),
                "source_dirty": bool(run_git(args.upstream, "status", "--porcelain")),
                "binary_sha256": sha256_file(args.binary), "model_sha256": sha256_file(args.model),
                "harness_sha256": sha256_file(Path(__file__)), "suite_sha256": sha256_file(args.suite),
                "build": cmake_cache_info(args.binary), "libraries": shared_library_info(args.binary),
                "limits": ["warm localhost requests; no model-load latency", "synthetic single-needle diagnostic"]}
    write_json(args.out / "metadata.json", metadata)
    write_json(args.out / "suite.json", spec)
    formats = ["f16", "q8_0", "q4_0"]
    random.Random(spec["seed"]).shuffle(formats)
    all_results = {}
    proc = None
    try:
        for kv in formats:
            kv_dir = args.out / kv
            kv_dir.mkdir()
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            base = f"http://127.0.0.1:{port}"
            argv = [str(args.binary), "-v", "-m", str(args.model), "-c", str(args.ctx), "-ngl", "99", "-fa", "on",
                    "-ctk", kv, "-ctv", kv, "-b", "512", "-ub", "512", "-t", "4", "-np", "1",
                    "--host", "127.0.0.1", "--port", str(port), "--no-context-shift",
                    "--chat-template-kwargs", '{"enable_thinking":false}', "--reasoning-budget", "0"]
            write_json(kv_dir / "launch.json", {"argv": argv, "cwd": str(args.upstream), "device_before": device_snapshot()})
            with (kv_dir / "server.stdout.bin").open("wb") as stdout, (kv_dir / "server.stderr.bin").open("wb") as stderr:
                proc = subprocess.Popen(argv, cwd=args.upstream, stdout=stdout, stderr=stderr)
                deadline = time.monotonic() + args.timeout
                while True:
                    if proc.poll() is not None:
                        raise RuntimeError(f"server exited: {proc.returncode}")
                    try:
                        http_json(base, "/health", timeout=2)
                        break
                    except (OSError, urllib.error.HTTPError):
                        if time.monotonic() > deadline:
                            raise TimeoutError("server startup timeout")
                        time.sleep(0.1)
                log = (kv_dir / "server.stderr.bin").read_text(errors="replace")
                offloads = re.findall(r"offloaded\s+(\d+)/(\d+)\s+layers", log)
                if not offloads or int(offloads[-1][0]) == 0 or offloads[-1][0] != offloads[-1][1] or not re.search(r"CUDA\d+\s+KV buffer", log):
                    raise ValueError("missing full CUDA offload/KV allocation proof")
                warm = http_json(base, "/completion", {"prompt": "Say hello. /no_think", "n_predict": 8, "cache_prompt": False, "temperature": 0}, timeout=args.timeout)
                write_json(kv_dir / "warmup.json", warm)
                latency_prompt, latency_tokens = prompt_for(base, spec, spec["ttft_content_token_target"], 0.5)
                latency = [stream_request(base, latency_prompt, kv_dir, f"ttft-{i}", args.timeout, spec)
                           for i in range(spec["ttft_repetitions"])]
                retrieval = []
                index = 0
                for target in spec["content_token_targets"]:
                    for position in spec["needle_positions"]:
                        code = spec["passcodes"][index]
                        prompt, count = prompt_for(base, spec, target, position, code)
                        if count + spec["max_generation_tokens"] >= args.ctx:
                            raise ValueError("prompt does not fit without truncation")
                        result = stream_request(base, prompt, kv_dir, f"retrieval-{index}", args.timeout, spec)
                        result.update({"target_content_tokens": target, "actual_prompt_tokens": count,
                                       "needle_position": position, "expected": code,
                                       "correct": code in re.findall(r"\b\d{6}\b", result["content"])})
                        write_json(kv_dir / (f"retrieval-{index}.result.json"), result)
                        retrieval.append(result)
                        index += 1
                all_results[kv] = {"latency": latency, "latency_prompt_tokens": latency_tokens,
                                   "retrieval": retrieval, "host_memory": process_memory(proc.pid),
                                   "device_after": device_snapshot(), "full_offload": offloads[-1]}
                write_json(kv_dir / "summary.json", all_results[kv])
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)
                proc = None
        gate = evaluate_gate(all_results, spec)
        summary = {"status": "complete", "gate": gate, "results": all_results,
                   "scope": spec["scope"], "ended_at": utc_now()}
        write_json(args.out / "summary.json", summary)
        metadata.update(status="complete", quality_gate_pass=gate["pass"], ended_at=utc_now())
        write_json(args.out / "metadata.json", metadata)
        print(json.dumps(gate))
        return 0 if gate["pass"] else 3
    except Exception as exc:
        write_json(args.out / "failure.json", {"at": utc_now(), "error": str(exc), "completed_formats": list(all_results)})
        metadata.update(status="failed", error=str(exc), ended_at=utc_now())
        write_json(args.out / "metadata.json", metadata)
        print(str(exc))
        return 1
    finally:
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)


if __name__ == "__main__":
    raise SystemExit(main())
