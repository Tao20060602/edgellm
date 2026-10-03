#!/usr/bin/env python3
"""Fetch and verify the pinned Wikitext-2 test member without extracting a ZIP."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCK = ROOT / "locks" / "quality.json"
MAX_DOWNLOAD_SECONDS = 60
CHUNK_BYTES = 1024 * 1024


def read_lock(path: Path = DEFAULT_LOCK) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    dataset = value["dataset"]
    for key in ("archive_url", "archive_sha256", "archive_bytes", "member", "member_sha256", "member_bytes"):
        if key not in dataset:
            raise ValueError(f"quality lock is missing dataset.{key}")
    return dataset


def download_archive(dataset: dict[str, Any], target) -> dict[str, Any]:
    digest = hashlib.sha256()
    byte_count = 0
    request = urllib.request.Request(dataset["archive_url"], headers={"User-Agent": "EdgeLLM-quality-fetch/1"})
    with urllib.request.urlopen(request, timeout=MAX_DOWNLOAD_SECONDS) as response:
        length_header = response.headers.get("Content-Length")
        if length_header is not None and int(length_header) != dataset["archive_bytes"]:
            raise ValueError(f"archive Content-Length mismatch: expected {dataset['archive_bytes']}, got {length_header}")
        while True:
            block = response.read(CHUNK_BYTES)
            if not block:
                break
            byte_count += len(block)
            if byte_count > dataset["archive_bytes"]:
                raise ValueError("archive is larger than the locked byte length")
            digest.update(block)
            target.write(block)
    actual = digest.hexdigest()
    if byte_count != dataset["archive_bytes"] or actual != dataset["archive_sha256"]:
        raise ValueError(
            f"archive verification failed: expected {dataset['archive_bytes']} bytes/{dataset['archive_sha256']}, "
            f"got {byte_count}/{actual}"
        )
    return {"url": dataset["archive_url"], "bytes": byte_count, "sha256": actual}


def read_verified_member(archive, dataset: dict[str, Any]) -> tuple[bytes, dict[str, Any]]:
    with zipfile.ZipFile(archive, "r") as zipped:
        matches = [entry for entry in zipped.infolist() if entry.filename == dataset["member"]]
        if len(matches) != 1:
            raise ValueError(f"expected exactly one ZIP member {dataset['member']!r}, found {len(matches)}")
        entry = matches[0]
        if entry.is_dir() or entry.flag_bits & 0x1:
            raise ValueError("locked ZIP member is a directory or encrypted")
        if entry.file_size != dataset["member_bytes"]:
            raise ValueError(f"member byte length mismatch: expected {dataset['member_bytes']}, got {entry.file_size}")
        digest = hashlib.sha256()
        content = bytearray()
        with zipped.open(entry, "r") as stream:
            while True:
                block = stream.read(CHUNK_BYTES)
                if not block:
                    break
                content.extend(block)
                if len(content) > dataset["member_bytes"]:
                    raise ValueError("member expanded beyond its locked byte length")
                digest.update(block)
        actual = digest.hexdigest()
        if len(content) != dataset["member_bytes"] or actual != dataset["member_sha256"]:
            raise ValueError(
                f"member verification failed: expected {dataset['member_bytes']} bytes/{dataset['member_sha256']}, "
                f"got {len(content)}/{actual}"
            )
        return bytes(content), {"member": entry.filename, "bytes": len(content), "sha256": actual}


def fetch(output: Path, lock_path: Path = DEFAULT_LOCK) -> dict[str, Any]:
    dataset = read_lock(lock_path)
    output = output.expanduser().resolve(strict=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing dataset: {output}")

    with tempfile.TemporaryFile(mode="w+b") as archive:
        archive_info = download_archive(dataset, archive)
        archive.seek(0)
        content, member_info = read_verified_member(archive, dataset)

    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(output, flags, 0o644)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        try:
            output.unlink()
        except OSError:
            pass
        raise
    return {"output": str(output), "archive": archive_info, "dataset": member_info}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="new local path for wiki.test.raw; existing files are never overwritten")
    parser.add_argument("--lock", default=str(DEFAULT_LOCK), help="quality lock JSON (defaults to this repository's lock)")
    args = parser.parse_args(argv)
    try:
        result = fetch(Path(args.out), Path(args.lock))
    except Exception as exc:
        print(f"dataset fetch failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
