"""Download one locked GGUF and verify size/SHA256 before use."""

import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=ROOT / "models")
    args = parser.parse_args()
    lock = json.loads((ROOT / "locks/model.json").read_text(encoding="utf-8"))
    args.dir.mkdir(parents=True, exist_ok=True)
    target = args.dir / lock["filename"]
    if not target.exists():
        partial = target.with_suffix(target.suffix + ".part")
        url = "https://huggingface.co/{}/resolve/{}/{}".format(
            lock["repo_id"], lock["revision"], lock["filename"])
        with urllib.request.urlopen(url, timeout=60) as response, partial.open("wb") as stream:
            while block := response.read(1024 * 1024):
                stream.write(block)
        if partial.stat().st_size != lock["size_bytes"] or digest(partial) != lock["sha256"]:
            raise SystemExit("Download checksum mismatch; partial file preserved, model not accepted.")
        partial.replace(target)
    if target.stat().st_size != lock["size_bytes"] or digest(target) != lock["sha256"]:
        raise SystemExit("Existing model differs from lock. Preserve it and use a different directory.")
    print(json.dumps({"path": str(target), "sha256": lock["sha256"], "verified": True}))


if __name__ == "__main__":
    main()
