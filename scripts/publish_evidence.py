"""Copy reviewed runs into a new public evidence tree; preserve originals."""

import argparse
import json
import shutil
from pathlib import Path

from bench import sha256_file, utc_now, write_json


def redact_hostname(value):
    if isinstance(value, dict):
        return {key: "redacted" if key == "hostname" else redact_hostname(item)
                for key, item in value.items()}
    if isinstance(value, list):
        return [redact_hostname(item) for item in value]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", required=True, nargs="+", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    publication = {"at": utc_now(), "script_sha256": sha256_file(Path(__file__)),
                   "scope": "only metadata hostname redacted; raw streams unchanged; profiler trace files excluded",
                   "runs": [], "redactions": [], "excluded": []}
    for source in args.runs:
        destination = args.out / source.name
        if destination.exists():
            raise ValueError("duplicate run name")
        def ignore(directory, names):
            excluded = [name for name in names if name.endswith((".nsys-rep", ".sqlite", ".qdstrm"))]
            for name in excluded:
                path = Path(directory) / name
                publication["excluded"].append({"source": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size})
            return excluded
        shutil.copytree(source, destination, ignore=ignore)
        publication["runs"].append({"source": str(source), "public_directory": source.name})
        for path in destination.rglob("metadata.json"):
            original = json.loads(path.read_text())
            redacted = redact_hostname(original)
            if original != redacted:
                before = sha256_file(path)
                write_json(path, redacted)
                publication["redactions"].append({"file": path.relative_to(args.out).as_posix(),
                                                  "original_sha256": before, "public_sha256": sha256_file(path)})
    write_json(args.out / "PUBLICATION.json", publication)
    manifest = "".join(f"{sha256_file(path)}  {path.relative_to(args.out).as_posix()}\n"
                       for path in sorted(args.out.rglob("*")) if path.is_file() and path.name != "SHA256SUMS")
    (args.out / "SHA256SUMS").write_text(manifest, encoding="utf-8", newline="\n")
    print(f"Published copy: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
