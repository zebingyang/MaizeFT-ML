#!/usr/bin/env python3
"""Create a SHA-256 manifest for the publication script package."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    package = args.package_dir.resolve()
    output = args.output.resolve() if args.output else package / "MANIFEST_SHA256.tsv"
    excluded_parts = {"__pycache__", "pipeline_test_output", ".git"}
    files = sorted(
        path for path in package.rglob("*")
        if path.is_file() and path.resolve() != output and not (set(path.relative_to(package).parts) & excluded_parts)
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["relative_path", "size_bytes", "sha256"])
        for path in files:
            writer.writerow([path.relative_to(package).as_posix(), path.stat().st_size, sha256(path)])
    print(f"Wrote {len(files)} records to {output}")


if __name__ == "__main__":
    main()
