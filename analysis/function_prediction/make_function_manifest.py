#!/usr/bin/env python3
"""Write checksums for the function-prediction code package."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "FT_publication_code_sha256.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    paths = sorted(
        path for path in ROOT.rglob("*")
        if path.is_file()
        and path.resolve() != OUTPUT.resolve()
        and "__pycache__" not in path.parts
    )
    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["File", "Size_bytes", "SHA256"])
        for path in paths:
            writer.writerow([path.relative_to(ROOT).as_posix(), path.stat().st_size, sha256(path)])
    print(f"files={len(paths)}")
    print(f"output={OUTPUT}")


if __name__ == "__main__":
    main()
