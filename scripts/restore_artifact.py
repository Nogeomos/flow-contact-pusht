"""Restore a SHA256-verified local release ZIP without replacing differing files."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import zipfile

ROOT = Path(__file__).resolve().parents[1]
RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def restore(archive, expected, output):
    if not re.fullmatch(r"[0-9a-fA-F]{64}", expected):
        raise ValueError("An explicit 64-character release ZIP SHA256 is required")
    if output.exists():
        raise FileExistsError(f"Preserving existing restoration report: {output}")
    actual = sha(archive)
    if actual.lower() != expected.lower():
        raise ValueError(f"ZIP checksum mismatch: expected {expected}, got {actual}; nothing extracted")
    entries, names = [], set()
    with zipfile.ZipFile(archive) as zipped:
        for item in zipped.infolist():
            raw = item.filename.replace("\\", "/")
            pure = PurePosixPath(raw)
            parts = pure.parts
            if (not parts or pure.is_absolute() or any(p in (".", "..") for p in raw.split("/"))
                    or any(":" in p or p.endswith((".", " ")) or p.split(".")[0].upper() in RESERVED for p in parts)
                    or stat.S_ISLNK(item.external_attr >> 16)):
                raise ValueError(f"Unsafe ZIP member: {raw}")
            if any(p.lower() in {".git", ".venv", ".venv-train", ".venv-cpu", ".venv-cuda126"} for p in parts):
                raise ValueError(f"Release artifacts cannot replace repository/environment internals: {raw}")
            key = pure.as_posix().casefold().rstrip("/")
            if key in names:
                raise ValueError(f"Duplicate ZIP member: {raw}")
            names.add(key)
            target = (ROOT / Path(*parts)).resolve()
            if not target.is_relative_to(ROOT) or target == ROOT:
                raise ValueError(f"ZIP member escapes checkout: {raw}")
            if item.is_dir():
                if target.exists() and not target.is_dir():
                    raise FileExistsError(f"Directory conflicts with existing file: {target}")
                continue
            for parent in target.parents:
                if parent == ROOT:
                    break
                if parent.exists() and not parent.is_dir():
                    raise FileExistsError(f"Parent conflicts with existing file: {parent}")
            with zipped.open(item) as source:
                digest = hashlib.file_digest(source, "sha256").hexdigest()
            if target.exists() and (not target.is_file() or sha(target) != digest):
                raise FileExistsError(f"Existing file differs; nothing extracted, preserved: {target}")
            entries.append((item, target, digest, target.exists()))
        file_paths = {p.as_posix().casefold() for _, p, _, _ in entries}
        if any(parent.as_posix().casefold() in file_paths for _, p, _, _ in entries for parent in p.parents):
            raise ValueError("ZIP contains conflicting file and directory paths")
        # Preflight every member before creating any file. Exclusive opens guard concurrent writers.
        for item, target, digest, existed in entries:
            if existed:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zipped.open(item) as source, target.open("xb") as destination:
                shutil.copyfileobj(source, destination)
            if sha(target) != digest:
                raise ValueError(f"Post-write checksum mismatch: {target}")
    report = {"status": "passed", "zip_sha256": actual, "zip_name": archive.name,
              "files": len(entries), "created": sum(not existed for _, _, _, existed in entries),
              "identical_existing_skipped": sum(existed for _, _, _, existed in entries),
              "overwrite_existing": False,
              "members": [{"path": p.relative_to(ROOT).as_posix(), "sha256": digest}
                          for _, p, digest, _ in entries]}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != "members"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    restore(args.zip.resolve(), args.sha256, args.output.resolve())
