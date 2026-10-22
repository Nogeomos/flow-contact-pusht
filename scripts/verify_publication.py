"""Verify the published repository and optional local release ZIPs, without modifying them.

This checks publication integrity. It does not rerun the historical experiments
or substitute published hashes for the original-workspace audit chain.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def safe_path(root, relative):
    posix = PurePosixPath(relative)
    if posix.is_absolute() or ".." in posix.parts or "\\" in relative or ":" in relative:
        raise ValueError(f"Unsafe publication path: {relative}")
    path = root.joinpath(*posix.parts).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Publication path escapes root: {relative}")
    return path


def verify(release_dir=None, require_git=False):
    manifest = json.loads((ROOT / "artifacts/PUBLICATION_MANIFEST.json").read_text(encoding="utf-8"))
    assets = json.loads((ROOT / "artifacts/RELEASE_ASSETS.json").read_text(encoding="utf-8"))
    if manifest["schema_version"] != 1 or assets["schema_version"] != 1:
        raise ValueError("Unsupported publication manifest schema")
    entries = manifest["files"]
    unique = {(e["destination"], e["published_path"]) for e in entries}
    if len(unique) != len(entries):
        raise ValueError("Duplicate publication manifest entry")
    if not all(e["destination"] in ("repository", "models", "evidence") for e in entries):
        raise ValueError("Unknown publication destination")
    repo_entries = [e for e in entries if e["destination"] == "repository"]
    for e in repo_entries:
        path = safe_path(ROOT, e["published_path"])
        if not path.is_file() or path.stat().st_size != e["bytes"] or digest(path) != e["published_sha256"]:
            raise ValueError(f"Published file verification failed: {e['published_path']}")
    frozen = manifest["frozen_scientific_sources"]
    for relative, expected in frozen.items():
        if digest(safe_path(ROOT, relative)) != expected:
            raise ValueError(f"Frozen scientific source changed: {relative}")
    git_verified = False
    if require_git:
        result = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, check=True, capture_output=True)
        tracked = {p.decode("utf-8") for p in result.stdout.split(b"\0") if p}
        expected = {e["published_path"] for e in repo_entries} | set(manifest["self_excluded"])
        if tracked != expected:
            raise ValueError(f"Git/manifest mismatch: missing={sorted(expected-tracked)}, extra={sorted(tracked-expected)}")
        status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, check=True, capture_output=True)
        if status.stdout.strip():
            raise ValueError("Tracked working tree is not clean")
        git_verified = True
    verified_assets = []
    if release_dir:
        for asset in assets["assets"]:
            archive = safe_path(release_dir, asset["filename"])
            if archive.stat().st_size != asset["bytes"] or digest(archive) != asset["sha256"]:
                raise ValueError(f"Release ZIP verification failed: {asset['filename']}")
            destination = Path(asset["filename"]).stem
            expected = {e["published_path"]: e for e in entries if e["destination"] == destination}
            with zipfile.ZipFile(archive) as zipped:
                infos = [i for i in zipped.infolist() if not i.is_dir()]
                names = [i.filename for i in infos]
                if len(names) != len(set(names)) or set(names) != set(expected):
                    raise ValueError(f"Release member set differs from manifest: {archive.name}")
                if len(infos) != asset["file_count"]:
                    raise ValueError(f"Release member count differs: {archive.name}")
                for item in infos:
                    e = expected[item.filename]
                    safe_path(ROOT, item.filename)
                    with zipped.open(item) as handle:
                        actual = hashlib.file_digest(handle, "sha256").hexdigest()
                    if item.file_size != e["bytes"] or actual != e["published_sha256"]:
                        raise ValueError(f"Release member verification failed: {archive.name}:{item.filename}")
            verified_assets.append({"filename": archive.name, "sha256": asset["sha256"], "verified_files": len(expected)})
    return {"status": "passed", "repository_files_verified": len(repo_entries),
            "frozen_scientific_sources_verified": len(frozen), "git_tracked_set_and_clean_tree_verified": git_verified,
            "release_assets_verified": verified_assets,
            "manifest_entries_by_destination": dict(Counter(e["destination"] for e in entries)),
            "scope": "Published file integrity, not a new training or historical experiment acceptance run."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-dir", type=Path, help="Folder containing the downloaded models.zip and evidence.zip")
    parser.add_argument("--require-clean-git", action="store_true")
    parser.add_argument("--output", type=Path, help="Optional new verification report; an existing file is preserved")
    args = parser.parse_args()
    if args.output and args.output.exists():
        raise FileExistsError(f"Preserving existing report: {args.output}")
    report = verify(args.release_dir.resolve() if args.release_dir else None, args.require_clean_git)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
