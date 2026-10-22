"""Regenerate all contact replay arrays in a new directory, preserving frozen split manifests.

The original reconstruction/audit kernels are imported unchanged. Fresh provenance
reports stay under --output-dir. Optional installation refuses any existing target.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

import audit_dataset as audit
import prepare_resources
import reconstruct_contacts as replay

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--install", action="store_true", help="Copy audited arrays into missing data/derived/replayed_contacts")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    install = ROOT / "data/derived/replayed_contacts"
    if output.exists():
        raise FileExistsError(f"Use a new output directory: {output}")
    if args.install and install.exists():
        raise FileExistsError(f"Existing replay arrays preserved; omit --install to independently rebuild: {install}")
    frozen = {name: ROOT / f"data/derived/{name}.json" for name in ("splits", "windows")}
    before = {name: audit.sha256(path) for name, path in frozen.items()}
    output.mkdir(parents=True)
    prepare_resources.run(output / "source_check", offline=True)
    manifest = json.loads((ROOT / "resource_manifest.json").read_text(encoding="utf-8"))
    # A small isolated tree lets the unmodified audit resolve its pinned source/report paths.
    paths = ["resource_manifest.json", "scripts/reconstruct_contacts.py"] + [item["path"] for item in manifest["files"]]
    for relative in paths:
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    old_root, old_argv = replay.ROOT, sys.argv
    try:
        replay.ROOT = output
        sys.argv = [str(Path(replay.__file__)), "--max-episodes", "206", "--points", "64", "--sigma", "5"]
        replay.main()
    finally:
        replay.ROOT, sys.argv = old_root, old_argv
    report = {"status": "failed", "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "fresh_reconstruction": True, "original_replay_kernel_sha256": audit.sha256(Path(replay.__file__)),
              "original_audit_kernel_sha256": audit.sha256(Path(audit.__file__)),
              "installed": False}
    old_audit_root, old_derived = audit.ROOT, audit.DERIVED
    try:
        audit.ROOT, audit.DERIVED = output, output / "data/derived"
        audit.run_audit(report)
        comparisons = {}
        for name, frozen_path in frozen.items():
            original = json.loads(frozen_path.read_text(encoding="utf-8"))
            rebuilt_path = output / f"data/derived/{name}.json"
            rebuilt = json.loads(rebuilt_path.read_text(encoding="utf-8"))
            # Fresh replay reports contain fresh timing/runtime information; their SHA must differ.
            semantic_keys = ("dataset_revision", "reconstruction_schema_version", "seed", "rng")
            semantic_keys += ("episode_counts", "splits") if name == "splits" else ("horizon", "stride", "counts", "windows")
            for key in semantic_keys:
                audit.require(original[key] == rebuilt[key], f"Frozen {name}.{key} does not match fresh reconstruction")
            comparisons[name] = {"membership_and_parameters_equal": True, "frozen_sha256": before[name],
                                 "fresh_sha256": audit.sha256(rebuilt_path),
                                 "fresh_provenance_not_substituted_for_historical_provenance": True}
        audit.require(before == {name: audit.sha256(path) for name, path in frozen.items()}, "Frozen manifests changed")
        report["frozen_manifest_comparison"] = comparisons
        if args.install:
            # Directory-level copy refuses pre-existing destinations, including a concurrent creator.
            shutil.copytree(output / "data/derived/replayed_contacts", install)
            report["installed"] = True
        report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        audit.ROOT, audit.DERIVED = old_audit_root, old_derived
        with (output / "fresh_derived_audit.json").open("x", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
    print(json.dumps({key: report[key] for key in ("status", "episodes", "frames", "eligible_windows", "installed")}))


if __name__ == "__main__":
    main()
