"""Read-only, fresh-process stage-2 audit of all reconstructed demonstrations.

The stage-1 validator recomputes rigid geometry, contact labels and eligible
windows from the actual arrays. Its manifest-writing hook is intercepted here:
newly recomputed manifests must equal the frozen manifests and are never saved
over them. No historical artifact is modified by this script.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import audit_dataset as audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "runs/stages234_20261022/data_audit.json")
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to((ROOT / "runs").resolve()):
        raise ValueError("Audit output must be inside the project's runs directory")
    if output.exists():
        raise FileExistsError(f"Preserving existing audit: {output}")

    start = time.perf_counter()
    protected = [ROOT / "resource_manifest.json", ROOT / "reports/replay_validation.json",
                 ROOT / "reports/data_audit.json", ROOT / "data/derived/splits.json",
                 ROOT / "data/derived/windows.json", ROOT / "scripts/audit_dataset.py",
                 ROOT / "scripts/reconstruct_contacts.py"]
    protected += sorted((ROOT / "data/derived/replayed_contacts").glob("episode_*.npz"))
    before = {str(p.relative_to(ROOT)).replace("\\", "/"): audit.sha256(p) for p in protected}
    report = {"audit_schema_version": 2, "status": "failed",
              "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "python_version": sys.version, "command": sys.argv,
              "fresh_process": True, "historical_data_read_only": True,
              "script_sha256": audit.sha256(Path(__file__)),
              "reused_validator_sha256": audit.sha256(ROOT / "scripts/audit_dataset.py")}
    compared = []

    def compare_frozen_manifest(path: Path, recomputed: dict) -> None:
        allowed = {ROOT / "data/derived/splits.json", ROOT / "data/derived/windows.json"}
        audit.require(path in allowed, f"Unexpected historical write attempted: {path}")
        old = json.loads(path.read_text(encoding="utf-8"))
        audit.require(old == recomputed, f"Frozen manifest differs from recomputation: {path}")
        compared.append(str(path.relative_to(ROOT)).replace("\\", "/"))

    original_save = audit.save_json
    try:
        audit.save_json = compare_frozen_manifest
        audit.run_audit(report)
        audit.require(len(compared) == 2, "Both frozen manifests must be freshly recomputed")
        splits = json.loads((ROOT / "data/derived/splits.json").read_text(encoding="utf-8"))
        windows = json.loads((ROOT / "data/derived/windows.json").read_text(encoding="utf-8"))
        horizon = int(windows["horizon"])
        sets = {name: set(members) for name, members in splits["splits"].items()}
        audit.require(not (sets["train"] & sets["validation"] or sets["train"] & sets["test"]
                          or sets["validation"] & sets["test"]), "Episode split overlap")
        checked = 0
        boundary_by_split = {}
        for name, records in windows["windows"].items():
            grouped = {}
            for episode, anchor in records:
                audit.require(episode in sets[name], f"{name}: wrong episode {episode}")
                grouped.setdefault(episode, []).append(anchor)
            for episode, anchors in grouped.items():
                with np.load(ROOT / f"data/derived/replayed_contacts/episode_{episode:06d}.npz",
                             allow_pickle=False) as data:
                    length = len(data["actions"])
                    for anchor in anchors:
                        # Last observable state is states[length-1]. states[length]
                        # is a simulated extension with no corresponding video frame.
                        audit.require(0 <= anchor and anchor + horizon <= length - 1,
                                      f"Episode {episode}: unverified/cross-boundary endpoint")
                        audit.require(data["contact_label_valid"][anchor:anchor+horizon].all(),
                                      f"Episode {episode}: invalid seam contact in window")
                        audit.require(data["transition_has_video_endpoint"][anchor:anchor+horizon].all(),
                                      f"Episode {episode}: unavailable video endpoint")
                        checked += 1
            boundary_by_split[name] = len(records)
        after = {str(p.relative_to(ROOT)).replace("\\", "/"): audit.sha256(p) for p in protected}
        audit.require(before == after, "Historical artifact changed during fresh audit")
        report.update({"status": "passed", "recomputed_manifests_equal_frozen": compared,
                       "independently_checked_windows": checked,
                       "independently_checked_windows_by_split": boundary_by_split,
                       "historical_files_verified_unchanged": len(before),
                       "protected_files_sha256": before,
                       "frozen_split_path": "data/derived/splits.json",
                       "frozen_windows_path": "data/derived/windows.json"})
    except Exception as error:
        report.update({"status": "failed", "error": f"{type(error).__name__}: {error}"})
        raise
    finally:
        audit.save_json = original_save
        report["elapsed_seconds"] = time.perf_counter() - start
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        print(json.dumps({key: report.get(key) for key in
                          ["status", "episodes", "frames", "eligible_windows", "window_split_counts",
                           "historical_files_verified_unchanged", "elapsed_seconds", "error"]},
                         ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
