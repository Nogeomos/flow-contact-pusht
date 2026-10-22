"""Audit reconstructed PushT data and create fixed episode/window manifests.

Run after the complete schema-v2 reconstruction. Horizon 8 means eight action
and contact intervals plus nine geometry samples (the anchor and eight future
states). Manifests contain indices only; this script does not expand windows.
"""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/lerobot_pusht"
DERIVED = ROOT / "data/derived"
SEED = 20261022
HORIZON = 8
STRIDE = 1
SPLIT_COUNTS = {"train": 164, "validation": 21, "test": 21}
OUTLINE = np.array([[-60, 0], [60, 0], [60, 30], [15, 30],
                    [15, 120], [-15, 120], [-15, 30], [-60, 30]], dtype=float)


class AuditError(RuntimeError):
    pass


def require(condition, message):
    if not bool(condition):
        raise AuditError(message)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def exterior_distances(points):
    edges = np.roll(OUTLINE, -1, axis=0) - OUTLINE
    offset = points[:, None, :] - OUTLINE
    fraction = np.clip((offset * edges).sum(-1) / (edges * edges).sum(-1), 0, 1)
    return np.linalg.norm(offset - fraction[..., None] * edges, axis=-1).min(-1)


def check_finite(name, array):
    if array.dtype.names:
        for field in array.dtype.names:
            check_finite(f"{name}.{field}", array[field])
    elif array.dtype.kind in "fciub":
        require(np.isfinite(array).all(), f"{name}: non-finite value")
    else:
        raise AuditError(f"{name}: unexpected dtype {array.dtype}")


def audit_sources(manifest):
    results = []
    require(bool(manifest.get("files")), "Resource manifest contains no files")
    seen = set()
    for item in manifest["files"]:
        relative = item["path"].replace("\\", "/")
        require(relative not in seen, f"Duplicate resource manifest path: {relative}")
        seen.add(relative)
        path = (ROOT / relative).resolve()
        require(path.is_relative_to(ROOT.resolve()), f"Resource path escapes project: {relative}")
        require(path.is_file(), f"Missing source file: {relative}")
        size = path.stat().st_size
        digest = sha256(path)
        require(size == item["bytes"], f"Source size mismatch: {relative}")
        require(digest == item["sha256"], f"Source SHA256 mismatch: {relative}")
        results.append({"path": relative, "bytes": size, "sha256": digest, "passed": True})
    return results


def audit_episode(path, meta, source_actions, source_agent, sigma, common_points):
    episode = int(meta["episode_index"])
    first, last = int(meta["dataset_from_index"]), int(meta["dataset_to_index"])
    length = last - first
    prefix = f"Episode {episode}"
    required = {
        "episode_index", "reset_input", "states", "velocities", "local_boundary_points",
        "points_world", "actions", "contact_events", "contact_soft_labels",
        "contact_label_valid", "transition_has_video_endpoint",
        "reference_visible_mask_iou", "reference_strict_color_mask_iou", "agent_position_error",
    }
    with np.load(path, allow_pickle=False) as archive:
        require(required.issubset(archive.files), f"{prefix}: missing schema-v2 arrays")
        arrays = {name: archive[name] for name in required}
    for name, array in arrays.items():
        check_finite(f"{prefix}.{name}", array)
    require(arrays["episode_index"].shape == (), f"{prefix}: episode_index must be scalar")
    require(int(arrays["episode_index"]) == episode, f"{prefix}: wrong stored episode ID")
    local = arrays["local_boundary_points"]
    require(local.ndim == 2 and local.shape[1] == 2 and len(local) >= 8,
            f"{prefix}: invalid local boundary points")
    n = len(local)
    expected_shapes = {
        "reset_input": (5,), "states": (length + 1, 5), "velocities": (length + 1, 5),
        "points_world": (length + 1, n, 2), "actions": (length, 2),
        "contact_soft_labels": (length, n), "contact_label_valid": (length,),
        "transition_has_video_endpoint": (length,), "reference_visible_mask_iou": (length,),
        "reference_strict_color_mask_iou": (length,), "agent_position_error": (length,),
    }
    for name, shape in expected_shapes.items():
        require(arrays[name].shape == shape, f"{prefix}.{name}: shape mismatch")
    for name in ["contact_label_valid", "transition_has_video_endpoint"]:
        require(arrays[name].dtype.kind == "b", f"{prefix}.{name}: expected boolean flags")
    require(np.array_equal(arrays["actions"], source_actions[first:last]),
            f"{prefix}: actions differ from the public parquet")
    require((exterior_distances(local) < 1e-8).all(), f"{prefix}: sampled points leave exterior")
    if common_points is not None:
        require(np.array_equal(local, common_points), f"{prefix}: point identities changed between episodes")
    rs = np.random.RandomState(episode)
    reset = np.array([rs.randint(50, 450), rs.randint(50, 450), rs.randint(100, 400),
                      rs.randint(100, 400), rs.randn() * 2 * np.pi - np.pi])
    require(np.array_equal(arrays["reset_input"], reset), f"{prefix}: initialization recipe differs")
    states = arrays["states"]
    require(((states[:, 4] >= 0) & (states[:, 4] < 2 * np.pi)).all(), f"{prefix}: invalid wrapped angles")
    rigid_max_error = 0.0
    for frame, state in enumerate(states):
        c, s = np.cos(state[4]), np.sin(state[4])
        rotation = np.array([[c, -s], [s, c]])
        expected_world = local @ rotation.T + state[2:4]
        rigid_max_error = max(rigid_max_error, float(np.max(np.abs(expected_world - arrays["points_world"][frame]))))
    require(rigid_max_error < 1e-4, f"{prefix}: world points violate the rigid transform")
    agent_error = np.linalg.norm(states[:-1, :2] - source_agent[first:last], axis=1)
    require(np.allclose(agent_error, arrays["agent_position_error"], rtol=1e-8, atol=1e-12),
            f"{prefix}: recorded agent error metric is inconsistent")
    require(agent_error.max() < 1e-3, f"{prefix}: agent replay is misaligned")
    for name in ["reference_visible_mask_iou", "reference_strict_color_mask_iou", "contact_soft_labels"]:
        require(((arrays[name] >= 0) & (arrays[name] <= 1)).all(), f"{prefix}.{name}: values outside [0,1]")
    events = arrays["contact_events"]
    event_fields = {
        "action_index", "substep", "manifold_index", "point_world", "point_local",
        "normal_into_block", "arbiter_total_impulse", "impulse_on_block", "block_is_shape_a",
        "signed_separation", "distance_to_exterior", "on_exterior",
    }
    require(events.ndim == 1 and event_fields.issubset(events.dtype.names or ()),
            f"{prefix}: invalid contact-event structure")
    for name in ["point_world", "point_local", "normal_into_block", "arbiter_total_impulse", "impulse_on_block"]:
        require(events[name].shape == (len(events), 2), f"{prefix}.{name}: invalid contact vector shape")
    expected_valid = np.ones(length, dtype=bool)
    if len(events):
        times = events["action_index"]
        require(((times >= 0) & (times < length)).all(), f"{prefix}: action index out of bounds")
        require(((events["substep"] >= 0) & (events["substep"] < 10)).all(), f"{prefix}: substep out of bounds")
        require((events["manifold_index"] > 0).all(), f"{prefix}: invalid manifold index")
        chronology = times.astype(np.int64) * 10 + events["substep"]
        require((np.diff(chronology) >= 0).all(), f"{prefix}: contact events are not chronological")
        distances = exterior_distances(events["point_local"])
        require(np.allclose(distances, events["distance_to_exterior"], rtol=0, atol=1e-9),
                f"{prefix}: exterior-distance metadata is inconsistent")
        require(np.array_equal(events["on_exterior"], distances < 1e-4), f"{prefix}: exterior flags disagree")
        require(np.allclose(np.linalg.norm(events["normal_into_block"], axis=1), 1, rtol=0, atol=1e-8),
                f"{prefix}: contact normal is not unit length")
        expected_impulse = events["arbiter_total_impulse"] * np.where(events["block_is_shape_a"], 1, -1)[:, None]
        require(np.allclose(expected_impulse, events["impulse_on_block"], rtol=0, atol=1e-10),
                f"{prefix}: impulse-on-block sign is inconsistent")
        normal_impulse = (events["impulse_on_block"] * events["normal_into_block"]).sum(-1)
        require((normal_impulse >= -1e-8).all(), f"{prefix}: contact impulse points outward")
        require((events["signed_separation"] <= 1e-8).all(), f"{prefix}: positive contact separation")
        expected_valid[np.unique(times[~events["on_exterior"]])] = False
        # Each manifold impulse is duplicated across its one or two contact points.
        boundaries = np.r_[0, np.flatnonzero((np.diff(times) != 0) |
                                            (np.diff(events["manifold_index"]) != 0)) + 1, len(events)]
        keys = []
        for start, stop in zip(boundaries[:-1], boundaries[1:]):
            group = events[start:stop]
            keys.append((int(group[0]["action_index"]), int(group[0]["manifold_index"])))
            require(1 <= len(group) <= 2, f"{prefix}: unexpected points per manifold")
            for field in ["substep", "block_is_shape_a", "arbiter_total_impulse", "impulse_on_block", "normal_into_block"]:
                require(np.all(group[field] == group[field][0]), f"{prefix}: inconsistent manifold {field}")
        require(len(keys) == len(set(keys)), f"{prefix}: manifold key reused at a different substep")
    require(np.array_equal(arrays["contact_label_valid"], expected_valid), f"{prefix}: invalid interval flags")
    expected_endpoint = np.ones(length, dtype=bool)
    expected_endpoint[-1] = False
    require(np.array_equal(arrays["transition_has_video_endpoint"], expected_endpoint),
            f"{prefix}: video endpoint flags do not exclude exactly the last transition")
    label_error = 0.0
    empty_intervals = 0
    for t in range(length):
        left, right = np.searchsorted(events["action_index"], [t, t + 1])
        interval_events = events[left:right]
        exterior = interval_events[interval_events["on_exterior"]]
        if len(exterior):
            d2 = ((local[:, None, :] - exterior["point_local"][None, :, :]) ** 2).sum(-1)
            expected_label = np.exp(-d2.min(axis=1) / (2 * sigma ** 2))
        else:
            expected_label = np.zeros(n)
            empty_intervals += 1
            require(np.count_nonzero(arrays["contact_soft_labels"][t]) == 0,
                    f"{prefix}, action {t}: nonzero labels without exterior contacts")
        label_error = max(label_error, float(np.max(np.abs(expected_label - arrays["contact_soft_labels"][t]))))
    require(label_error < 1e-6, f"{prefix}: exterior-only soft labels are inconsistent")
    eligible = arrays["contact_label_valid"] & arrays["transition_has_video_endpoint"]
    candidate_count = max(0, length - HORIZON + 1)
    invalid_cumulative = np.r_[0, np.cumsum(~eligible)]
    starts = [t for t in range(0, candidate_count, STRIDE)
              if invalid_cumulative[t + HORIZON] == invalid_cumulative[t]]
    summary = {
        "episode_index": episode, "frames": length, "points": n,
        "contact_point_events": len(events),
        "internal_seam_events": int(np.count_nonzero(~events["on_exterior"])),
        "invalid_label_intervals": int(np.count_nonzero(~arrays["contact_label_valid"])),
        "unverified_endpoints": int(np.count_nonzero(~arrays["transition_has_video_endpoint"])),
        "intervals_without_exterior_contacts": empty_intervals,
        "zero_impulse_point_events": int(np.count_nonzero(np.linalg.norm(events["impulse_on_block"], axis=1) < 1e-12)),
        "rigid_point_max_absolute_error": rigid_max_error,
        "soft_label_max_absolute_error": label_error,
        "agent_position_max_error": float(agent_error.max()),
        "candidate_windows": candidate_count, "eligible_windows": len(starts),
        "npz_sha256": sha256(path),
    }
    return summary, starts, local


def run_audit(report):
    manifest = json.loads((ROOT / "resource_manifest.json").read_text(encoding="utf-8"))
    replay = json.loads((ROOT / "reports/replay_validation.json").read_text(encoding="utf-8"))
    require(replay.get("data_schema_version") == 2 and replay.get("episodes") == 206,
            "Complete schema-v2 reconstruction is not ready; run this audit after it finishes")
    require(replay["script_sha256"] == sha256(ROOT / "scripts/reconstruct_contacts.py"),
            "Replay report does not match the current reconstruction script")
    require(replay["dataset_revision"] == manifest["dataset_revision"], "Dataset revision mismatch")
    require(replay["repositories"] == manifest["repositories"], "Repository revision mismatch")
    report["source_files"] = audit_sources(manifest)
    report["resource_manifest_sha256"] = sha256(ROOT / "resource_manifest.json")
    report["replay_report_sha256"] = sha256(ROOT / "reports/replay_validation.json")
    report["dataset_revision"] = manifest["dataset_revision"]
    source = pq.read_table(RAW / "data/chunk-000/file-000.parquet")
    metadata = pq.read_table(RAW / "meta/episodes/chunk-000/file-000.parquet").to_pylist()
    ids = [int(meta["episode_index"]) for meta in metadata]
    require(len(ids) == 206 and len(set(ids)) == 206, "Expected 206 unique source episodes")
    source_actions = np.asarray(source["action"].to_pylist(), dtype=float)
    source_agent = np.asarray(source["observation.state"].to_pylist(), dtype=float)
    source_ids = source["episode_index"].to_numpy()
    source_frames = source["frame_index"].to_numpy()
    require(source.num_rows == replay["frames"], "Replay/source total frame mismatch")
    source_end = 0
    for meta in metadata:
        first, last = int(meta["dataset_from_index"]), int(meta["dataset_to_index"])
        require(first == source_end and last > first, "Episode slices are not contiguous/nonempty")
        require(last <= source.num_rows, "Episode slice extends beyond source table")
        require(np.all(source_ids[first:last] == meta["episode_index"]), "Source episode IDs disagree with metadata")
        require(np.array_equal(source_frames[first:last], np.arange(last - first)), "Source frame IDs are not contiguous")
        source_end = last
    require(source_end == source.num_rows, "Episode metadata does not cover every source row")
    paths = list((DERIVED / "replayed_contacts").glob("episode_*.npz"))
    require({p.name for p in paths} == {f"episode_{ep:06d}.npz" for ep in ids},
            "Reconstruction directory has missing or unexpected episodes")
    sigma = float(replay["configuration"]["sigma"])
    require(np.isfinite(sigma) and sigma > 0, "Invalid smoothing sigma")
    episode_results, window_starts, common_points = [], {}, None
    for meta in metadata:
        episode = int(meta["episode_index"])
        result, starts, common_points = audit_episode(
            DERIVED / f"replayed_contacts/episode_{episode:06d}.npz", meta,
            source_actions, source_agent, sigma, common_points,
        )
        episode_results.append(result)
        window_starts[episode] = starts
    for report_key, episode_key in [
        ("contact_point_events", "contact_point_events"),
        ("internal_seam_events", "internal_seam_events"),
        ("invalid_contact_label_intervals", "invalid_label_intervals"),
    ]:
        require(sum(item[episode_key] for item in episode_results) == replay[report_key],
                f"Reconstruction report count mismatch: {report_key}")
    permutation = np.random.Generator(np.random.PCG64(SEED)).permutation(sorted(ids)).tolist()
    split_ids, position = {}, 0
    for name, count in SPLIT_COUNTS.items():
        split_ids[name] = sorted(permutation[position:position + count])
        position += count
    flat = [ep for members in split_ids.values() for ep in members]
    require(len(flat) == len(set(flat)) == 206 and set(flat) == set(ids), "Split leakage or incomplete coverage")
    windows = {name: [[ep, start] for ep in members for start in window_starts[ep]]
               for name, members in split_ids.items()}
    for name, entries in windows.items():
        require(all(ep in split_ids[name] for ep, _ in entries), f"{name}: window assigned to wrong split")
        require(len(entries) == len(set(map(tuple, entries))), f"{name}: duplicate windows")
    common = {
        "schema_version": 1, "dataset_revision": manifest["dataset_revision"],
        "reconstruction_schema_version": 2,
        "resource_manifest_sha256": report["resource_manifest_sha256"],
        "replay_report_sha256": report["replay_report_sha256"],
        "seed": SEED, "rng": "numpy.random.Generator(PCG64)",
        "numpy_version": np.__version__,
    }
    splits_document = {
        **common, "episode_counts": SPLIT_COUNTS, "splits": split_ids,
        "assignment": "Shuffle sorted episode IDs once, take 164/21/21, then sort each split for storage.",
        "selection": "No success, reward, IoU, or task-performance filtering.",
    }
    windows_document = {
        **common, "horizon": HORIZON, "stride": STRIDE,
        "record_columns": ["episode_index", "start_action_index"],
        "slices": {
            "actions_and_labels": "[start:start+horizon] (8 intervals)",
            "states_and_points": "[start:start+horizon+1] (9 samples: anchor plus 8 future states)",
        },
        "eligibility": "Every interval must have contact_label_valid=True and transition_has_video_endpoint=True.",
        "selection": "All eligible windows; no success or IoU filtering; no padding or cross-episode windows.",
        "counts": {name: len(entries) for name, entries in windows.items()}, "windows": windows,
    }
    save_json(DERIVED / "splits.json", splits_document)
    save_json(DERIVED / "windows.json", windows_document)
    report.update({
        "status": "passed", "episodes": len(ids), "frames": source.num_rows,
        "contact_point_events": sum(item["contact_point_events"] for item in episode_results),
        "internal_seam_events": sum(item["internal_seam_events"] for item in episode_results),
        "invalid_label_intervals": sum(item["invalid_label_intervals"] for item in episode_results),
        "unverified_endpoints": sum(item["unverified_endpoints"] for item in episode_results),
        "zero_impulse_point_events": sum(item["zero_impulse_point_events"] for item in episode_results),
        "candidate_windows": sum(item["candidate_windows"] for item in episode_results),
        "eligible_windows": sum(item["eligible_windows"] for item in episode_results),
        "episode_split_counts": SPLIT_COUNTS, "window_split_counts": windows_document["counts"],
        "max_rigid_point_error": max(item["rigid_point_max_absolute_error"] for item in episode_results),
        "max_soft_label_error": max(item["soft_label_max_absolute_error"] for item in episode_results),
        "outputs": {
            "splits": {"path": "data/derived/splits.json", "sha256": sha256(DERIVED / "splits.json")},
            "windows": {"path": "data/derived/windows.json", "sha256": sha256(DERIVED / "windows.json")},
        },
        "checks": [
            "Source file bytes and SHA256 match the pinned resource manifest.",
            "All 206 episodes match source IDs, frame slices, and actions.",
            "Schema-v2 arrays are finite and have consistent shapes.",
            "World points follow the rigid transform of stable local boundary points.",
            "Contact indices, normals, impulse signs, manifolds, and exterior flags are consistent.",
            "Soft labels match exterior contacts, with zero labels when no exterior contact occurs.",
            "Invalid labels and unverified final endpoints are excluded from every window.",
            "Episode splits are disjoint and exhaustively cover the source episodes.",
        ],
        "limitations": [
            "Passing checks establishes internal consistency, not independent contact ground truth.",
            "Video IoU values are checked for valid range, not used as an exact pose-error measure or filter.",
            "Contact points are sampled inside physics substeps; they must not be transformed using only the action-start pose.",
            "Geometry labels include zero-impulse contacts; these are not necessarily effective pushes.",
            "Overlapping windows stay within each episode split; they are not independent statistical samples.",
        ],
        "per_episode": episode_results,
    })


def main():
    report = {"audit_schema_version": 1, "status": "failed",
              "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "audit_script_sha256": sha256(Path(__file__)), "python_version": sys.version}
    try:
        run_audit(report)
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        report["outputs_written_this_run"] = False
        save_json(ROOT / "reports/data_audit.json", report)
        print(json.dumps({"status": "failed", "error": report["error"]}), flush=True)
        raise
    save_json(ROOT / "reports/data_audit.json", report)
    print(json.dumps({key: report[key] for key in ["status", "episodes", "frames", "eligible_windows",
                                                   "episode_split_counts", "window_split_counts"]}), flush=True)


if __name__ == "__main__":
    main()
