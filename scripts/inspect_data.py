"""Inspect raw PushT data, live simulator parameters, and local DP requirements.

No previous reconstruction/audit report is used. The first three episodes are
replayed only in memory to measure state alignment and actual contact friction.
The selected output directory may exist; existing report files are not replaced.
"""
from pathlib import Path
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/lerobot_pusht"
sys.dont_write_bytecode = True
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "external/gym-pusht"))

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pymunk
from shapely.geometry import Polygon
from shapely.ops import unary_union
from gym_pusht.envs import pymunk_override
from reconstruct_contacts import ContactReplayEnv, initial_state, OUTLINE


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_number(value):
    value = float(value)
    return value if np.isfinite(value) else str(value)


def stats(values):
    array = np.asarray(values, dtype=float)
    if not array.size:
        return {"count": 0}
    return {"count": int(array.size), "finite": bool(np.isfinite(array).all()),
            "min": json_number(array.min()), "max": json_number(array.max()),
            "mean": json_number(array.mean()), "median": json_number(np.median(array)),
            "p95": json_number(np.quantile(array, 0.95))}


def verify_manifest():
    path = ROOT / "resource_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    checks = []
    for item in manifest["files"]:
        relative = item["path"].replace("\\", "/")
        file = (ROOT / relative).resolve()
        if not file.is_relative_to(ROOT.resolve()):
            raise ValueError(f"Manifest path escapes the project: {relative}")
        exists = file.is_file()
        size = file.stat().st_size if exists else None
        digest = sha256(file) if exists else None
        checks.append({"path": relative, "exists": exists, "expected_bytes": item["bytes"],
                       "actual_bytes": size, "expected_sha256": item["sha256"],
                       "actual_sha256": digest,
                       "passed": exists and size == item["bytes"] and digest == item["sha256"]})
    return {"manifest_sha256": sha256(path), "dataset_revision": manifest["dataset_revision"],
            "repositories": manifest["repositories"], "files": checks,
            "all_passed": all(item["passed"] for item in checks)}


class FrictionProbe(ContactReplayEnv):
    def __init__(self):
        self.friction_records = []
        super().__init__()

    def _handle_collision(self, arbiter, space, data):
        super()._handle_collision(arbiter, space, data)
        a, b = arbiter.shapes
        if {a.body, b.body} != {self.agent, self.block}:
            return
        self.friction_records.append({
            "action_index": self.action_index, "substep": self.substep,
            "arbiter_friction": float(arbiter.friction), "arbiter_restitution": float(arbiter.restitution),
            "shape_a_friction": float(a.friction), "shape_b_friction": float(b.friction),
            "shape_a_elasticity": float(a.elasticity), "shape_b_elasticity": float(b.elasticity),
            "contact_point_count": len(arbiter.contact_point_set.points),
        })


def body_snapshot(body, name):
    body_types = {pymunk.Body.DYNAMIC: "DYNAMIC", pymunk.Body.KINEMATIC: "KINEMATIC",
                  pymunk.Body.STATIC: "STATIC"}
    shapes = []
    for shape in body.shapes:
        item = {"type": type(shape).__name__, "friction": float(shape.friction),
                "elasticity": float(shape.elasticity), "sensor": bool(shape.sensor),
                "collision_type": int(shape.collision_type), "radius": float(shape.radius)}
        if isinstance(shape, pymunk.Poly):
            item["local_vertices"] = [list(point) for point in shape.get_vertices()]
        shapes.append(item)
    shapes.sort(key=lambda item: json.dumps(item, sort_keys=True))
    return {"name": name, "body_type": body_types[body.body_type], "body_type_code": int(body.body_type),
            "mass": json_number(body.mass), "moment": json_number(body.moment),
            "position": list(body.position), "angle": float(body.angle),
            "velocity": list(body.velocity), "angular_velocity": float(body.angular_velocity),
            "local_center_of_gravity": list(body.center_of_gravity),
            "world_origin": list(body.local_to_world((0, 0))),
            "world_center_of_gravity": list(body.local_to_world(body.center_of_gravity)),
            "body_friction_attribute": getattr(body, "friction", None),
            "body_class_declares_friction_property": hasattr(type(body), "friction"),
            "shapes": shapes,
            "nonfinite_note": "inf mass/moment are expected for kinematic or static bodies."}


def simulator_snapshot(env):
    polygons = [Polygon([tuple(point) for point in shape.get_vertices()]) for shape in env._block_shapes]
    exterior = unary_union(polygons)
    screen = env._draw()
    source_path = ROOT / "external/gym-pusht/gym_pusht/envs/pusht.py"
    source_lines = source_path.read_text(encoding="utf-8").splitlines()
    evidence = [{"line": i + 1, "text": line.strip()} for i, line in enumerate(source_lines)
                if "body.friction" in line or "inertia2 =" in line or "center_of_gravity =" in line]
    declared_outline = Polygon(OUTLINE)
    return {
        "source_file": str(source_path.relative_to(ROOT)), "source_sha256": sha256(source_path),
        "reconstruction_class_file_sha256": sha256(ROOT / "scripts/reconstruct_contacts.py"),
        "objects": {"pusher": body_snapshot(env.agent, "pusher"),
                    "block": body_snapshot(env.block, "block"),
                    "walls": body_snapshot(env.space.static_body, "walls")},
        "space": {"gravity": list(env.space.gravity), "damping": float(env.space.damping),
                  "iterations": int(env.space.iterations), "collision_slop": float(env.space.collision_slop),
                  "collision_bias": float(env.space.collision_bias)},
        "control": {"physics_dt_seconds": float(env.dt), "physics_hz": 1 / env.dt,
                    "control_hz": env.control_hz, "physics_substeps_per_action": int(1 / (env.dt * env.control_hz)),
                    "pd_position_gain": env.k_p, "pd_velocity_gain": env.k_v,
                    "action_low": env.action_space.low.tolist(), "action_high": env.action_space.high.tolist(),
                    "action_semantics": "Absolute pusher target position; PD updates its kinematic velocity before each physics substep."},
        "geometry": {"union_area_local_units_squared": float(exterior.area),
                     "union_area_centroid_local": list(exterior.centroid.coords[0]),
                     "actual_union_exterior": [list(point) for point in list(exterior.exterior.coords)[:-1]],
                     "declared_outline_matches_union": bool(declared_outline.equals(exterior)),
                     "symmetric_difference_area": float(declared_outline.symmetric_difference(exterior).area),
                     "origin_note": "Body position is the transformed local origin, not the center of gravity.",
                     "cog_note": "The code averages the two rectangle centroids without area weighting; the actual polygon centroid is reported separately."},
        "coordinates": {
            "renderer_surface_size": list(screen.get_size()), "observed_image_shape": list(env._render().shape),
            "positive_y_is_up": pymunk_override.positive_y_is_up,
            "to_screen_point_100_100": list(pymunk_override.to_pygame((100, 100), screen)),
            "to_screen_point_100_110": list(pymunk_override.to_pygame((100, 110), screen)),
            "interpretation": "On the renderer, +x points right and +y points down. Positive mathematical rotation consequently appears clockwise on screen.",
            "unit_note": "Positions are simulator canvas units (512-unit canvas), not calibrated meters. The observation is resized to 96x96. Time is seconds and angle is radians; no real-world mass/friction calibration is established.",
        },
        "source_evidence": evidence,
        "friction_interpretation": "body.friction is a Python attribute in this implementation. Collision friction is controlled by Shape.friction (or a collision-handler override); inspect the measured shape and arbiter values rather than treating body.friction as the effective coefficient.",
    }


def yaml_scalars(text):
    """Extract simple scalar entries with source lines; this is not a YAML parser."""
    stack, output = [], {}
    for number, line in enumerate(text.splitlines(), 1):
        match = re.match(r"^(\s*)([A-Za-z_][\w.-]*):\s*(.*)$", line)
        if not match:
            continue
        indent, key, value = len(match[1]), match[2], match[3].strip()
        while stack and stack[-1][0] >= indent:
            stack.pop()
        path = ".".join([item[1] for item in stack] + [key])
        if value and not value.startswith("#"):
            raw = value.split(" #", 1)[0].strip()
            try:
                parsed = {"true": True, "false": False, "null": None}.get(raw.lower(), ast.literal_eval(raw))
            except (ValueError, SyntaxError):
                parsed = {"true": True, "false": False, "null": None}.get(raw.lower(), raw)
            output[path] = {"value": parsed, "raw_value": raw, "line": number}
        else:
            stack.append((indent, key))
    return output


def inspect_dp():
    base = ROOT / "external/diffusion_policy"
    names = ["README.md", "conda_environment.yaml", "image_pusht_diffusion_policy_cnn.yaml",
             "diffusion_policy/config/task/pusht_image.yaml",
             "diffusion_policy/config/train_diffusion_unet_hybrid_workspace.yaml"]
    files = []
    contents = {}
    for name in names:
        path = base / name
        contents[name] = path.read_text(encoding="utf-8")
        files.append({"path": str(path.relative_to(ROOT)), "sha256": sha256(path)})
    readme = contents["README.md"].splitlines()
    evidence = [{"line": i + 1, "text": line.strip()} for i, line in enumerate(readme)
                if any(term in line for term in ["Linux machine with Nvidia GPU", "Ubuntu 20.04",
                                                 "MacOS and does not have full support", "--num-gpus=3",
                                                 "Launch training with seed 42", "fork` on linux"])]
    pins = []
    for number, line in enumerate(contents["conda_environment.yaml"].splitlines(), 1):
        if re.match(r"^\s*-\s*(python|cudatoolkit|pytorch|torchvision|gym|pymunk|numpy|diffusers|zarr|pygame|shapely)[=]", line):
            pins.append({"line": number, "requirement": line.strip()[2:]})
    config = yaml_scalars(contents["image_pusht_diffusion_policy_cnn.yaml"])
    requested = ["horizon", "n_obs_steps", "n_action_steps", "dataloader.batch_size",
                 "dataloader.num_workers", "training.device", "training.num_epochs",
                 "policy.num_inference_steps", "task.dataset.max_train_episodes", "task.dataset.val_ratio",
                 "task.dataset.zarr_path", "task.env_runner.n_train", "task.env_runner.n_test",
                 "task.env_runner.max_steps", "task.env_runner.legacy_test"]
    return {
        "schema_version": 1, "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "basis": "Read-only inspection of the locally downloaded official source files; no Torch import or training.",
        "files": files, "readme_evidence": evidence, "original_environment_pins": pins,
        "resolved_pusht_config_values": {key: config.get(key, {"missing": True}) for key in requested},
        "hardware_assessment": {
            "official_simulation_platform": "Linux with an NVIDIA GPU; README uses Ubuntu 20.04 as its installation example.",
            "native_windows_benchmark_support": "Not established by the inspected README; the current Windows data environment does not validate the original training stack.",
            "minimum_vram_gib": None, "minimum_ram_gib": None,
            "minimum_note": "No numeric minimum VRAM/RAM requirement was found in these README/environment/PushT configuration files. GPU availability alone does not prove the default batch/model fit.",
            "single_seed_vs_multiseed": "README supplies a single-seed cuda:0 command; the three-GPU Ray command is a multi-seed example, not a single-run minimum.",
        },
        "integration_gaps": [
            "The original conda environment pins Python 3.9 / PyTorch 1.12.1 / CUDA toolkit 11.6; this data-only Python environment is separate.",
            "The original PushT image dataset loader expects Zarr. The available LeRobot parquet/video and reconstructed NPZ need a deliberate adapter or conversion.",
            "The original config limits training to 90 episodes with val_ratio 0.02. These defaults differ from this project's episode-level 164/21/21 split and must not silently replace it.",
            "The original DP horizon is 16 with 2 observations and 8 executed actions; project contact windows of horizon 8 are a different convention.",
            "Model fit, CUDA execution, training speed, and policy quality remain untested. Do not report a GPU memory sufficiency claim from this inspection.",
        ],
    }


def inspect_raw():
    source_manifest = verify_manifest()
    if not source_manifest["all_passed"]:
        raise ValueError("Source manifest verification failed; refusing to interpret changed input data")
    paths = sorted(RAW.glob("data/chunk-*/*.parquet"))
    meta_paths = sorted(RAW.glob("meta/episodes/chunk-*/*.parquet"))
    if not paths or not meta_paths:
        raise FileNotFoundError("Raw data/episode parquet files are absent")
    table = pa.concat_tables([pq.read_table(path) for path in paths])
    metadata = pa.concat_tables([pq.read_table(path) for path in meta_paths]).to_pylist()
    info = json.loads((RAW / "meta/info.json").read_text(encoding="utf-8"))
    fields = [{"name": field.name, "arrow_type": str(field.type),
               "null_count": table[field.name].null_count} for field in table.schema]
    actions = np.asarray(table["action"].to_pylist(), dtype=float)
    pusher = np.asarray(table["observation.state"].to_pylist(), dtype=float)
    timestamps = table["timestamp"].to_numpy().astype(float)
    episode_ids = table["episode_index"].to_numpy()
    frame_ids = table["frame_index"].to_numpy()
    if actions.shape != (table.num_rows, 2) or pusher.shape != actions.shape:
        raise ValueError("Unexpected action/state dimensions; do not infer their semantics")
    episode_timestamp_checks = []
    for meta in metadata:
        start, end = int(meta["dataset_from_index"]), int(meta["dataset_to_index"])
        if not np.all(episode_ids[start:end] == meta["episode_index"]):
            raise ValueError("Episode IDs disagree with source metadata")
        episode_timestamp_checks.append({
            "episode_index": int(meta["episode_index"]), "frames": end - start,
            "frame_indices_contiguous": bool(np.array_equal(frame_ids[start:end], np.arange(end-start))),
            "timestamp_start": float(timestamps[start]),
            "timestamp_monotonic": bool((np.diff(timestamps[start:end]) > 0).all()),
            "timestamp_max_error_from_frame_index_over_fps": float(np.max(np.abs(timestamps[start:end] - np.arange(end-start)/info["fps"]))),
        })
    first_three, all_friction = [], []
    physics = None
    for meta in metadata[:3]:
        ep = int(meta["episode_index"])
        start, end = int(meta["dataset_from_index"]), int(meta["dataset_to_index"])
        env = FrictionProbe()
        observation, reset_info = env.reset(options={"reset_to_state": initial_state(ep)})
        if physics is None:
            physics = simulator_snapshot(env)
        env.friction_records.clear()  # Inspect action-period contacts separately from reset.
        errors = []
        for local_t, index in enumerate(range(start, end)):
            errors.append(float(np.linalg.norm(np.asarray(env.agent.position) - pusher[index])))
            env.action_index = local_t
            env.step(actions[index])
        differences = np.linalg.norm(actions[start:end] - pusher[start:end], axis=1)
        first_three.append({
            "episode_index": ep, "frames": end-start, "source_row_range_exclusive": [start, end],
            "first_three_rows": table.slice(start, min(3, end-start)).to_pylist(),
            "last_two_rows": table.slice(max(start, end-2), min(2, end-start)).to_pylist(),
            "timestamp_dt_statistics_seconds": stats(np.diff(timestamps[start:end])),
            "action_coordinate_min": actions[start:end].min(0).tolist(),
            "action_coordinate_max": actions[start:end].max(0).tolist(),
            "pusher_coordinate_min": pusher[start:end].min(0).tolist(),
            "pusher_coordinate_max": pusher[start:end].max(0).tolist(),
            "action_target_minus_current_pusher_distance": stats(differences),
            "action_equals_current_pusher_count": int(np.count_nonzero(differences < 1e-6)),
            "reset_input": initial_state(ep).tolist(), "actual_reset_observation": observation.tolist(),
            "replayed_pusher_pre_action_position_error": stats(errors),
            "observed_contact_arbiter_callbacks": len(env.friction_records),
        })
        all_friction.extend(env.friction_records)
        env.close()
    friction = [item["arbiter_friction"] for item in all_friction]
    restitution = [item["arbiter_restitution"] for item in all_friction]
    physics["observed_contact_probe"] = {
        "episodes": [item["episode_index"] for item in first_three],
        "arbiter_callback_count": len(all_friction),
        "unique_arbiter_friction": sorted(set(friction)),
        "unique_arbiter_restitution": sorted(set(restitution)),
        "all_friction_equals_shape_product": all(abs(item["arbiter_friction"] - item["shape_a_friction"] * item["shape_b_friction"]) < 1e-12 for item in all_friction),
        "first_ten_callbacks": all_friction[:10],
        "scope": "Observed in these first-three-episode action rollouts; no simulation properties were changed to force friction values.",
    }
    return {
        "schema_version": 1, "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "script_sha256": sha256(Path(__file__)), "source_manifest_check": source_manifest,
        "raw_data": {
            "parquet_files": [str(path.relative_to(ROOT)) for path in paths],
            "rows": table.num_rows, "episode_count": len(np.unique(episode_ids)),
            "fields": fields, "action_shape": list(actions.shape), "observation_state_shape": list(pusher.shape),
            "numeric_arrays_finite": bool(np.isfinite(actions).all() and np.isfinite(pusher).all() and np.isfinite(timestamps).all()),
            "actual_action_target_minus_current_pusher_distance": stats(np.linalg.norm(actions-pusher, axis=1)),
            "raw_object_pose_field_present": any("object" in field.name.lower() or "block_pose" in field.name.lower() for field in table.schema),
            "raw_contact_field_present": any("contact" in field.name.lower() for field in table.schema),
            "state_interpretation": "observation.state has two values. The first three actual rollouts verify alignment with the pusher position; it is not a complete object pose or simulator state.",
            "action_interpretation": "Action is a desired absolute pusher position. Its difference from current observed pusher position is expected under PD dynamics, not a dataset mismatch.",
            "metadata_codebase_version": info["codebase_version"], "metadata_fps": info["fps"],
            "image_metadata": info["features"]["observation.image"],
            "timestamp_checks": episode_timestamp_checks,
        },
        "first_three_episodes": first_three, "simulator_parameters": physics,
        "limitations": [
            "No previous report is used as evidence; this is a fresh read and a first-three-episode simulator probe.",
            "No full 206-episode video comparison was performed by this script.",
            "Object pose and contacts reconstructed by a simulator are not independently supplied annotations.",
            "The source table contains terminal flag/reward conventions that must be checked before treating the final action endpoint as observed.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    targets = [output / "data_inspection.json", output / "dp_requirements.json"]
    existing = [str(path) for path in targets if path.exists()]
    if existing:
        parser.error("Refusing to overwrite existing output: " + ", ".join(existing))
    output.mkdir(parents=True, exist_ok=True)
    reports = [inspect_raw(), inspect_dp()]
    for target, report in zip(targets, reports):
        with target.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
    report = reports[0]
    print(json.dumps({"outputs": list(map(str, targets)), "source_hashes_passed": report["source_manifest_check"]["all_passed"],
                      "rows": report["raw_data"]["rows"], "episodes": report["raw_data"]["episode_count"],
                      "contact_friction": report["simulator_parameters"]["observed_contact_probe"]["unique_arbiter_friction"]}))


if __name__ == "__main__":
    main()
