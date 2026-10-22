"""Deterministic, untrained approach/push/retreat demonstration.

The controller supplies world-coordinate targets at 10 Hz. The environment's
PD controller moves a kinematic circular pusher, and contact physics determines
the T's translation and rotation. No policy, reward optimization, or learned
contact prediction is used. Existing output directories are never overwritten.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import math
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from reconstruct_contacts import (
    ROOT, OUTLINE, EVENT_DTYPE, ContactReplayEnv, boundary_points,
    exterior_distance, velocities,
)


def font(size):
    for path in (Path("C:/Windows/Fonts/arial.ttf"),
                 Path("C:/Windows/Fonts/segoeui.ttf")):
        if path.is_file():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def arrow(draw, start, end, color, width=3):
    start, end = np.asarray(start), np.asarray(end)
    vector = end - start
    length = np.linalg.norm(vector)
    if length < 1:
        return
    direction = vector / length
    side = np.array([-direction[1], direction[0]])
    draw.line([tuple(start), tuple(end)], fill=color, width=width)
    head = [end, end - 9 * direction + 4 * side,
            end - 9 * direction - 4 * side]
    draw.polygon([tuple(x) for x in head], fill=color)


def transformed(points, pose):
    c, s = math.cos(pose[2]), math.sin(pose[2])
    rotation = np.array([[c, -s], [s, c]])
    return points @ rotation.T + pose[:2]


def make_frame(index, states, poses, cog, points, actions, phases, labels,
               events, local_contact, pusher_radius):
    """Draw directly from saved physical coordinates, without modifying physics."""
    canvas = Image.new("RGB", (1020, 650), "#f3f5f8")
    draw = ImageDraw.Draw(canvas)
    title, regular, small = font(23), font(17), font(12)
    draw.text((22, 13), "PushT: untrained approach / push / retreat", fill="#18334e", font=title)
    draw.text((22, 47), f"t = {index / 10:.1f} s | state {index}/{len(actions)} | "
              f"phase: {phases[index - 1] if index else 'initial'}", fill="#18334e", font=regular)
    offset = np.array([24.0, 91.0])

    def xy(p):
        return tuple(np.asarray(p) + offset)

    draw.rectangle((24, 91, 536, 603), fill="white", outline="#9aa9b6", width=2)
    for wall in ([(5, 5), (506, 5)], [(5, 5), (5, 506)],
                 [(5, 506), (506, 506)], [(506, 5), (506, 506)]):
        draw.line([xy(p) for p in wall], fill="#d3d3d3", width=4)
    initial_outline = transformed(OUTLINE, poses[0])
    draw.line([xy(p) for p in np.vstack((initial_outline, initial_outline[0]))],
              fill="#aebac5", width=2)
    polygon = transformed(OUTLINE, poses[index])
    draw.polygon([xy(p) for p in polygon], fill="#b5c4d2", outline="#526a80", width=2)

    if index:
        draw.line([xy(p) for p in states[:index + 1, :2]], fill="#2878cf", width=2)
        draw.line([xy(p) for p in cog[:index + 1]], fill="#9137a8", width=3)
        # Contacts are world locations at the collision substep, not projected
        # onto a later T pose. This keeps event timing explicit.
        preceding = events[events["action_index"] == index - 1]
        for event in preceding:
            x, y = xy(event["point_world"])
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill="#f37824")
        for event in preceding[::max(1, len(preceding) // 3)]:
            start = np.asarray(xy(event["point_world"]))
            arrow(draw, start, start + 25 * event["normal_into_block"], "#b54411", 2)
        draw.line([xy(actions[index - 1] - [5, 0]), xy(actions[index - 1] + [5, 0])],
                  fill="#cd2948", width=2)
        draw.line([xy(actions[index - 1] - [0, 5]), xy(actions[index - 1] + [0, 5])],
                  fill="#cd2948", width=2)
    x, y = xy(states[index, :2])
    draw.ellipse((x - pusher_radius, y - pusher_radius, x + pusher_radius,
                  y + pusher_radius), fill="#417ae2", outline="#17458d", width=2)
    scores = labels[index - 1] if index else np.zeros(points.shape[1])
    for number, (p, score) in enumerate(zip(points[index], scores)):
        color = tuple(int(a * (1 - score) + b * score)
                      for a, b in zip((52, 114, 139), (246, 87, 27)))
        x, y = xy(p)
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=color)
        if number % 4 == 0:
            draw.text((x + 3, y + 2), str(number), fill="#153a51", font=small)
    chosen = transformed(np.asarray([local_contact]), poses[index])[0]
    x, y = xy(chosen)
    draw.rectangle((x - 5, y - 5, x + 5, y + 5), outline="#db2556", width=2)
    draw.text((x - 26, y - 27), "chosen", fill="#b11a45", font=small)
    x, y = xy(cog[index])
    draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill="#9137a8")
    origin = poses[index, :2]
    arrow(draw, xy(origin), xy(origin + [30, 0]), "#474747", 2)
    arrow(draw, xy(origin), xy(origin + [0, 30]), "#474747", 2)
    draw.text(xy(origin + [32, -7]), "x", fill="#474747", font=small)
    draw.text(xy(origin + [-3, 30]), "y", fill="#474747", font=small)

    delta = cog[index] - cog[0]
    angle_change = math.degrees(poses[index, 2] - poses[0, 2])
    x = 568
    lines = [
        "Controller (specified)",
        "Local contact: (-60, 15)",
        "Approach: +81 units / 2.0 s",
        "Wait: 0.5 s", "Push: +60 units along world +x / 3.0 s",
        "Wait: 0.5 s", "Retreat: -35 units / 1.0 s", "",
        "Physics (observed)",
        f"CoG displacement: ({delta[0]:.2f}, {delta[1]:.2f})",
        f"Angle change: {angle_change:.2f} degrees",
        f"Preceding interval contact: {bool(index and np.any(scores > 0))}", "",
        "Blue: actual pusher center / trail",
        "Purple: physical center of gravity / trail",
        "Pink square: chosen material point",
        "Red cross: commanded pusher target",
        "Orange: measured contact / inward normal",
        "Colored dots: 64 fixed material points",
        "Labels show every fourth point ID.",
        "Soft score uses the preceding 0.1 s interval.", "",
        "Units are simulator coordinates, not meters.",
        "This is an interface check, not a learned policy.",
    ]
    yy = 94
    for line in lines:
        draw.text((x, yy), line, fill="#203b54", font=regular)
        yy += 21
    draw.text((25, 616), "x right, y down; positive angle rotates clockwise on this image. No goal-task success claim.",
              fill="#40586d", font=regular)
    return canvas


def controller_targets(start, contact_world, radius):
    before_contact = contact_world - np.array([radius + 4, 0])
    push_end = before_contact + np.array([60.0, 0.0])
    retreat_end = push_end - np.array([35.0, 0.0])
    targets, phases = [], []
    for phase, begin, end, count in (
        ("approach", start, before_contact, 20),
        ("wait_before_push", before_contact, before_contact, 5),
        ("push", before_contact, push_end, 30),
        ("wait_after_push", push_end, push_end, 5),
        ("retreat", push_end, retreat_end, 10),
    ):
        for fraction in np.arange(1, count + 1) / count:
            targets.append(begin + fraction * (end - begin))
            phases.append(phase)
    parameters = {
        "controller_kind": "fixed open-loop piecewise-linear world target path; no trained model",
        "action_semantics": "absolute desired pusher center [x,y], tracked by environment PD controller",
        "contact_selection": "manually specified point on left exterior edge of T crossbar",
        "desired_push_direction_world": [1, 0],
        "approach_start_world": start.tolist(),
        "precontact_target_world": before_contact.tolist(),
        "push_end_target_world": push_end.tolist(),
        "retreat_end_target_world": retreat_end.tolist(),
        "approach": {"duration_s": 2.0, "displacement_world": (before_contact - start).tolist(),
                     "target_path_speed_units_per_s": float(np.linalg.norm(before_contact - start) / 2)},
        "wait_before_push_duration_s": 0.5,
        "push": {"duration_s": 3.0, "displacement_world": [60, 0], "target_path_speed_units_per_s": 20},
        "wait_after_push_duration_s": 0.5,
        "retreat": {"duration_s": 1.0, "displacement_world": [-35, 0], "target_path_speed_units_per_s": 35},
        "note": "Speeds describe target interpolation, not exact instantaneous physical pusher velocity.",
    }
    return np.array(targets), np.array(phases), parameters


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="New output directory; an existing directory causes an error")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)

    env = ContactReplayEnv()
    reset_input = np.array([100.0, 215.0, 260.0, 200.0, 0.0])
    env.reset(seed=0, options={"reset_to_state": reset_input})
    local_contact = np.array([-60.0, 15.0])
    local_points = boundary_points(64)
    sigma = 5.0
    pusher_radius = float(next(iter(env.agent.shapes)).radius)
    contact_world = np.array(env.block.local_to_world(tuple(local_contact)))
    actions, phases, controller = controller_targets(np.array(env.agent.position), contact_world, pusher_radius)
    count = len(actions)
    states = np.zeros((count + 1, 5), dtype=np.float64)
    poses = np.zeros((count + 1, 3), dtype=np.float64)
    cog = np.zeros((count + 1, 2), dtype=np.float64)
    vels = np.zeros((count + 1, 5), dtype=np.float64)
    points = np.zeros((count + 1, 64, 2), dtype=np.float64)
    labels = np.zeros((count, 64), dtype=np.float32)
    valid = np.ones(count, dtype=bool)
    has_contact = np.zeros(count, dtype=bool)
    all_events = []

    def capture(t):
        states[t] = env.get_obs()
        poses[t] = [*env.block.position, env.block.angle]
        cog[t] = env.block.local_to_world(env.block.center_of_gravity)
        vels[t] = velocities(env)
        points[t] = [env.block.local_to_world(tuple(p)) for p in local_points]

    physics = {
        "block_mass": float(env.block.mass), "block_moment": float(env.block.moment),
        "block_center_of_gravity_local": list(env.block.center_of_gravity),
        "block_shape_friction": [float(s.friction) for s in env._block_shapes],
        "pusher_shape_friction": [float(s.friction) for s in env.agent.shapes],
        "wall_shape_friction": [float(s.friction) for s in env.space.shapes if s.body is env.space.static_body],
        "space_damping": float(env.space.damping), "space_gravity": list(env.space.gravity),
        "physics_step_s": env.dt, "control_hz": env.control_hz,
        "substeps_per_control": int(1 / (env.dt * env.control_hz)),
        "pd_kp": env.k_p, "pd_kv": env.k_v,
        "pusher_body_type": "kinematic", "pusher_radius": pusher_radius,
        "block_body_type": "dynamic",
        "parameters_modified_by_demo": False,
    }
    try:
        for t, action in enumerate(actions):
            capture(t)
            env.action_index = t
            env.step(action)
            events = np.array(env.contact_events, dtype=EVENT_DTYPE)
            has_contact[t] = bool(len(events))
            if len(events):
                if not np.all((events["substep"] >= 0) & (events["substep"] < 10)):
                    raise AssertionError("Contact substep out of bounds")
                events["distance_to_exterior"] = exterior_distance(events["point_local"])
                events["on_exterior"] = events["distance_to_exterior"] < 1e-4
                valid[t] = events["on_exterior"].all()
                valid_events = events[events["on_exterior"]]
                if len(valid_events):
                    squared = ((local_points[:, None] - valid_events["point_local"][None]) ** 2).sum(-1)
                    labels[t] = np.exp(-squared.min(1) / (2 * sigma ** 2))
                all_events.append(events)
        capture(count)
    finally:
        env.close()
    events = np.concatenate(all_events) if all_events else np.empty(0, dtype=EVENT_DTYPE)
    if not np.all(np.isfinite(states)) or not len(events):
        raise AssertionError("Demo requires finite physical states and actual collision contacts")
    displacement = cog[-1] - cog[0]
    if np.linalg.norm(displacement) < 1:
        raise AssertionError("Demo did not move the object by at least one coordinate unit")
    if not np.all(labels[~has_contact] == 0):
        raise AssertionError("Non-contact intervals must have all-zero labels")
    geometry_error = max(float(np.abs(points[t] - transformed(local_points, poses[t])).max())
                         for t in range(count + 1))
    if geometry_error > 1e-9:
        raise AssertionError("Rigid material-point transformation failed")
    arrays = {
        "reset_input": reset_input, "time_s": np.arange(count + 1) / env.control_hz,
        "states": states, "block_pose": poses, "block_center_of_gravity_world": cog,
        "pusher_position_actual": states[:, :2], "velocities": vels,
        "actions": actions, "phases": phases, "local_boundary_points": local_points,
        "points_world": points, "chosen_contact_local": local_contact,
        "contact_events": events, "contact_soft_labels": labels,
        "contact_label_valid": valid, "has_contact": has_contact,
    }
    np.savez_compressed(output / "simple_push.npz", **arrays)
    # Hash numerical fields individually; zip timestamps must not influence the
    # reproducibility comparison. Unicode phase strings are deterministic too.
    field_hashes = {name: hashlib.sha256(value.tobytes()).hexdigest() for name, value in arrays.items()}
    report = {
        "status": "verified untrained contact-and-motion interface demonstration",
        "claim_scope": "not a reconstructed public demonstration; not a learned policy; not a goal-task success result",
        "schema_version": 1, "control_steps": count, "duration_s": count / env.control_hz,
        "initial_reset_input": reset_input.tolist(), "initial_actual_state": states[0].tolist(),
        "final_actual_state": states[-1].tolist(), "controller": controller, "physics": physics,
        "coordinates": {"length_unit": "simulator coordinate unit (512-unit scene), no real-world meter calibration",
                        "axes": "x right, y down in saved images", "angle_unit": "radian",
                        "positive_angle": "clockwise in saved images",
                        "block_pose_reference": "body local origin, not center of gravity",
                        "block_pose_angle": "unwrapped; states[:,4] uses modulo 2*pi",
                        "point_transform": "p_world = R(theta) @ p_local + body.position"},
        "label_semantics": {"kind": "simulated geometric contacts, not optimality or success probabilities",
                            "sigma_local_units": sigma,
                            "formula": "exp(-min_squared_distance_to_valid_contact/(2*sigma^2))",
                            "time_alignment": "states[t] precedes actions[t]; labels[t] aggregates its ten physics substeps; states[t+1] follows",
                            "no_contact": "has_contact[t] is false and all 64 labels are zero",
                            "normal": "points inward into the block",
                            "impulse": "impulse_on_block is manifold total, duplicated over manifold points; deduplicate before summing"},
        "observed": {
            "block_origin_displacement_world": (poses[-1, :2] - poses[0, :2]).tolist(),
            "block_cog_displacement_world": displacement.tolist(),
            "block_cog_displacement_norm": float(np.linalg.norm(displacement)),
            "block_angle_change_rad": float(poses[-1, 2] - poses[0, 2]),
            "block_angle_change_degrees": math.degrees(poses[-1, 2] - poses[0, 2]),
            "pusher_displacement_world": (states[-1, :2] - states[0, :2]).tolist(),
            "contact_point_events": len(events), "contact_control_intervals": int(has_contact.sum()),
            "first_contact_action_index": int(events["action_index"].min()),
            "last_contact_action_index": int(events["action_index"].max()),
            "internal_contact_events": int((~events["on_exterior"]).sum()),
            "max_rigid_point_transform_abs_error": geometry_error,
            "no_contact_labels_are_zero": bool(np.all(labels[~has_contact] == 0)),
        },
        "fields": {name: {"shape": list(value.shape), "dtype": str(value.dtype)} for name, value in arrays.items()},
        "field_sha256": field_hashes,
        "versions": {"python": sys.version, "numpy": importlib.metadata.version("numpy"),
                     "pymunk": importlib.metadata.version("pymunk"),
                     "gymnasium": importlib.metadata.version("gymnasium"),
                     "pillow": importlib.metadata.version("pillow")},
        "source_sha256": {name: hashlib.sha256((ROOT / "scripts" / name).read_bytes()).hexdigest()
                          for name in ("simple_push.py", "reconstruct_contacts.py")},
    }
    (output / "simple_push.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    frames = [make_frame(i, states, poses, cog, points, actions, phases, labels,
                         events, local_contact, pusher_radius) for i in range(count + 1)]
    frames[0].save(output / "initial.png")
    frames[int(np.median(events["action_index"])) + 1].save(output / "during_contact.png")
    frames[-1].save(output / "final.png")
    frames[0].save(output / "simple_push.gif", save_all=True, append_images=frames[1:],
                   duration=100, loop=0, optimize=False)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps({"output_dir": str(output), "observed": report["observed"],
                      "steps": count, "duration_s": report["duration_s"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
