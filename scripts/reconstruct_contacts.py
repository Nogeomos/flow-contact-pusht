"""Replay the public PushT demonstrations and export simulator contact labels.

State/point sample t is BEFORE action t. Contacts[t] come from the ten
physics substeps while action t executes. They are reconstructed simulation
labels, not provided annotations or proof of unique optimal contacts.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "external/gym-pusht"))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import av
import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw
from gym_pusht.envs.pusht import PushTEnv

RAW = ROOT / "data/raw/lerobot_pusht"
# The exterior outline excludes the internal seam between the two rectangles.
OUTLINE = np.array([[-60, 0], [60, 0], [60, 30], [15, 30],
                    [15, 120], [-15, 120], [-15, 30], [-60, 30]], dtype=float)
EVENT_DTYPE = np.dtype([
    ("action_index", "i4"), ("substep", "i2"),
    ("manifold_index", "i2"),
    ("point_world", "f8", (2,)), ("point_local", "f8", (2,)),
    ("normal_into_block", "f8", (2,)),
    ("arbiter_total_impulse", "f8", (2,)),
    ("impulse_on_block", "f8", (2,)),
    ("block_is_shape_a", "?"), ("signed_separation", "f8"),
    ("distance_to_exterior", "f8"), ("on_exterior", "?"),
])


def initial_state(episode_index):
    rs = np.random.RandomState(episode_index)
    return np.array([rs.randint(50, 450), rs.randint(50, 450),
                     rs.randint(100, 400), rs.randint(100, 400),
                     rs.randn() * 2 * np.pi - np.pi])


def boundary_points(n):
    edges = np.roll(OUTLINE, -1, axis=0) - OUTLINE
    lengths = np.linalg.norm(edges, axis=1)
    cumulative = np.r_[0, np.cumsum(lengths)]
    distance = np.arange(n) * cumulative[-1] / n
    segment = np.searchsorted(cumulative[1:], distance, side="right")
    fraction = (distance - cumulative[segment]) / lengths[segment]
    return OUTLINE[segment] + edges[segment] * fraction[:, None]


def exterior_distance(points):
    edges = np.roll(OUTLINE, -1, axis=0) - OUTLINE
    offset = points[:, None, :] - OUTLINE[None, :, :]
    fraction = np.clip((offset * edges).sum(-1) / (edges * edges).sum(-1), 0, 1)
    residual = offset - fraction[..., None] * edges
    return np.sqrt((residual * residual).sum(-1).min(-1))


def velocities(env):
    return [*env.agent.velocity, *env.block.velocity, env.block.angular_velocity]


class ContactReplayEnv(PushTEnv):
    def __init__(self):
        self.contact_events = []
        self.action_index = -1
        self.substep = -1
        self.manifold_index = 0
        super().__init__()

    def _setup(self):
        super()._setup()
        original_step = self.space.step

        def tracked_step(dt):
            self.substep += 1
            original_step(dt)

        self.space.step = tracked_step

    def step(self, action):
        self.contact_events = []
        self.substep = -1
        self.manifold_index = 0
        return super().step(action)

    def _handle_collision(self, arbiter, space, data):
        super()._handle_collision(arbiter, space, data)
        a, b = arbiter.shapes
        if not ((a.body is self.agent and b.body is self.block)
                or (a.body is self.block and b.body is self.agent)):
            return
        block_is_a = a.body is self.block
        contact_set = arbiter.contact_point_set
        inward = np.array(contact_set.normal) * (-1 if block_is_a else 1)
        impulse = np.array(arbiter.total_impulse)
        impulse_on_block = impulse * (1 if block_is_a else -1)
        self.manifold_index += 1
        for point in contact_set.points:
            on_block = point.point_a if block_is_a else point.point_b
            self.contact_events.append((
                self.action_index, self.substep, self.manifold_index, tuple(on_block),
                tuple(self.block.world_to_local(on_block)), tuple(inward),
                tuple(impulse), tuple(impulse_on_block), block_is_a, point.distance,
                np.nan, False,
            ))


def strict_color_mask(rgb):
    # Approximate visible T mask from its known renderer color. AV1 compression
    # means this is a visual consistency check, not exact ground-truth pose error.
    rgb = rgb.astype(np.int16)
    delta = np.abs(rgb - np.array([119, 136, 153]))
    return (delta.max(axis=-1) < 38) & ((rgb[..., 2] - rgb[..., 1]) < 42)


def block_mask(rgb):
    # The renderer lightens the T interior from (119,136,153) to (143,163,184).
    # Nearest-palette assignment accounts for both colors instead of clipping
    # compressed interior pixels at the edge of a fixed single-color threshold.
    palette = np.array([[119, 136, 153], [143, 163, 184], [255, 255, 255],
                        [211, 211, 211], [144, 238, 144], [65, 105, 225],
                        [78, 126, 255]], dtype=np.float32)
    squared = ((rgb.astype(np.float32)[..., None, :] - palette) ** 2).sum(-1)
    return squared.argmin(-1) < 2


def overlap(a, b):
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 1.0


def make_panel(reference, replay, env, events, episode, frame, iou):
    canvas = Image.new("RGB", (768, 440), "#f3f5f8")
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 12), f"Episode {episode} | frame {frame} | visible T IoU {iou:.3f}", fill="#17243a")
    draw.text((20, 42), "Public demonstration", fill="#17243a")
    draw.text((404, 42), "Physics replay + reconstructed contacts", fill="#17243a")
    canvas.paste(Image.fromarray(reference).resize((360, 360)), (12, 66))
    canvas.paste(Image.fromarray(replay).resize((360, 360)), (396, 66))
    # Project local contact points onto the current replay pose to visualize the
    # regions visited during the preceding action, rather than stale world points.
    for event in events:
        world = np.array(env.block.local_to_world(event["point_local"].tolist()))
        x, y = np.array([396, 66]) + world / 512 * 360
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill="#f47727")
    return canvas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-episodes", type=int, default=206)
    parser.add_argument("--points", type=int, default=64)
    parser.add_argument("--sigma", type=float, default=5.0)
    args = parser.parse_args()
    if args.points < 8 or args.sigma <= 0:
        raise ValueError("Require at least 8 points and a positive sigma")
    table = pq.read_table(RAW / "data/chunk-000/file-000.parquet")
    metadata = pq.read_table(RAW / "meta/episodes/chunk-000/file-000.parquet").to_pylist()
    if not 1 <= args.max_episodes <= len(metadata):
        raise ValueError(f"--max-episodes must be between 1 and {len(metadata)}")
    actions = np.asarray(table["action"].to_pylist(), dtype=np.float64)
    recorded_agent = np.asarray(table["observation.state"].to_pylist(), dtype=np.float64)
    frame_ids = table["frame_index"].to_numpy()
    episode_ids = table["episode_index"].to_numpy()
    video_path = RAW / "videos/observation.image/chunk-000/file-000.mp4"
    container = av.open(str(video_path))
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"
    video = iter(container.decode(video=0))
    local_points = boundary_points(args.points)
    output = ROOT / "data/derived/replayed_contacts"
    report_dir = ROOT / "reports"
    if args.max_episodes != len(metadata):
        output = ROOT / f"data/derived/probe_{args.max_episodes}_episodes"
        report_dir = report_dir / f"probe_{args.max_episodes}_episodes"
    output.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    summaries, preview_frames = [], []
    worst_frames = []
    decoded = 0
    previous_events = np.empty(0, dtype=EVENT_DTYPE)
    for meta in metadata[:args.max_episodes]:
        episode = meta["episode_index"]
        start, end = meta["dataset_from_index"], meta["dataset_to_index"]
        length = end - start
        assert start == decoded
        assert np.all(episode_ids[start:end] == episode)
        assert np.array_equal(frame_ids[start:end], np.arange(length))
        expected_start_time = meta["videos/observation.image/from_timestamp"]
        env = ContactReplayEnv()
        env.reset(options={"reset_to_state": initial_state(episode)})
        states = np.zeros((length + 1, 5), dtype=np.float64)
        state_velocities = np.zeros((length + 1, 5), dtype=np.float64)
        world_points = np.zeros((length + 1, args.points, 2), dtype=np.float32)
        labels = np.zeros((length, args.points), dtype=np.float32)
        label_valid = np.ones(length, dtype=bool)
        observed_transition = np.ones(length, dtype=bool)
        observed_transition[-1] = False
        all_events, ious, strict_ious, agent_errors = [], [], [], []
        previous_events = np.empty(0, dtype=EVENT_DTYPE)
        for t, index in enumerate(range(start, end)):
            video_frame = next(video)
            video_time = float(video_frame.pts * video_frame.time_base)
            assert abs(video_time - (expected_start_time + t / 10)) < 0.02
            reference = video_frame.to_ndarray(format="rgb24")
            decoded += 1
            states[t] = env.get_obs()
            state_velocities[t] = velocities(env)
            world_points[t] = [env.block.local_to_world(p.tolist()) for p in local_points]
            replay = env._render()
            iou = overlap(block_mask(reference), block_mask(replay))
            ious.append(iou)
            strict_ious.append(overlap(strict_color_mask(reference), strict_color_mask(replay)))
            agent_errors.append(float(np.linalg.norm(states[t, :2] - recorded_agent[index])))
            if len(worst_frames) < 6 or iou < worst_frames[-1][0]:
                panel = make_panel(reference, replay, env, previous_events, episode, t, iou)
                worst_frames.append((iou, episode, t, panel))
                worst_frames.sort(key=lambda item: item[0])
                worst_frames = worst_frames[:6]
            if episode == 0 and t % 3 == 0:
                preview_frames.append(make_panel(reference, replay, env, previous_events, episode, t, iou))
            env.action_index = t
            if t == length - 1:
                last_observed_coverage = float(env._get_coverage())
            env.step(actions[index])
            events = np.array(env.contact_events, dtype=EVENT_DTYPE)
            if len(events):
                assert events["substep"].min() >= 0 and events["substep"].max() < 10
                events["distance_to_exterior"] = exterior_distance(events["point_local"])
                events["on_exterior"] = events["distance_to_exterior"] < 1e-4
                valid_events = events[events["on_exterior"]]
                label_valid[t] = bool(events["on_exterior"].all())
                if len(valid_events):
                    squared = ((local_points[:, None, :] - valid_events["point_local"][None, :, :]) ** 2).sum(-1)
                    labels[t] = np.exp(-squared.min(axis=1) / (2 * args.sigma ** 2))
                all_events.append(events)
            previous_events = events
        states[length] = env.get_obs()
        state_velocities[length] = velocities(env)
        world_points[length] = [env.block.local_to_world(p.tolist()) for p in local_points]
        events = np.concatenate(all_events) if all_events else np.empty(0, dtype=EVENT_DTYPE)
        np.savez_compressed(
            output / f"episode_{episode:06d}.npz",
            episode_index=episode, reset_input=initial_state(episode),
            states=states, velocities=state_velocities,
            local_boundary_points=local_points, points_world=world_points,
            actions=actions[start:end], contact_events=events, contact_soft_labels=labels,
            contact_label_valid=label_valid, transition_has_video_endpoint=observed_transition,
            reference_visible_mask_iou=np.array(ious), agent_position_error=np.array(agent_errors),
            reference_strict_color_mask_iou=np.array(strict_ious),
        )
        summary = {"episode": episode, "frames": length, "contact_point_events": len(events),
                   "contact_action_intervals": int(np.count_nonzero(labels.max(axis=1) > 0)),
                   "visible_mask_iou_mean": float(np.mean(ious)),
                   "strict_color_mask_iou_mean": float(np.mean(strict_ious)),
                   "visible_mask_iou_min": float(np.min(ious)),
                   "visible_mask_iou_p05": float(np.quantile(ious, 0.05)),
                   "frames_iou_below_0_8": int(np.count_nonzero(np.array(ious) < 0.8)),
                   "agent_position_error_max": float(np.max(agent_errors)),
                   "internal_seam_events": int(np.count_nonzero(~events["on_exterior"])),
                   "invalid_contact_label_intervals": int(np.count_nonzero(~label_valid)),
                   "zero_impulse_point_events": int(np.count_nonzero(np.linalg.norm(events["impulse_on_block"],axis=1) < 1e-12)),
                   "last_observed_replayed_coverage": last_observed_coverage,
                   "post_final_action_coverage_unverified": float(env._get_coverage())}
        summaries.append(summary)
        env.close()
        if episode % 10 == 0 or episode == args.max_episodes - 1:
            print(json.dumps(summary), flush=True)
    if len(summaries) == len(metadata):
        assert next(video, None) is None, "Unexpected trailing video frames"
    container.close()
    resource_manifest = json.loads((ROOT / "resource_manifest.json").read_text(encoding="utf-8"))
    summary = {
        "episodes": len(summaries), "frames": decoded,
        "contact_point_events": sum(s["contact_point_events"] for s in summaries),
        "contact_action_intervals": sum(s["contact_action_intervals"] for s in summaries),
        "internal_seam_events": sum(s["internal_seam_events"] for s in summaries),
        "invalid_contact_label_intervals": sum(s["invalid_contact_label_intervals"] for s in summaries),
        "zero_impulse_point_events": sum(s["zero_impulse_point_events"] for s in summaries),
        "visible_mask_iou_mean": float(np.average([s["visible_mask_iou_mean"] for s in summaries],
                                                   weights=[s["frames"] for s in summaries])),
        "strict_color_mask_iou_mean": float(np.average([s["strict_color_mask_iou_mean"] for s in summaries],
                                                   weights=[s["frames"] for s in summaries])),
        "visible_mask_iou_min": min(s["visible_mask_iou_min"] for s in summaries),
        "agent_position_error_max": max(s["agent_position_error_max"] for s in summaries),
        "frames_iou_below_0_8": sum(s["frames_iou_below_0_8"] for s in summaries),
        "dataset_revision": resource_manifest["dataset_revision"],
        "repositories": resource_manifest["repositories"],
        "configuration": vars(args),
        "data_schema_version": 2,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python_version": sys.version,
        "packages": {name: importlib.metadata.version(name) for name in
                     ["numpy", "pymunk", "gymnasium", "pygame", "shapely", "opencv-python", "av", "pyarrow"]},
        "normal_convention": "World-space normal pointing from pusher toward block.",
        "impulse_convention": "arbiter_total_impulse acts on shape A; impulse_on_block corrects its sign. Both are whole-manifold impulses repeated for each contact point. Deduplicate by episode, action_index, manifold_index before summing.",
        "label_convention": "Index t: contact regions visited during action t; geometry states[t] is before action t.",
        "limitations": ["Contacts are reconstructed from a pinned simulator, not independent ground truth.",
                       "Mask overlap checks visible geometry at 96x96 after lossy video decoding; it is not exact pose error.",
                       "Soft labels use local Euclidean distance; concave-surface geodesic smoothing is a future option.",
                       "Internal seam events are retained and flagged; their action intervals are excluded from training windows.",
                       "The last action endpoint has no reference video frame and is excluded from training windows.",
                       "Zero-impulse contacts remain geometric contact labels, not effective-push labels.",
                       "Do not sum repeated arbiter_total_impulse across points in a manifold.",
                       "No model has been trained; replay performance is not learned-policy performance."],
        "per_episode": summaries,
    }
    (report_dir / "replay_validation.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for rank, (iou, episode, t, panel) in enumerate(worst_frames):
        panel.save(report_dir / f"worst_{rank:02d}_episode_{episode:03d}_frame_{t:03d}.png")
    if preview_frames:
        preview_frames[0].save(report_dir / "replay_preview.gif", save_all=True,
                               append_images=preview_frames[1:], duration=300, loop=0)
        preview_frames[len(preview_frames)//2].save(report_dir / "replay_comparison.png")
    print("SUMMARY " + json.dumps({k: v for k, v in summary.items() if k != "per_episode"}), flush=True)


if __name__ == "__main__":
    main()
