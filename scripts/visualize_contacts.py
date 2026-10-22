"""Inspect reconstructed contacts and offline demonstration flow, without training.

Only numpy and Pillow are needed. Output directories must not already exist.
Contact interval t spans the ten physics substeps of action t; states[t] is
the observation BEFORE that action. Local contact positions projected into
states[t] illustrate material locations, not simultaneous world contacts.
"""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
from PIL import Image, ImageDraw, ImageFont


OUTLINE = np.array([[-60, 0], [60, 0], [60, 30], [15, 30],
                    [15, 120], [-15, 120], [-15, 30], [-60, 30]], dtype=float)
BG = "#f4f6fa"
INK = "#13243b"
BLUE = "#3569dd"
ORANGE = "#e67116"
GREEN = "#14835e"
PURPLE = "#883fbb"
FONT = next((p for p in [Path("C:/Windows/Fonts/msyh.ttc"),
                         Path("C:/Windows/Fonts/simhei.ttf")]
             if p.is_file()), None)


def font(size):
    if FONT is None:
        raise RuntimeError("A Chinese font is required: install Microsoft YaHei or SimHei.")
    return ImageFont.truetype(str(FONT), size)


FONTS = {size: font(size) for size in [16, 18, 20, 22, 24, 28, 36]}


def transform(local, state):
    theta = float(state[4])
    rotation = np.array([[np.cos(theta), -np.sin(theta)],
                         [np.sin(theta), np.cos(theta)]])
    return np.asarray(local) @ rotation.T + state[2:4]


def arrow(draw, start, end, color, width=3, tip=10):
    start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    delta = end - start
    length = np.linalg.norm(delta)
    if length < 0.8:
        return
    direction = delta / length
    normal = np.array([-direction[1], direction[0]])
    head = min(tip, length * 0.45)
    draw.line([tuple(start), tuple(end)], fill=color, width=width)
    points = [end, end - head * direction + 0.45 * head * normal,
              end - head * direction - 0.45 * head * normal]
    draw.polygon([tuple(p) for p in points], fill=color)


def cross(draw, point, color, radius=9, width=4):
    x, y = point
    draw.line((x-radius, y-radius, x+radius, y+radius), fill=color, width=width)
    draw.line((x-radius, y+radius, x+radius, y-radius), fill=color, width=width)


def heat(score):
    low, high = np.array([74, 139, 174]), np.array([239, 76, 34])
    return tuple(np.rint(low * (1-score) + high * score).astype(int))


def outward_normals(local):
    edges = np.roll(OUTLINE, -1, axis=0) - OUTLINE
    offsets = local[:, None, :] - OUTLINE[None, :, :]
    fractions = np.clip((offsets * edges).sum(-1) / (edges * edges).sum(-1), 0, 1)
    distance = np.linalg.norm(offsets - fractions[..., None] * edges, axis=-1)
    indices = distance.argmin(1)
    normals = np.stack([edges[:, 1], -edges[:, 0]], axis=1)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    return normals[indices]


def draw_snapshot(data, index, horizon):
    image = Image.new("RGB", (1920, 1270), BG)
    draw = ImageDraw.Draw(image)
    state = data["states"][index]
    points = data["points_world"][index]
    local = data["local_boundary_points"]
    scores = data["contact_soft_labels"][index]
    events = data["contact_events"]
    events = events[events["action_index"] == index]
    exterior = events[events["on_exterior"]]
    future = data["points_world"][index + horizon]
    displacement = np.linalg.norm(future - points, axis=1)
    episode = int(data["episode_index"])
    draw.text((40, 22), "PushT：固定轮廓点、示范运动与接触标签", font=FONTS[36], fill=INK)
    draw.text((40, 78), f"示范 {episode:03d}  |  动作区间 t = {index}  |  未来 {horizon} 步 = {horizon / 10:.1f} 秒  |  接触事件 {len(events)} 条",
              font=FONTS[24], fill=INK)
    draw.rounded_rectangle((30, 131, 905, 1039), 18, fill="white", outline="#d8e0ea", width=2)
    draw.rounded_rectangle((935, 131, 1890, 1039), 18, fill="white", outline="#d8e0ea", width=2)
    draw.text((55, 148), "世界坐标：当前状态与未来示范轨迹", font=FONTS[28], fill=INK)
    draw.text((960, 148), f"物体局部坐标：{len(local)} 个固定编号点", font=FONTS[28], fill=INK)
    offset, scale = np.array([58, 218]), 1.60

    def screen(world):
        return offset + np.asarray(world) * scale

    draw.rectangle((*offset, *(offset + 512 * scale)), outline="#c6d1de", width=3)
    for value in [128, 256, 384]:
        draw.line([tuple(screen([value, 0])), tuple(screen([value, 512]))], fill="#edf1f5")
        draw.line([tuple(screen([0, value])), tuple(screen([512, value]))], fill="#edf1f5")
    future_outline = screen(transform(OUTLINE, data["states"][index + horizon]))
    draw.line([*map(tuple, future_outline), tuple(future_outline[0])], fill="#55b797", width=3)
    current_outline = screen(transform(OUTLINE, state))
    draw.polygon(list(map(tuple, current_outline)), fill="#d8e2ec", outline="#61798f", width=3)
    # Every eighth material point gets a true, unscaled displacement arrow.
    for point_id in range(0, len(local), max(1, len(local)//8)):
        arrow(draw, screen(points[point_id]), screen(future[point_id]), GREEN, width=3)
    for point, score in zip(points, scores):
        x, y = screen(point)
        draw.ellipse((x-5, y-5, x+5, y+5), fill=heat(float(score)), outline="white")
    px, py = screen(state[:2])
    radius = 15 * scale
    draw.ellipse((px-radius, py-radius, px+radius, py+radius), fill="#c8d9ff", outline=BLUE, width=3)
    cross(draw, (px, py), BLUE, radius=5, width=2)
    target = screen(data["actions"][index])
    draw.line([(px, py), tuple(target)], fill=PURPLE, width=2)
    cross(draw, target, PURPLE)
    origin = screen(state[2:4])
    cross(draw, origin, INK, radius=7, width=2)
    # World event coordinates are from different substeps; keep that distinction.
    for event in exterior:
        x, y = screen(event["point_world"])
        draw.rectangle((x-3, y-3, x+3, y+3), outline=ORANGE, width=2)
    for event in exterior[::max(1, len(exterior)//3)]:
        arrow(draw, screen(event["point_world"]),
              screen(event["point_world"] + event["normal_into_block"] * 15), ORANGE, width=2, tip=7)
    arrow(draw, screen([18, 18]), screen([68, 18]), INK, width=2)
    arrow(draw, screen([18, 18]), screen([18, 68]), INK, width=2)
    draw.text(tuple(screen([72, 12])), "+x", font=FONTS[18], fill=INK)
    draw.text(tuple(screen([8, 72])), "+y", font=FONTS[18], fill=INK)
    # Local coordinates isolate point numbering from orientation and translation.
    local_offset, local_scale = np.array([1410, 267]), 4.45

    def local_screen(q):
        return local_offset + np.asarray(q) * local_scale

    local_outline = local_screen(OUTLINE)
    draw.polygon(list(map(tuple, local_outline)), fill="#e7edf3", outline="#61798f", width=3)
    for point_id, (q, score, normal) in enumerate(zip(local, scores, outward_normals(local))):
        xy = local_screen(q)
        x, y = xy
        draw.ellipse((x-6, y-6, x+6, y+6), fill=heat(float(score)), outline="white")
        label_position = xy + normal * 21
        # Concave corners have no unique exterior normal. Put their labels in
        # the bar interior so neither number collides with a neighboring label.
        if np.allclose(q, [15, 30]):
            label_position = xy + [-22, -23]
        elif np.allclose(q, [-15, 30]):
            label_position = xy + [22, -23]
        draw.text(tuple(label_position), str(point_id), font=FONTS[18], fill=INK, anchor="mm")
    for event in exterior:
        cross(draw, local_screen(event["point_local"]), ORANGE, radius=5, width=2)
    cross(draw, local_screen([0, 0]), INK, radius=7, width=2)
    draw.text((980, 828), "黑十字：物体位姿参考原点（不是质心）", font=FONTS[22], fill=INK)
    draw.text((980, 868), "橙十字：子步接触的局部材料位置", font=FONTS[22], fill=ORANGE)
    for k in range(400):
        draw.line((992+k, 924, 992+k, 949), fill=heat(k/399))
    draw.text((980, 962), "0：无接触标签", font=FONTS[20], fill=INK)
    draw.text((1285, 962), "1：接触邻域中心", font=FONTS[20], fill=INK)
    draw.text((1450, 921), f"最大软标签 = {float(scores.max()):.3f}", font=FONTS[22], fill=INK)
    draw.text((1450, 961), f"非零标签数 = {int(np.count_nonzero(scores))}", font=FONTS[22], fill=INK)
    # Legend occupies its own area, with no overlap with the scene.
    draw.text((42, 1056), "蓝圆：推杆实际位置     紫叉：本步动作目标     绿箭头：未来真实点位移（未放大）", font=FONTS[24], fill=INK)
    draw.text((42, 1098), "橙方块 / 箭头：各物理子步的接触世界位置 / 指向物体的法向；不是同一瞬间。", font=FONTS[22], fill=INK)
    draw.text((42, 1138), "右图只按局部材料位置汇总接触；标签 t 汇总执行动作 t 的 10 个物理子步。", font=FONTS[22], fill=INK)
    draw.text((42, 1178), "注意：这里使用离线示范的未来 flow；闭环测试必须改为由目标与规划器生成。", font=FONTS[24], fill="#9b3b28")
    gap = float(np.linalg.norm(data["actions"][index] - state[:2]))
    draw.text((42, 1223), f"动作目标与实际推杆中心距离：{gap:.2f} 仿真长度单位；未来轮廓点平均位移：{displacement.mean():.2f}。", font=FONTS[20], fill=INK)
    return image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=8)
    args = parser.parse_args()
    if args.horizon < 1:
        raise ValueError("Horizon must be positive.")
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output_dir}")
    with np.load(args.input, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    horizon = args.horizon
    length, n = data["contact_soft_labels"].shape
    if data["states"].shape != (length+1, 5):
        raise ValueError("Expected L+1 states for L action intervals.")
    if data["points_world"].shape != (length+1, n, 2):
        raise ValueError("Unexpected world point dimensions.")
    if data["actions"].shape != (length, 2):
        raise ValueError("Unexpected action dimensions.")
    for key in ["states", "points_world", "local_boundary_points", "actions", "contact_soft_labels"]:
        if not np.isfinite(data[key]).all():
            raise ValueError(f"Nonfinite values in {key}")
    projected = np.stack([transform(data["local_boundary_points"], state) for state in data["states"]])
    projection_error = float(np.abs(projected - data["points_world"]).max())
    if projection_error >= 1e-4:
        raise ValueError(f"Rigid point projection inconsistent: {projection_error}")
    counts = np.bincount(data["contact_events"]["action_index"], minlength=length)
    if len(counts) != length:
        raise ValueError("Contact action index out of range")
    no_contact = counts == 0
    if not np.all(data["contact_soft_labels"][no_contact] == 0):
        raise ValueError("A no-contact interval has nonzero labels.")
    valid = data["contact_label_valid"] & data["transition_has_video_endpoint"]
    starts = np.array([s for s in range(length-horizon+1) if valid[s:s+horizon].all()], dtype=int)
    moving = np.array([np.linalg.norm(data["points_world"][s+horizon]-data["points_world"][s], axis=1).mean()
                       for s in starts])
    contact_candidates = starts[(counts[starts] > 0) & (moving > 0.1)
                                & (data["contact_soft_labels"][starts].max(1) > 0)]
    free_candidates = np.array([s for s in starts if no_contact[s:s+horizon].all()], dtype=int)
    if not len(contact_candidates) or not len(free_candidates):
        raise ValueError("Need one moving contact window and one entirely no-contact window.")
    contact_start = int(contact_candidates[0])
    free_start = int(free_candidates[0])
    chosen = [contact_start, free_start]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name, start in zip(["contact_sample", "no_contact_sample"], chosen):
        draw_snapshot(data, start, horizon).save(args.output_dir / f"{name}.png")
    # Include approach-to-contact change, and preserve 10 Hz playback time.
    gif_indices = [s for s in range(max(0, contact_start-8), min(length-horizon, contact_start+24))
                   if valid[s:s+horizon].all()]
    frames = [draw_snapshot(data, s, horizon).resize((1440, 953), Image.Resampling.LANCZOS)
              .convert("P", palette=Image.Palette.ADAPTIVE, colors=192) for s in gif_indices]
    frames[0].save(args.output_dir / "contact_flow_preview.gif", save_all=True,
                   append_images=frames[1:], duration=100, loop=0, disposal=2)
    np.savez_compressed(
        args.output_dir / "samples.npz",
        episode_index=data["episode_index"], sample_start_index=np.array(chosen, dtype=np.int32),
        local_boundary_points=data["local_boundary_points"],
        anchor_states=data["states"][chosen],
        states=np.stack([data["states"][s:s+horizon+1] for s in chosen]),
        points_world=np.stack([data["points_world"][s:s+horizon+1] for s in chosen]),
        flow_from_anchor=np.stack([data["points_world"][s+1:s+horizon+1]-data["points_world"][s]
                                   for s in chosen]),
        actions=np.stack([data["actions"][s:s+horizon] for s in chosen]),
        contact_soft_labels=np.stack([data["contact_soft_labels"][s:s+horizon] for s in chosen]),
        contact_label_valid=np.stack([data["contact_label_valid"][s:s+horizon] for s in chosen]),
        transition_has_video_endpoint=np.stack([data["transition_has_video_endpoint"][s:s+horizon] for s in chosen]),
        contact_event_count=np.stack([counts[s:s+horizon] for s in chosen]),
    )
    validation = {
        "status": "passed", "input": str(args.input.resolve()),
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "episode_index": int(data["episode_index"]), "point_count": n,
        "horizon": horizon, "sample_start_indices": chosen,
        "sample_kinds": ["moving_contact", "all_no_contact"],
        "gif_frame_indices": gif_indices, "gif_duration_ms_per_frame": 100,
        "rigid_projection_max_abs_error": projection_error,
        "rigid_projection_tolerance": 1e-4,
        "no_contact_intervals_checked": int(no_contact.sum()),
        "all_no_contact_labels_exactly_zero": True,
        "selected_windows_all_label_valid": True,
        "selected_windows_all_endpoints_video_verified": True,
        "selected_windows_same_episode": True,
        "new_dataset_split_created": False,
        "contact_sample_anchor_event_count": int(counts[contact_start]),
        "contact_sample_mean_point_displacement": float(np.linalg.norm(
            data["points_world"][contact_start+horizon]-data["points_world"][contact_start], axis=1).mean()),
        "actual_pusher_action_target_distance": [float(np.linalg.norm(data["states"][s,:2]-data["actions"][s])) for s in chosen],
        "limitations": ["Contact labels are reconstructed simulator events, not optimal contact or success probability.",
                        "Displayed future flow is offline demonstration data and must not be used as a closed-loop test planner.",
                        "World contact events span different physics substeps. Local contact positions identify material regions.",
                        "Window samples are inspection artifacts; split full episodes before generating training windows."],
    }
    (args.output_dir / "visualization_validation.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
    schema = f"""# 接触与离线 flow 检查样本

已从输入 NPZ 实际生成图像、GIF 和 `samples.npz`。这是回放检查数据，未训练任何模型。

## 查看顺序

- `contact_sample.png`：有真实仿真接触且未来物体发生运动的区间，起点 {contact_start}。
- `no_contact_sample.png`：整个 {horizon} 步窗口都没有接触，起点 {free_start}；所有标签严格为 0。
- `contact_flow_preview.gif`：实际 10 Hz 时间，每幅图中的绿箭头表示未来 {horizon} 步的示范位移。
- `visualization_validation.json`：输入与脚本 SHA256、自动选择的索引、几何投影误差及验证结果。

## 图像含义

左图是世界坐标：蓝圆为半径 15 的推杆，蓝十字为实际中心；紫叉是动作的目标坐标。
动作目标通常与当前推杆中心不同，底层 PD 控制器逐子步追踪目标。
绿箭头为物体固定材料点从当前帧到未来帧的实际位移，没有放大。
橙方块和法向来自执行动作 t 的多个物理子步，不代表同一瞬间。
右图固定在物体局部坐标下展示全部点编号和软标签。橙十字标出子步接触的局部材料位置。

图像坐标 x 向右、y 向下；局部坐标的长度与世界坐标一致，世界范围为 512×512 仿真长度单位，未标定为米。
角度为弧度。变换 p_world = R(theta) p_local + t；t 是物体 body.position，局部原点在横条顶边中心，
不是质心；质心在局部 (0, 45)。正角度在 y 向下的图像上看起来为顺时针旋转。

## samples.npz 字段（K=2，H={horizon}，N={n}）

| 字段 | 形状 | 含义 |
|---|---|---|
| episode_index | 标量 | 来源完整示范编号 |
| sample_start_index | K | 原示范中的起点索引；顺序为有接触、无接触 |
| local_boundary_points | N×2 | 固定编号的局部轮廓点 |
| anchor_states | K×5 | 起点的推杆 x/y、物体参考点 x/y、角度 |
| states | K×(H+1)×5 | 每个动作前的状态及最后一个动作后的状态 |
| points_world | K×(H+1)×N×2 | 与 states 对齐的世界轮廓点 |
| flow_from_anchor | K×H×N×2 | 未来每帧的点坐标减去起点坐标，不是逐帧速度 |
| actions | K×H×2 | 每 0.1 秒更新的推杆目标坐标 |
| contact_soft_labels | K×H×N | 每个动作的 10 个物理子步接触汇总 |
| contact_label_valid | K×H | 所有值为真，已排除有内部拼接缝接触的区间 |
| transition_has_video_endpoint | K×H | 所有值为真，已排除无视频终点的末动作 |
| contact_event_count | K×H | 该区间原始接触点事件数量 |

点分数表示到该步接触材料位置的邻域接近程度，不能解释为最优性或成功概率。
本工具不新增训练/验证/测试划分；这些窗口只用于人工检查。正式训练必须先按完整示范划分，
再分别创建窗口，避免同一示范的重叠窗口泄漏。
图中的未来 flow 只可用于训练和离线分析；真实闭环测试必须由当前状态、目标和规划器构造期望 flow。
"""
    (args.output_dir / "SAMPLE_FIELDS.md").write_text(schema, encoding="utf-8")
    print(json.dumps(validation, ensure_ascii=True))


if __name__ == "__main__":
    main()
