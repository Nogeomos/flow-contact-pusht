"""Native Windows, lightweight visual Diffusion Policy baseline.

The denoising U-Net is imported unmodified from the pinned official repository.
This is a deliberately smaller, newly trained baseline, not a reproduction of
the paper's full ResNet/large U-Net/3050-epoch experiment.  No object pose, flow,
future image or test demonstration is available to the online policy.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "external/diffusion_policy"))
RAW = ROOT / "data/raw/lerobot_pusht"
DEFAULT_CACHE = ROOT / "data/derived/visual_dp_cache"
DEFAULT_OUTPUT = ROOT / "runs/stages234_20261022/visual_dp"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def prepare_cache(cache=DEFAULT_CACHE):
    """Decode the single aligned RGB stream once, retaining uint8 on disk."""
    import av
    import pyarrow.parquet as pq

    cache = Path(cache)
    parquet = RAW / "data/chunk-000/file-000.parquet"
    video = RAW / "videos/observation.image/chunk-000/file-000.mp4"
    expected = {"parquet_sha256": sha256(parquet), "video_sha256": sha256(video)}
    complete = cache / "cache_manifest.json"
    if complete.exists():
        meta = json.loads(complete.read_text(encoding="utf-8"))
        if any(meta[k] != v for k, v in expected.items()):
            raise ValueError("Visual cache source hashes no longer match")
        for name, digest in meta["cache_sha256"].items():
            if sha256(cache / name) != digest:
                raise ValueError(f"Visual cache changed: {name}")
        return meta
    cache.mkdir(parents=True, exist_ok=True)
    if any(cache.iterdir()):
        raise FileExistsError("Incomplete/nonempty cache; choose a new --cache directory")
    table = pq.read_table(parquet)
    episodes = table["episode_index"].to_numpy().astype(np.int64)
    frames = table["frame_index"].to_numpy().astype(np.int64)
    global_index = table["index"].to_numpy().astype(np.int64)
    timestamps = table["timestamp"].to_numpy().astype(np.float64)
    if not np.array_equal(global_index, np.arange(len(table))):
        raise AssertionError("Video decode requires contiguous global frame index")
    for ep in np.unique(episodes):
        indexes = np.flatnonzero(episodes == ep)
        if not np.array_equal(frames[indexes], np.arange(len(indexes))):
            raise AssertionError("Noncontiguous episode frames")
        if not np.allclose(timestamps[indexes], frames[indexes] / 10, atol=2e-5):
            raise AssertionError("Unexpected observation time alignment")
    shape = (len(table), 96, 96, 3)
    images = np.lib.format.open_memmap(cache / "images.npy", mode="w+", dtype="uint8", shape=shape)
    count = 0
    with av.open(str(video)) as container:
        for count, frame in enumerate(container.decode(video=0), start=1):
            if count > len(table):
                raise AssertionError("More video frames than parquet rows")
            rgb = frame.to_ndarray(format="rgb24")
            if rgb.shape != shape[1:]:
                raise AssertionError(f"Unexpected image shape {rgb.shape}")
            if abs(float(frame.time) - (count - 1) / 10) > 1e-4:
                raise AssertionError("Unexpected video timestamp")
            images[count - 1] = rgb
    if count != len(table):
        raise AssertionError(f"Decoded {count} frames for {len(table)} rows")
    images.flush()
    del images
    np.savez(cache / "metadata.npz", episodes=episodes, frames=frames,
             agent_positions=np.asarray(table["observation.state"].to_pylist(), dtype=np.float32),
             actions=np.asarray(table["action"].to_pylist(), dtype=np.float32))
    meta = {"schema_version": 1, **expected, "image_shape": list(shape), "frames": count,
            "episode_count": int(len(np.unique(episodes))),
            "alignment": "row t/image t is before absolute pusher-target action t; 10 Hz",
            "cache_sha256": {name: sha256(cache / name) for name in ["images.npy", "metadata.npz"]}}
    write_json(complete, meta)
    return meta


# Cache preparation remains usable before torch has been installed.
def torch_components():
    import torch
    from torch import nn
    from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
    from diffusers.schedulers.scheduling_ddim import DDIMScheduler
    return torch, nn, ConditionalUnet1D, DDPMScheduler, DDIMScheduler


@dataclass
class Config:
    horizon: int = 16
    observation_steps: int = 2
    action_steps: int = 8
    diffusion_steps: int = 100
    inference_steps: int = 20
    encoder_features: int = 128
    diffusion_embedding: int = 64
    down_dims: tuple = (64, 128, 256)
    seed: int = 20261022


def build_model(config):
    torch, nn, ConditionalUnet1D, _, _ = torch_components()

    class VisualDiffusionModel(nn.Module):
        def __init__(self):
            super().__init__()
            blocks = []
            channels = [3, 32, 64, 96, 128]
            for incoming, outgoing in zip(channels[:-1], channels[1:]):
                blocks += [nn.Conv2d(incoming, outgoing, 3, stride=2, padding=1),
                           nn.GroupNorm(8, outgoing), nn.SiLU()]
            # Flatten preserves spatial information; no global average pooling.
            self.encoder = nn.Sequential(*blocks, nn.Flatten(),
                                         nn.Linear(128 * 6 * 6, config.encoder_features), nn.SiLU())
            self.denoiser = ConditionalUnet1D(
                input_dim=2, global_cond_dim=config.observation_steps * (config.encoder_features + 2),
                diffusion_step_embed_dim=config.diffusion_embedding,
                down_dims=list(config.down_dims), kernel_size=5, n_groups=8,
                cond_predict_scale=True)

        def condition(self, images, agents):
            batch = len(images)
            encoded = self.encoder(images.reshape(-1, 3, 96, 96).float() / 127.5 - 1)
            features = encoded.reshape(batch, config.observation_steps, -1)
            return torch.cat([features, agents / 256 - 1], dim=-1).reshape(batch, -1)

        def forward(self, noisy_actions, timesteps, images, agents):
            return self.denoiser(noisy_actions, timesteps, global_cond=self.condition(images, agents))

    return VisualDiffusionModel()


class ImageWindows:
    """Two observed frames, sixteen actions; anchors never cross episodes.

    At current time t, input observations are [t-1,t] (repeat t=0 at reset).
    The model predicts action[t-1:t+15]; online execution selects [1:9].
    Padding at episode boundaries is excluded from the denoising loss.
    """
    def __init__(self, cache, split, config, max_samples=None):
        self.cache, self.split, self.config = Path(cache), split, config
        self.images = np.load(self.cache / "images.npy", mmap_mode="r")
        with np.load(self.cache / "metadata.npz") as d:
            self.episodes = d["episodes"]
            self.frames = d["frames"]
            self.agents = d["agent_positions"]
            self.actions = d["actions"]
        split_doc = json.loads((ROOT / "data/derived/splits.json").read_text(encoding="utf-8"))
        self.episode_ids = split_doc["splits"][split]
        self.bounds = {int(ep): (int(idx[0]), int(idx[-1]) + 1)
                       for ep in self.episode_ids for idx in [np.flatnonzero(self.episodes == ep)]}
        self.anchors = np.flatnonzero(np.isin(self.episodes, self.episode_ids))
        if max_samples and len(self.anchors) > max_samples:
            self.anchors = self.anchors[np.linspace(0, len(self.anchors)-1, max_samples, dtype=int)]
        assert set(self.episodes[self.anchors]) <= set(self.episode_ids)

    def __len__(self):
        return len(self.anchors)

    def __getitem__(self, index):
        anchor = int(self.anchors[index])
        ep = int(self.episodes[anchor])
        start, end = self.bounds[ep]
        observation_indexes = np.maximum(start, [anchor - 1, anchor])
        action_indexes = np.arange(anchor - 1, anchor - 1 + self.config.horizon)
        valid = (action_indexes >= start) & (action_indexes < end)
        action_indexes = np.clip(action_indexes, start, end - 1)
        return {"images": self.images[observation_indexes].transpose(0, 3, 1, 2).copy(),
                "agents": self.agents[observation_indexes].copy(),
                "actions": self.actions[action_indexes].copy(), "valid": valid,
                "episode": ep, "anchor": anchor, "frame": int(self.frames[anchor])}


def scheduler_for(config, inference=False):
    *_, DDPMScheduler, DDIMScheduler = torch_components()
    cls = DDIMScheduler if inference else DDPMScheduler
    return cls(num_train_timesteps=config.diffusion_steps, beta_schedule="squaredcos_cap_v2",
               clip_sample=True, prediction_type="epsilon")


def seed_all(seed):
    torch, *_ = torch_components()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)


def batch_loss(model, batch, scheduler, device, generator=None):
    torch, *_ = torch_components()
    actions = batch["actions"].to(device) / 256 - 1
    images, agents = batch["images"].to(device), batch["agents"].to(device)
    noise = torch.randn(actions.shape, device=device, generator=generator)
    times = torch.randint(0, scheduler.config.num_train_timesteps, (len(actions),),
                          device=device, generator=generator)
    noisy = scheduler.add_noise(actions, noise, times)
    prediction = model(noisy, times, images, agents)
    mask = batch["valid"].to(device).unsqueeze(-1)
    return ((prediction - noise).square() * mask).sum() / (mask.sum() * 2)


def versions():
    names = ["torch", "numpy", "diffusers", "einops", "av", "pyarrow"]
    return {name: importlib.metadata.version(name) for name in names}


def train(args):
    torch, *_ = torch_components()
    prepare_cache(args.cache)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "training_manifest.json").exists():
        raise FileExistsError("Training output exists; use a new directory")
    config = Config(seed=args.seed, inference_steps=args.inference_steps)
    seed_all(config.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available is false")
    dataset = ImageWindows(args.cache, "train", config, args.max_train_samples)
    valset = ImageWindows(args.cache, "validation", config, args.max_val_samples)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                                        num_workers=0, pin_memory=device.type == "cuda")
    val_loader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=False,
                                            num_workers=0, pin_memory=device.type == "cuda")
    model = build_model(config).to(device)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.95, 0.999), weight_decay=1e-6)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr / 10)
    noise_scheduler = scheduler_for(config)
    amp = device.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    manifest = {
        "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": asdict(config), "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "versions": versions(), "parameters": sum(p.numel() for p in model.parameters()),
        "train_samples": len(dataset), "validation_samples": len(valset),
        "train_episode_ids": dataset.episode_ids, "validation_episode_ids": valset.episode_ids,
        "split_sha256": sha256(ROOT / "data/derived/splits.json"),
        "implementation_sha256": sha256(__file__),
        "upstream_unet_sha256": sha256(ROOT / "external/diffusion_policy/diffusion_policy/model/diffusion/conditional_unet1d.py"),
        "upstream_commit": "5ba07ac6661db573af695b419a7947ecb704690f",
        "image_cache_manifest_sha256": sha256(Path(args.cache) / "cache_manifest.json"),
        "normalization": "Fixed known world bounds xy/256-1; RGB/127.5-1. No test statistics.",
        "online_inputs": ["two RGB frames", "two actual pusher positions"],
        "selection": "lowest fixed-noise validation epsilon-MSE of EMA; test not read during training",
        "adaptations": ["CNN trained from scratch instead of ResNet", "U-Net widths 64/128/256 instead of 512/1024/2048",
                         f"DDIM {config.inference_steps} inference steps instead of DDPM 100", "shared fixed 164/21/21 episode split",
                         "shorter training budget", "masked episode-boundary action padding", "fixed 0..512 normalization"]}
    write_json(output / "training_manifest.json", manifest)
    np.savez(output / "window_indices.npz", train=dataset.anchors, validation=valset.anchors)
    print(json.dumps({"event": "train_start", "parameters": manifest["parameters"],
                      "train_samples": len(dataset), "validation_samples": len(valset),
                      "device": str(device), "epochs": args.epochs}), flush=True)
    history, best, step = [], float("inf"), 0
    start_time = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        total, n = 0.0, 0
        for bi, batch in enumerate(loader):
            if args.limit_steps and bi >= args.limit_steps:
                break
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                loss = batch_loss(model, batch, noise_scheduler, device)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite denoising loss")
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            step += 1
            decay = min(0.995, (1 + step) / (10 + step))
            with torch.no_grad():
                for ep, p in zip(ema.parameters(), model.parameters()):
                    ep.lerp_(p, 1 - decay)
            total += float(loss.detach()) * len(batch["actions"])
            n += len(batch["actions"])
        val_total, val_n = 0.0, 0
        val_generator = torch.Generator(device=device).manual_seed(config.seed + 999)
        with torch.inference_mode():
            for batch in val_loader:
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                    value = batch_loss(ema, batch, noise_scheduler, device, val_generator)
                val_total += float(value) * len(batch["actions"])
                val_n += len(batch["actions"])
        row = {"epoch": epoch, "optimizer_steps": step, "train_noise_mse": total / n,
               "validation_noise_mse": val_total / val_n, "lr": opt.param_groups[0]["lr"],
               "seconds": time.perf_counter() - epoch_start}
        history.append(row)
        improved = row["validation_noise_mse"] < best
        if improved:
            best = row["validation_noise_mse"]
            torch.save({"state_dict": ema.state_dict(), "config": asdict(config), "epoch": epoch,
                        "validation_noise_mse": best, "manifest": manifest, "optimizer_steps": step,
                        "train_anchors": dataset.anchors.tolist()}, output / "best.pt")
        write_json(output / "history.json", history)
        print(json.dumps({"event": "epoch", **row, "new_best": improved}), flush=True)
        lr_scheduler.step()
    torch.save({"state_dict": ema.state_dict(), "config": asdict(config), "epoch": args.epochs,
                "validation_noise_mse": history[-1]["validation_noise_mse"], "manifest": manifest,
                "optimizer_steps": step}, output / "last.pt")
    result = {"epochs": args.epochs, "optimizer_steps": step, "best_validation_noise_mse": best,
              "best_epoch": int(min(history, key=lambda r: r["validation_noise_mse"])["epoch"]),
              "wall_seconds": time.perf_counter() - start_time,
              "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
              "best_checkpoint_sha256": sha256(output / "best.pt"), "test_evaluated": False}
    write_json(output / "training_result.json", result)
    print(json.dumps({"event": "train_complete", **result}), flush=True)


class LoadedPolicy:
    def __init__(self, checkpoint, device="cpu", seed=0):
        torch, *_ = torch_components()
        self.device = torch.device(device)
        saved = torch.load(checkpoint, map_location=self.device, weights_only=True)
        self.config = Config(**saved["config"])
        self.model = build_model(self.config).to(self.device).eval()
        self.model.load_state_dict(saved["state_dict"])
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self.scheduler = scheduler_for(self.config, inference=True)
        self.checkpoint = str(checkpoint)

    def predict_batch(self, images, agent_positions):
        torch, *_ = torch_components()
        # Batch API uses channel-first B,2,3,96,96 uint8.
        image_tensor = torch.as_tensor(images, device=self.device)
        agent_tensor = torch.as_tensor(agent_positions, dtype=torch.float32, device=self.device)
        if image_tensor.ndim != 5 or tuple(image_tensor.shape[1:]) != (2, 3, 96, 96):
            raise ValueError("Expected B,2,3,96,96 RGB uint8 images")
        if image_tensor.dtype != torch.uint8:
            raise ValueError("Images must be uint8 RGB, not pre-normalized")
        if tuple(agent_tensor.shape) != (len(images), 2, 2):
            raise ValueError("Expected B,2,2 actual pusher positions")
        with torch.inference_mode():
            cond = self.model.condition(image_tensor, agent_tensor)
            sample = torch.randn((len(images), self.config.horizon, 2), device=self.device,
                                 generator=self.generator)
            self.scheduler.set_timesteps(self.config.inference_steps, device=self.device)
            for timestep in self.scheduler.timesteps:
                noise = self.model.denoiser(sample, timestep, global_cond=cond)
                sample = self.scheduler.step(noise, timestep, sample, eta=0.0).prev_sample
            actions = ((sample[:, 1:1+self.config.action_steps] + 1) * 256).clamp(0, 512)
        return actions.cpu().numpy()

    def predict(self, images, agent_positions):
        images = np.asarray(images)
        agents = np.asarray(agent_positions, dtype=np.float32)
        if images.shape != (2, 96, 96, 3):
            raise ValueError("Expected two RGB frames with shape (2,96,96,3)")
        return self.predict_batch(images.transpose(0, 3, 1, 2)[None], agents[None])[0]


def load_policy(checkpoint, device="cpu", seed=0):
    return LoadedPolicy(checkpoint, device=device, seed=seed)


def evaluate(args):
    torch, *_ = torch_components()
    seed_all(args.seed)
    prepare_cache(args.cache)
    policy = load_policy(args.checkpoint, args.device, args.seed)
    dataset = ImageWindows(args.cache, args.split, policy.config, args.max_samples)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    rows = []
    start = time.perf_counter()
    for batch in loader:
        pred = policy.predict_batch(batch["images"], batch["agents"])
        target = batch["actions"].numpy()[:, 1:9]
        valid = batch["valid"].numpy()[:, 1:9]
        dists = np.linalg.norm(pred - target, axis=-1)
        hold = np.linalg.norm(batch["agents"].numpy()[:, -1, None] - target, axis=-1)
        for i in range(len(pred)):
            rows.append({"episode": int(batch["episode"][i]), "frame": int(batch["frame"][i]),
                         "anchor": int(batch["anchor"][i]), "first_action_l2": float(dists[i, 0]),
                         "valid_horizon_action_l2": float(dists[i, valid[i]].mean()),
                         "hold_baseline_action_l2": float(hold[i, valid[i]].mean())})
    per_episode = [{"episode": ep, "windows": sum(r["episode"] == ep for r in rows),
                    "mean_first_action_l2": float(np.mean([r["first_action_l2"] for r in rows if r["episode"] == ep])),
                    "mean_horizon_action_l2": float(np.mean([r["valid_horizon_action_l2"] for r in rows if r["episode"] == ep])),
                    "mean_hold_baseline_l2": float(np.mean([r["hold_baseline_action_l2"] for r in rows if r["episode"] == ep]))}
                   for ep in dataset.episode_ids if any(r["episode"] == ep for r in rows)]
    values = np.asarray([r["mean_horizon_action_l2"] for r in per_episode])
    rng = np.random.default_rng(args.seed)
    boot = rng.choice(values, (2000, len(values)), replace=True).mean(axis=1)
    result = {"split": args.split, "checkpoint": str(args.checkpoint), "checkpoint_sha256": sha256(args.checkpoint),
              "windows": len(rows), "episodes": len(per_episode), "sampling_seed": args.seed,
              "mean_first_action_l2": float(np.mean([r["first_action_l2"] for r in rows])),
              "mean_horizon_action_l2": float(np.mean([r["valid_horizon_action_l2"] for r in rows])),
              "mean_hold_baseline_action_l2": float(np.mean([r["hold_baseline_action_l2"] for r in rows])),
              "equal_episode_mean_horizon_action_l2": float(values.mean()),
              "episode_bootstrap_95ci": np.quantile(boot, [0.025, 0.975]).tolist(),
              "wall_seconds": time.perf_counter() - start,
              "interpretation": "Offline action imitation error in world units, not closed-loop success. One DDIM sample per window.",
              "per_episode": per_episode, "per_window": rows}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    write_json(output, result)
    print(json.dumps({k: v for k, v in result.items() if k not in ["per_episode", "per_window"]}), flush=True)


def smoke(args):
    torch, *_ = torch_components()
    seed_all(100)
    config = Config()
    model = build_model(config).to(args.device)
    dataset = ImageWindows(args.cache, "train", config, 2)
    batch = next(iter(torch.utils.data.DataLoader(dataset, batch_size=2)))
    loss = batch_loss(model, batch, scheduler_for(config), torch.device(args.device))
    loss.backward()
    assert torch.isfinite(loss)
    assert model.encoder[0].weight.grad.abs().sum() > 0, "Visual encoder must receive training gradients"
    for split in ["train", "validation", "test"]:
        ds = ImageWindows(args.cache, split, config, 20)
        for index in range(len(ds)):
            item = ds[index]
            ep = int(item["episode"])
            assert ep in ds.episode_ids
            assert item["actions"].shape == (16, 2) and item["images"].shape == (2, 3, 96, 96)
    print(json.dumps({"status": "passed", "device": args.device, "loss": float(loss.detach()),
                      "parameters": sum(p.numel() for p in model.parameters()),
                      "visual_encoder_nonzero_gradient": True, "episode_window_boundaries_checked": True}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    tr = sub.add_parser("train")
    tr.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    tr.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    tr.add_argument("--device", default="cuda")
    tr.add_argument("--seed", type=int, default=20261022)
    tr.add_argument("--epochs", type=int, default=60)
    tr.add_argument("--batch-size", type=int, default=64)
    tr.add_argument("--lr", type=float, default=1e-4)
    tr.add_argument("--inference-steps", type=int, default=20)
    tr.add_argument("--max-train-samples", type=int, default=None)
    tr.add_argument("--max-val-samples", type=int, default=0, help="0 evaluates every validation anchor")
    tr.add_argument("--limit-steps", type=int, default=None)
    tr.add_argument("--no-amp", action="store_true")
    ev = sub.add_parser("evaluate")
    ev.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    ev.add_argument("--checkpoint", type=Path, required=True)
    ev.add_argument("--output", type=Path, required=True)
    ev.add_argument("--split", choices=["validation", "test"], default="test")
    ev.add_argument("--device", default="cuda")
    ev.add_argument("--seed", type=int, default=20261022)
    ev.add_argument("--max-samples", type=int, default=0, help="0 evaluates every anchor in the selected split")
    ev.add_argument("--batch-size", type=int, default=32)
    sm = sub.add_parser("smoke")
    sm.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    sm.add_argument("--device", default="cpu")
    args = p.parse_args()
    if args.command == "prepare":
        print(json.dumps(prepare_cache(args.cache), indent=2))
    elif args.command == "train":
        train(args)
    elif args.command == "evaluate":
        evaluate(args)
    elif args.command == "smoke":
        smoke(args)


if __name__ == "__main__":
    main()
