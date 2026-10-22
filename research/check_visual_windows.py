"""Small-memory independent checks of DP time alignment and boundary masking."""
from pathlib import Path
import hashlib
import json
import numpy as np
import av

import visual_dp as dp

ROOT = Path(__file__).resolve().parents[1]


def main():
    output = ROOT / "runs/stages234_20261022/visual_window_checks.json"
    if output.exists():
        raise FileExistsError(output)
    cache = ROOT / "data/derived/visual_dp_cache"
    with np.load(cache / "metadata.npz") as z:
        ids, frames, actions, positions = [z[k] for k in
                                         ["episodes", "frames", "actions", "agent_positions"]]
    images = np.load(cache / "images.npy", mmap_mode="r")
    check_count = 0
    boundary_masks = []
    for split in ["train", "validation", "test"]:
        dataset = dp.ImageWindows(cache, split, dp.Config())
        mapping = {int(a): i for i, a in enumerate(dataset.anchors)}
        for ep in dataset.episode_ids:
            indexes = np.flatnonzero(ids == ep)
            for ordinal in [0, len(indexes)//2, len(indexes)-1]:
                anchor = int(indexes[ordinal])
                item = dataset[mapping[anchor]]
                assert item["episode"] == ep and item["frame"] == ordinal
                assert item["images"].shape == (2,3,96,96)
                assert item["images"].dtype == np.uint8
                assert item["actions"].shape == (16,2)
                # Policy execution begins at generated action index 1. This must
                # be the command associated with the CURRENT observation row.
                np.testing.assert_array_equal(item["actions"][1], actions[anchor])
                np.testing.assert_array_equal(item["agents"][1], positions[anchor])
                np.testing.assert_array_equal(item["images"][1], images[anchor].transpose(2,0,1))
                previous = indexes[max(ordinal-1, 0)]
                np.testing.assert_array_equal(item["agents"][0], positions[previous])
                np.testing.assert_array_equal(item["images"][0], images[previous].transpose(2,0,1))
                for j in range(16):
                    relative = ordinal+j-1
                    valid = 0 <= relative < len(indexes)
                    assert bool(item["valid"][j]) == valid
                    expected_row = indexes[min(max(relative,0), len(indexes)-1)]
                    assert ids[expected_row] == ep
                    np.testing.assert_array_equal(item["actions"][j], actions[expected_row])
                if ordinal in [0, len(indexes)-1]:
                    boundary_masks.append({"episode": int(ep), "frame": ordinal,
                                           "valid_action_count": int(item["valid"].sum())})
                check_count += 1
    # The original AV1 decoder returns RGB, and the cache keeps exactly those
    # channel values; this catches an RGB/BGR conversion mismatch at its source.
    video = ROOT / "data/raw/lerobot_pusht/videos/observation.image/chunk-000/file-000.mp4"
    with av.open(str(video)) as container:
        for i, frame in enumerate(container.decode(video=0)):
            np.testing.assert_array_equal(images[i], frame.to_ndarray(format="rgb24"))
            if i == 1:
                break
    result = {"status": "passed", "episode_count": 206,
              "checked_windows": check_count, "first_last_boundary_masks": boundary_masks,
              "raw_rgb_frames_checked": 2,
              "current_observation_matches_executed_action_index": 1,
              "no_cross_episode_padding": True,
              "visual_dp_sha256": hashlib.sha256((ROOT/"research/visual_dp.py").read_bytes()).hexdigest(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "scope": "Data/window consistency; no model performance evaluated."}
    output.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k!="first_last_boundary_masks"}), flush=True)


if __name__ == "__main__":
    main()
