"""Independent small-sample checks of stage-2 coordinates and supervision.

Only nine windows are materialized per configuration, avoiding a second full
dataset copy while model training is in progress. This is a verification run,
not a model experiment and not a source of hyperparameter selection.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
from shapely.geometry import Point, Polygon

import contact_learning as learning

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    output = ROOT / "runs/stages234_20261022/contact_feature_checks.json"
    if output.exists():
        raise FileExistsError(output)
    full = json.loads((ROOT / "data/derived/windows.json").read_text())
    small = dict(full)
    small["windows"] = {key: [rows[0], rows[len(rows)//2], rows[-1]]
                        for key, rows in full["windows"].items()}
    results = []
    polygon = Polygon(learning.OUTLINE)
    for n in [32, 64, 128]:
        local = learning.boundary_points(n)
        norm = learning.normals(n)
        np.testing.assert_allclose(np.linalg.norm(norm, axis=1), 1)
        # At vertices an outgoing normal can land on an adjoining edge, so use
        # covers rather than contains. Concave corners are a normal convention.
        assert all(polygon.covers(Point(p + 1e-5*v)) for p, v in zip(local, norm))
        assert np.allclose([np.linalg.norm(x) for x in norm], 1)
        for label in ["soft", "hard"]:
            with patch.object(learning.json, "loads", return_value=small):
                data = learning.load_dataset(n, label)
            assert data["geometry"].shape == (9, n, 13)
            errors = {"flow": 0.0, "geometry": 0.0, "action": 0.0, "target": 0.0, "distance": 0.0}
            for i, (ep, anchor) in enumerate(data["ids"]):
                with np.load(ROOT/f"data/derived/replayed_contacts/episode_{ep:06d}.npz",
                             allow_pickle=False) as raw:
                    st = raw["states"]
                    now = st[anchor]
                    angle = now[4]
                    c, s = np.cos(angle), np.sin(angle)
                    r = np.array([[c, -s], [s, c]])
                    points = []
                    for state in st[anchor:anchor+9]:
                        ct, ss = np.cos(state[4]), np.sin(state[4])
                        world = np.column_stack([ct*local[:,0]-ss*local[:,1],
                                                 ss*local[:,0]+ct*local[:,1]]) + state[2:4]
                        points.append(world)
                    displacement = np.stack(points[1:]) - points[0]
                    expected_flow = np.zeros((n, 8, 2))
                    for j in range(n):
                        for t in range(8):
                            expected_flow[j, t] = r.T @ displacement[t, j] / 30
                    expected_action = np.stack([r.T @ (a-now[:2])/100
                                                for a in raw["actions"][anchor:anchor+8]])
                    rel = (r.T @ (now[:2]-now[2:4]))[None] - local
                    global_context = np.r_[now[:4]/256-1, c, s]
                    expected_geometry = np.c_[local/100, norm, rel/100,
                                               np.linalg.norm(rel, axis=1)/100,
                                               np.tile(global_context, (n,1))]
                    ev = raw["contact_events"]
                    selected = ev[(ev["action_index"] >= anchor) &
                                  (ev["action_index"] < anchor+8) & ev["on_exterior"]]
                    if len(selected):
                        expected_dist = np.array([min(np.linalg.norm(p-q) for q in selected["point_local"])
                                                  for p in local])
                        expected_target = np.exp(-expected_dist**2/50) if label=="soft" else expected_dist<=7.5
                    else:
                        expected_dist, expected_target = np.full(n,512.), np.zeros(n)
                    expected = {"flow": expected_flow, "geometry": expected_geometry,
                                "action": expected_action, "distance": expected_dist,
                                "target": expected_target}
                    for key, value in expected.items():
                        errors[key] = max(errors[key], float(np.max(np.abs(data[key][i]-value))))
                        np.testing.assert_allclose(data[key][i], value, rtol=1e-6, atol=2e-5)
                    if n == 64 and label == "soft":
                        # Recomputed future union must equal the max of the eight
                        # original per-interval soft labels (same Gaussian sigma).
                        np.testing.assert_allclose(data["target"][i],
                                                   raw["contact_soft_labels"][anchor:anchor+8].max(0),
                                                   rtol=1e-6, atol=1e-7)
            results.append({"points": n, "label": label, "windows": len(data["ids"]),
                            "max_absolute_errors": errors, "status": "passed"})
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {"status": "passed", "materialized_windows_per_configuration": 9,
              "selected_windows": small["windows"], "configurations": results,
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "reviewed_contact_learning_sha256": hashlib.sha256((ROOT/"research/contact_learning.py").read_bytes()).hexdigest(),
              "scope": "Numerical feature/supervision consistency only; no trained performance claim."}
    output.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
