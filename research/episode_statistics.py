"""Episode-level uncertainty summaries for matched PushT evaluations.

This module deliberately rejects missing or duplicate episodes. Overlapping
training windows never become independent evaluation replicates here.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total <= 0 or not 0 <= successes <= total:
        raise ValueError("Expected total>0 and 0<=successes<=total")
    p = successes / total
    denominator = 1 + z*z/total
    centre = (p + z*z/(2*total)) / denominator
    radius = z * math.sqrt(p*(1-p)/total + z*z/(4*total*total)) / denominator
    return [max(0.0, centre-radius), min(1.0, centre+radius)]


def validate_records(records: list[dict]) -> dict[int, dict]:
    if not records:
        raise ValueError("No episode records")
    result = {}
    for item in records:
        ep = int(item["episode_index"])
        if ep in result:
            raise ValueError(f"Duplicate episode {ep}; do not pool multiple training seeds as independent episodes")
        if not isinstance(item["success"], (bool, np.bool_)):
            raise ValueError(f"Episode {ep}: success must be boolean")
        for key in ["final_coverage", "max_coverage"]:
            value = float(item[key])
            if not np.isfinite(value) or not 0 <= value <= 1.000001:
                raise ValueError(f"Episode {ep}: invalid {key}")
        if float(item["final_coverage"]) > float(item["max_coverage"]) + 1e-8:
            raise ValueError(f"Episode {ep}: final coverage exceeds recorded maximum")
        expected = float(item["max_coverage"]) >= 0.95
        if bool(item["success"]) != expected:
            raise ValueError(f"Episode {ep}: success inconsistent with >=0.95 research criterion")
        if int(item["steps"]) < 0:
            raise ValueError(f"Episode {ep}: invalid step count")
        result[ep] = item
    return result


def summarize(records: list[dict]) -> dict:
    by_id = validate_records(records)
    ordered = [by_id[ep] for ep in sorted(by_id)]
    successes = sum(bool(x["success"]) for x in ordered)
    return {
        "episodes": len(ordered), "episode_ids": sorted(by_id),
        "successes": successes, "success_rate": successes/len(ordered),
        "success_rate_wilson_95": wilson_interval(successes, len(ordered)),
        "mean_final_coverage": float(np.mean([x["final_coverage"] for x in ordered])),
        "median_final_coverage": float(np.median([x["final_coverage"] for x in ordered])),
        "mean_max_coverage": float(np.mean([x["max_coverage"] for x in ordered])),
        "mean_steps": float(np.mean([x["steps"] for x in ordered])),
        "statistical_unit": "episode; single trained checkpoint unless explicitly stated otherwise",
        "success_criterion": "max_coverage >= 0.95",
    }


def paired_comparison(first: list[dict], second: list[dict], *,
                      bootstrap_samples: int = 10000, seed: int = 20261022) -> dict:
    a, b = validate_records(first), validate_records(second)
    if set(a) != set(b):
        raise ValueError("Paired comparison requires identical episode IDs; missing runs must not be silently dropped")
    if bootstrap_samples < 100:
        raise ValueError("Use at least 100 bootstrap samples")
    ids = sorted(a)
    differences = np.array([float(a[ep]["final_coverage"])-float(b[ep]["final_coverage"])
                            for ep in ids])
    rng = np.random.Generator(np.random.PCG64(seed))
    sampled = differences[rng.integers(0, len(ids), size=(bootstrap_samples, len(ids)))].mean(axis=1)
    return {
        "difference_direction": "first minus second", "episode_ids": ids,
        "mean_final_coverage_difference": float(differences.mean()),
        "paired_episode_bootstrap_percentile_95": np.quantile(sampled, [0.025, 0.975]).tolist(),
        "first_only_success": sum(bool(a[ep]["success"]) and not bool(b[ep]["success"]) for ep in ids),
        "second_only_success": sum(bool(b[ep]["success"]) and not bool(a[ep]["success"]) for ep in ids),
        "bootstrap_seed": seed, "bootstrap_samples": bootstrap_samples,
        "warning": "Exploratory interval on a small episode set; it does not include training-seed uncertainty.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--records-key", default="episodes",
                        help="JSON list key; ignored when the whole document is a list")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Preserving existing result: {args.output}")

    def read(path):
        doc = json.loads(path.read_text(encoding="utf-8"))
        return doc if isinstance(doc, list) else doc[args.records_key]

    records = read(args.input)
    result = {"input": str(args.input.resolve()), "summary": summarize(records)}
    if args.compare:
        other = read(args.compare)
        result["comparison_input"] = str(args.compare.resolve())
        result["comparison_summary"] = summarize(other)
        result["paired_comparison"] = paired_comparison(records, other)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
