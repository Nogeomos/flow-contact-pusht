"""Compute review statistics from completed, immutable experiment outputs."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import numpy as np

from episode_statistics import summarize, paired_comparison
from verify_experiments import EXPECTED_MODELS

ROOT=Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def exact_paired_success(a,b):
    by_a={r["episode_index"]:r for r in a}
    by_b={r["episode_index"]:r for r in b}
    assert set(by_a)==set(by_b)
    a_only=[ep for ep in by_a if by_a[ep]["success"] and not by_b[ep]["success"]]
    b_only=[ep for ep in by_a if by_b[ep]["success"] and not by_a[ep]["success"]]
    n=len(a_only)+len(b_only)
    p=1.0 if n==0 else min(1.,2*sum(math.comb(n,k) for k in range(min(len(a_only),len(b_only))+1))/2**n)
    return {"first_only_success_episodes":a_only,"second_only_success_episodes":b_only,
            "discordant_episodes":n,"exact_two_sided_mcnemar_p":p,
            "interpretation":"Exploratory paired success test for these checkpoints and these episodes."}


def failure_observations(directory):
    doc=read(directory/"summary.json")
    rows=[]
    for rec in doc["results"]:
        if rec["success"]:
            continue
        ep=rec["episode_index"]
        detail=read(directory/f"episode_{ep:06d}.json")
        with np.load(directory/f"episode_{ep:06d}.npz",allow_pickle=False) as data:
            states=data["states"]
            displacement=np.linalg.norm(states[:,2:4]-states[0,2:4],axis=1)
            angle_motion=float(np.abs(np.unwrap(states[:,4])-states[0,4]).max())
            rows.append({"episode_index":ep,"steps":rec["steps"],"final_coverage":rec["final_coverage"],
                         "max_coverage":rec["max_coverage"],"drop_from_best":rec["max_coverage"]-rec["final_coverage"],
                         "final_angle_error_deg":rec["final_angle_error_deg"],
                         "contact_intervals":rec["contact_intervals"],
                         "max_body_origin_displacement":float(displacement.max()),
                         "max_body_angle_displacement_rad":angle_motion,
                         "no_contact_and_no_origin_motion":bool(rec["contact_intervals"]==0 and displacement.max()<1e-9),
                         "no_contact_and_no_rigid_body_motion":bool(rec["contact_intervals"]==0 and displacement.max()<1e-9 and angle_motion<1e-9),
                         "no_feasible_contact_plan_count":sum(p["kind"]=="no_feasible_contact" for p in detail.get("plans",[]))})
    return {"failures":len(rows),"rows":rows,
            "no_contact_and_no_origin_motion_count":sum(r["no_contact_and_no_origin_motion"] for r in rows),
            "interpretation":"Observed trajectory properties, not experimentally established failure causes. no_feasible_contact also includes insufficient remaining action budget."}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir",type=Path,default=ROOT/"runs/stages234_20261022")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--allow-missing-visual",action="store_true",
                        help="Interim analysis while the visual DP experiment is still running")
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    base=args.run_dir
    offline=[]
    for name in EXPECTED_MODELS:
        path=base/name/"test_metrics.json"
        report=read(path)
        row={"name":name,"task":report["config"]["task"],"seed":report["config"]["seed"],
             "points":report["config"]["points"],"metrics":report["metrics"],"test_metrics_sha256":sha(path)}
        if row["task"]=="contact":
            with np.load(base/name/"test_predictions.npz",allow_pickle=False) as d:
                distance=d["distance"]
                oracle=distance.min(1)
                positive=oracle<511
                chosen=distance[np.arange(len(distance)),d["prediction"].argmax(1)]
                row.update(oracle_min_distance=float(oracle[positive].mean()),
                           top1_distance_minus_oracle=float((chosen-oracle)[positive].mean()),
                           contact_windows=int(positive.sum()),no_contact_windows=int((~positive).sum()))
        offline.append(row)
    test_summaries={}
    # Explicit primary paths exclude fresh-process reproducibility reruns.
    for name in ["rule_test","contact_test","contact_seed1_test","contact_seed2_test","flow_action_test","visual_dp_test"]:
        path=base/name/"summary.json"
        if not path.exists() and name=="visual_dp_test" and args.allow_missing_visual:
            continue
        if not path.exists():
            raise FileNotFoundError(f"Final test missing: {name}")
        d=read(path)
        assert d["split"]=="test"
        test_summaries[name]=(path,d)
    closed=[]
    for name,(path,d) in sorted(test_summaries.items()):
        closed.append({"name":name,"method":d["method"],"summary_sha256":sha(path),
                       **summarize(d["results"]),"total_candidate_physics_substeps":d["total_candidate_physics_substeps"],
                       "wall_time_seconds":d["total_wall_time_seconds"]})
    a=test_summaries["contact_test"][1]["results"]
    b=test_summaries["rule_test"][1]["results"]
    pair={**paired_comparison(a,b),**exact_paired_success(a,b)}
    seed_names=["contact_test","contact_seed1_test","contact_seed2_test"]
    seed_results=[test_summaries[n][1] for n in seed_names]
    rates=np.array([d["success_rate"] for d in seed_results])
    coverages=np.array([d["mean_final_coverage"] for d in seed_results])
    seed_report={"seeds":[0,1,2],"success_rates":rates.tolist(),"mean_success_rate":float(rates.mean()),
                 "sample_sd_success_rate":float(rates.std(ddof=1)),"mean_final_coverages":coverages.tolist(),
                 "mean_final_coverage_across_seeds":float(coverages.mean()),
                 "warning":"The same 21 episode IDs are reused for all seeds. This is not a 63-independent-episode test set.",
                 "comparison_to_rule_success_rate":test_summaries["rule_test"][1]["success_rate"]}
    failures={name:failure_observations(path.parent) for name,(path,d) in test_summaries.items()}
    supplementary=[]
    diagnostic_path=base/"contact_no_flow_test/summary.json"
    if diagnostic_path.exists():
        d=read(diagnostic_path)
        protocol_path=base/"contact_no_flow_diagnostic_protocol.json"
        supplementary.append({"name":"contact_no_flow_test","summary":summarize(d["results"]),
                              "protocol":read(protocol_path),"protocol_sha256":sha(protocol_path),
                              "summary_sha256":sha(diagnostic_path),
                              "paired_primary_contact_minus_no_flow":paired_comparison(a,d["results"]),
                              "success_comparison":exact_paired_success(a,d["results"]),
                              "interpretation":"Supplementary post-hoc diagnostic, separate from predeclared primary results; same 21 initial conditions."})
        failures["contact_no_flow_test"]=failure_observations(diagnostic_path.parent)
    output={"status":"partial_measured" if args.allow_missing_visual else "measured","offline":offline,"closed_loop":closed,
            "paired_primary_contact_minus_rule":pair,"contact_training_seed_variation":seed_report,
            "failure_observations":failures,"supplementary_closed_loop_diagnostics":supplementary,
            "oracle_note":"Oracle is the closest sampled contour point to the actual future contact set, not an optimal controller or action oracle.",
            "script_sha256":sha(__file__)}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(output,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"status":output["status"],"offline_models":len(offline),"closed_loop_runs":len(closed),
                      "paired_primary":pair,"seed_variation":seed_report},indent=2),flush=True)


if __name__=="__main__":
    main()
