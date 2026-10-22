"""Describe demonstrated vs actually queried planner-flow distributions.

This diagnostic neither tunes models nor changes the held-out test protocol.
For online contact controllers it measures the flows saved at planning calls,
not the unused re-computation at every individual control step.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def hashfile(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metrics(flow):
    # Supports [B,8,N,2] cumulative WORLD displacements.
    endpoint=np.linalg.norm(flow[:,-1],axis=-1)
    linear=flow[:,-1,None]*np.arange(1,9)[None,:,None,None]/8
    residual=np.sqrt(np.mean(np.sum((flow-linear)**2,axis=-1),axis=(1,2)))
    return np.c_[endpoint.mean(1),np.sqrt(np.mean(endpoint**2,axis=1)),residual]


def summary(array):
    names=["endpoint_mean_point_displacement","endpoint_rms_point_displacement",
           "rms_deviation_from_linear_point_path"]
    return {"query_windows":len(array),
            "endpoint_rms_under_one_world_unit_fraction":float((array[:,1]<1).mean()),
            "metrics":{name:{"mean":float(array[:,i].mean()),"std":float(array[:,i].std()),
                              "min":float(array[:,i].min()),"max":float(array[:,i].max()),
                              "quantiles_p01_p10_p50_p90_p99":np.quantile(array[:,i],[.01,.1,.5,.9,.99]).tolist()}
                       for i,name in enumerate(names)}}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir",type=Path,default=ROOT/"runs/stages234_20261022")
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    windows_path=ROOT/"data/derived/windows.json"
    windows=json.loads(windows_path.read_text())["windows"]["train"]
    grouped={}
    for ep,t in windows:
        grouped.setdefault(ep,[]).append(t)
    measured=[]
    for ep,starts in grouped.items():
        with np.load(ROOT/f"data/derived/replayed_contacts/episode_{ep:06d}.npz",allow_pickle=False) as z:
            world=z["points_world"]
            starts=np.array(starts)
            flow=world[starts[:,None]+np.arange(1,9)[None]]-world[starts,None]
            measured.append(metrics(flow))
    demonstrated=np.concatenate(measured)
    online=[]
    for path in sorted(args.run_dir.rglob("summary.json")):
        if path.parent.parent.resolve()!=args.run_dir.resolve():
            continue  # Fresh-process reproductions are not new online samples.
        doc=json.loads(path.read_text())
        if doc.get("split")!="test" or doc.get("method") not in ["rule","learned_contact"]:
            continue
        all_flows=[]
        for rec in doc["results"]:
            detail=json.loads((path.parent/f'episode_{rec["episode_index"]:06d}.json').read_text())
            all_flows.extend(plan["desired_flow_world"] for plan in detail["plans"] if plan["kind"]=="contact")
        if all_flows:
            values=metrics(np.asarray(all_flows))
            train_low,train_high=np.quantile(demonstrated[:,1],[.01,.99])
            online.append({"method":doc["method"],"checkpoint":doc["checkpoint"],
                           "evaluation_kind":"supplementary_diagnostic" if path.parent.name=="contact_no_flow_test" else "primary",
                           "summary_path":str(path.resolve()),"summary_sha256":hashfile(path),
                           **summary(values),
                           "endpoint_rms_outside_train_p01_p99_fraction":float(((values[:,1]<train_low)|(values[:,1]>train_high)).mean())})
    result={"status":"measured","source":"training demonstrations vs saved held-out online planning calls",
            "train_windows_manifest_sha256":hashfile(windows_path),"training":summary(demonstrated),
            "online":online,"units":"512-world simulator length units; horizon 8 at 10 Hz",
            "interpretation":"Descriptive input-distribution diagnostic only; differences are not a causal explanation of failures and are not used for tuning.",
            "limitations":["Online plans are correlated within episodes, so pooled queries are not independent samples.",
                            "Only plans with contact candidates are included; no-feasible holds do not query a usable contact trajectory.",
                            "Low-dimensional magnitude summaries cannot detect all differences in flow shape or state distribution."],
            "script_sha256":hashfile(__file__)}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2),flush=True)


if __name__=="__main__":
    main()
