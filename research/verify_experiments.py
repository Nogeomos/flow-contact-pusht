"""Fresh-process final acceptance checks for stages 2--4.

Checks saved measured trajectories, independently recomputes goal coverage and
planner flow, verifies frozen splits/checkpoint hashes, and rejects incomplete
test suites by default. Use --partial only for a clearly labeled interim audit.
No GPU or torch import is needed and no experiment result is overwritten.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from shapely.geometry import Polygon

from episode_statistics import summarize, paired_comparison

ROOT = Path(__file__).resolve().parents[1]
OUTLINE = np.array([[-60,0],[60,0],[60,30],[15,30],[15,120],[-15,120],[-15,30],[-60,30]], float)
EXPECTED_MODELS = ["contact_main", "flow_action", "contact_seed1", "contact_seed2",
                   "contact_32", "contact_128", "contact_hard", "contact_endpoint",
                   "contact_gru", "contact_no_flow"]
EXPECTED_CONFIGS = {
    "contact_main": (64,"sequence","soft","contact",0),
    "flow_action": (64,"sequence","soft","action",0),
    "contact_seed1": (64,"sequence","soft","contact",1),
    "contact_seed2": (64,"sequence","soft","contact",2),
    "contact_32": (32,"sequence","soft","contact",0),
    "contact_128": (128,"sequence","soft","contact",0),
    "contact_hard": (64,"sequence","hard","contact",0),
    "contact_endpoint": (64,"endpoint","soft","contact",0),
    "contact_gru": (64,"gru","soft","contact",0),
    "contact_no_flow": (64,"none","soft","contact",0),
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def resolve(path):
    value = Path(path)
    return value if value.is_absolute() else ROOT/value


def need(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def world(local, pose):
    c, s = np.cos(pose[2]), np.sin(pose[2])
    return np.c_[c*local[:,0]-s*local[:,1], s*local[:,0]+c*local[:,1]] + pose[:2]


def independent_flow(state, goal):
    edges = np.roll(OUTLINE,-1,axis=0)-OUTLINE
    lengths = np.linalg.norm(edges,axis=1)
    edges_at = np.searchsorted(np.cumsum(lengths),np.arange(64)*lengths.sum()/64,side="right")
    fractions = (np.arange(64)*lengths.sum()/64-np.r_[0,np.cumsum(lengths)][edges_at])/lengths[edges_at]
    local = OUTLINE[edges_at]+edges[edges_at]*fractions[:,None]
    now = world(local,state[2:])
    cog = world(np.array([[0.,45.]]),state[2:])[0]
    target_cog = world(np.array([[0.,45.]]),goal)[0]
    movement = target_cog-cog
    norm = np.linalg.norm(movement)
    if norm > 24:
        movement = movement*(24/norm)
    angle = np.clip((goal[2]-state[4]+np.pi)%(2*np.pi)-np.pi,-.24,.24)
    output=[]
    for k in range(1,9):
        theta = state[4]+angle*k/8
        new_origin = cog+movement*k/8-np.array([-45*np.sin(theta),45*np.cos(theta)])
        output.append(world(local,np.r_[new_origin,theta])-now)
    return np.array(output)


def verify_closed_loop(path, test_ids, split_hash, frozen):
    doc=read(path)
    need(doc["split"]=="test",f"{path}: not final test")
    need(doc["split_sha256"]==split_hash,f"{path}: split hash differs")
    need(doc["episodes"]==test_ids,f"{path}: incomplete/reordered/extra test episodes")
    need(doc["max_steps_per_episode"]==300,f"{path}: mismatched action budget")
    for key, source in [("controller_sha256","controller.py"),("evaluator_sha256","evaluate_closed_loop.py")]:
        need(doc[key]==digest(ROOT/"research"/source),f"{path}: source changed since run ({key})")
        need(doc[key]==frozen[key],f"{path}: source differs from pre-test freeze ({key})")
    if doc["checkpoint"]:
        need(digest(resolve(doc["checkpoint"]))==doc["checkpoint_sha256"],f"{path}: checkpoint changed")
    records=doc["results"]
    need([r["episode_index"] for r in records]==test_ids,f"{path}: result episode mismatch")
    aggregate=summarize(records)
    for key in ["successes","success_rate","mean_final_coverage","mean_max_coverage","mean_steps"]:
        need(np.isclose(aggregate[key],doc[key],rtol=0,atol=1e-12),f"{path}: summary {key} inconsistent")
    rows=[]
    for rec in records:
        ep=int(rec["episode_index"])
        detailed_path=path.parent/f"episode_{ep:06d}.json"
        details=read(detailed_path)
        need(all(details[k]==v for k,v in rec.items()),f"Episode {ep}: detailed JSON differs")
        archive_path=path.parent/f"episode_{ep:06d}.npz"
        with np.load(archive_path,allow_pickle=False) as data:
            states,actions,coverage,goal=[data[k] for k in ["states","actions","coverage","goal_pose"]]
            steps=len(actions)
            need(steps==rec["steps"] and steps<=300,f"Episode {ep}: steps differ")
            need(states.shape==(steps+1,5) and actions.shape==(steps,2),f"Episode {ep}: invalid state/action shape")
            need(coverage.shape==(steps+1,),f"Episode {ep}: invalid coverage shape")
            need(np.isfinite(states).all() and np.isfinite(actions).all(),f"Episode {ep}: nonfinite states/actions")
            need(((actions>=0)&(actions<=512)).all(),f"Episode {ep}: actions out of world bounds")
            need(np.array_equal(states[0],rec["initial_state"]) and np.array_equal(states[-1],rec["final_state"]),
                 f"Episode {ep}: JSON state differs from trajectory")
            need(np.isclose(coverage[-1],rec["final_coverage"],atol=1e-12,rtol=0),f"Episode {ep}: final coverage mismatch")
            need(np.isclose(coverage.max(),rec["max_coverage"],atol=1e-12,rtol=0),f"Episode {ep}: maximum coverage mismatch")
            need(bool(coverage.max()>=.95)==rec["success"],f"Episode {ep}: success mismatch")
            need(bool(coverage.max()>.95)==rec["native_success"],f"Episode {ep}: native success mismatch")
            shape=Polygon(world(OUTLINE,goal))
            recomputed=np.array([Polygon(world(OUTLINE,state[2:])).intersection(shape).area/shape.area for state in states])
            coverage_error=float(np.max(np.abs(recomputed-coverage)))
            need(coverage_error<1e-9,f"Episode {ep}: geometry coverage calculation differs")
            need(data["observations_rgb"].shape==(steps+1,96,96,3) and data["observations_rgb"].dtype==np.uint8,
                 f"Episode {ep}: invalid observed RGB images")
            contacts=data["contact_event_counts"]
            need(contacts.shape==(steps,),f"Episode {ep}: invalid contact counts")
            need(np.count_nonzero(contacts)==rec["contact_intervals"] and int(contacts.sum())==rec["contact_events"],
                 f"Episode {ep}: contact count mismatch")
            desired=data["desired_flow_world"]
            need(desired.shape==(steps,8,64,2),f"Episode {ep}: invalid planner flow shape")
            flow_error=0.0
            for t in range(steps):
                error=float(np.max(np.abs(independent_flow(states[t],goal)-desired[t])))
                flow_error=max(flow_error,error)
            need(flow_error<2e-5,f"Episode {ep}: saved flow differs from current-state-only planner")
            rows.append({"episode_index":ep,"steps":steps,"max_coverage_recompute_error":coverage_error,
                         "max_current_state_planner_flow_error":flow_error,"trajectory_sha256":digest(archive_path)})
    return {"path":str(path.relative_to(ROOT)),"method":doc["method"],"checkpoint":doc["checkpoint"],
            "summary_sha256":digest(path),"summary":aggregate,"trajectory_checks":rows}


def verify_contact_model(directory, expected_ids, split_hash):
    path=directory/"test_metrics.json"
    metrics=read(path)
    config=read(directory/"config.json")
    history=read(directory/"history.json")
    need(metrics["split"]=="test",f"{path}: not test")
    need(metrics["checkpoint_sha256"]==digest(directory/"best.pt"),f"{path}: checkpoint mismatch")
    need(config["split_sha256"]==split_hash,f"{path}: split mismatch")
    need(config["source_sha256"]==digest(ROOT/"research/contact_learning.py"),f"{path}: trained implementation changed")
    need(metrics["config"]==config,f"{path}: saved config differs")
    actual=tuple(config[k] for k in ["points","mode","label","task","seed"])
    need(actual==EXPECTED_CONFIGS[directory.name],f"{path}: model configuration differs from declared experiment")
    criterion="mean_contact_distance" if config["task"]=="contact" else "first_action_target_error"
    need(all(np.isfinite(row["validation"][criterion]) and np.isfinite(row["train_loss"]) for row in history),
         f"{path}: nonfinite training/validation history")
    best=min(history,key=lambda x:x["validation"][criterion])
    need(metrics["chosen_epoch"]==best["epoch"],f"{path}: checkpoint not chosen by recorded validation criterion")
    need(len(history)==config["epochs"],f"{path}: incomplete training history")
    with np.load(directory/"test_predictions.npz",allow_pickle=False) as d:
        need(np.array_equal(d["ids"],expected_ids),f"{path}: test prediction IDs differ from frozen windows")
        need(np.isfinite(d["prediction"]).all(),f"{path}: non-finite predictions")
        if config["task"]=="contact":
            need(d["prediction"].shape==(len(expected_ids),config["points"]),f"{path}: prediction shape mismatch")
            need(d["presence"].shape==(len(expected_ids),),f"{path}: presence shape mismatch")
            need(((d["prediction"]>=0)&(d["prediction"]<=1)).all(),f"{path}: score bounds")
            distance=d["distance"]
            need(distance.shape==d["prediction"].shape and np.isfinite(distance).all() and (distance>=0).all(),
                 f"{path}: distance targets invalid")
            positive=distance.min(1)<511
            errors=distance[np.arange(len(distance)),d["prediction"].argmax(1)]
            top5=np.take_along_axis(distance,np.argsort(-d["prediction"],axis=1)[:,:5],axis=1).min(1)
            presence=d["presence"]>=.5
            tp=np.count_nonzero(presence&positive)
            fp=np.count_nonzero(presence&~positive)
            fn=np.count_nonzero(~presence&positive)
            by_episode=[float((errors[(expected_ids[:,0]==ep)&positive]<=10).mean())
                        for ep in np.unique(expected_ids[:,0]) if ((expected_ids[:,0]==ep)&positive).any()]
            rebuilt={"windows":len(expected_ids),"contact_windows":int(positive.sum()),
                     "top1_within10":float((errors[positive]<=10).mean()),
                     "top5_within10":float((top5[positive]<=10).mean()),
                     "mean_contact_distance":float(errors[positive].mean()),
                     "median_contact_distance":float(np.median(errors[positive])),
                     "episode_macro_top1_within10":float(np.mean(by_episode)),
                     "presence_f1":float(2*tp/max(2*tp+fp+fn,1)),
                     "presence_balanced_accuracy":float(.5*(presence[positive].mean()+(~presence[~positive]).mean())),
                     "no_contact_false_positive_rate":float(presence[~positive].mean())}
        else:
            need(d["prediction"].shape==(len(expected_ids),8,2),f"{path}: action shape mismatch")
            targets=np.empty_like(d["prediction"])
            for ep in np.unique(expected_ids[:,0]):
                rows=np.flatnonzero(expected_ids[:,0]==ep)
                with np.load(ROOT/f"data/derived/replayed_contacts/episode_{ep:06d}.npz",allow_pickle=False) as raw:
                    states,actions=raw["states"],raw["actions"]
                for index in rows:
                    t=expected_ids[index,1]
                    angle=states[t,4]
                    rotation=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
                    targets[index]=(actions[t:t+8]-states[t,:2])@rotation/100
            errors=np.linalg.norm((d["prediction"]-targets)*100,axis=-1)
            rebuilt={"windows":len(expected_ids),"first_action_target_error":float(errors[:,0].mean()),
                     "action_sequence_target_error":float(errors.mean())}
        for key,value in rebuilt.items():
            need(np.isclose(metrics["metrics"][key],value,rtol=0,atol=1e-6),f"{path}: offline metric {key} differs from saved predictions")
    return {"model":directory.name,"task":config["task"],"seed":config["seed"],
            "test_windows":len(expected_ids),"chosen_epoch":best["epoch"],"checkpoint_sha256":digest(directory/"best.pt"),
            "independently_recomputed_metrics":rebuilt}


def verify_visual_training(directory, splits, split_hash):
    manifest=read(directory/"training_manifest.json")
    result=read(directory/"training_result.json")
    history=read(directory/"history.json")
    need(manifest["train_episode_ids"]==splits["train"],"Visual DP train episodes differ")
    need(manifest["validation_episode_ids"]==splits["validation"],"Visual DP val episodes differ")
    need(manifest["split_sha256"]==split_hash,"Visual DP split hash mismatch")
    need(manifest["implementation_sha256"]==digest(ROOT/"research/visual_dp.py"),"Visual DP trained implementation changed")
    need(manifest["upstream_unet_sha256"]==digest(ROOT/"external/diffusion_policy/diffusion_policy/model/diffusion/conditional_unet1d.py"),
         "Visual DP upstream U-Net changed")
    need(result["best_checkpoint_sha256"]==digest(directory/"best.pt"),"Visual DP checkpoint changed")
    need(result["epochs"]==len(history),"Visual DP history incomplete")
    need(all(np.isfinite(row["validation_noise_mse"]) and np.isfinite(row["train_noise_mse"]) for row in history),
         "Visual DP has nonfinite training/validation history")
    best=min(history,key=lambda row:row["validation_noise_mse"])
    need(result["best_epoch"]==best["epoch"],"Visual DP checkpoint selection differs from validation")
    cache_path=resolve(manifest["arguments"]["cache"])
    need(digest(cache_path/"cache_manifest.json")==manifest["image_cache_manifest_sha256"],"Visual cache manifest changed")
    with np.load(cache_path/"metadata.npz",allow_pickle=False) as meta, np.load(directory/"window_indices.npz",allow_pickle=False) as indexes:
        for name in ["train","validation"]:
            anchors=indexes[name]
            need(len(anchors)==manifest[f"{name}_samples"],f"Visual DP {name} count differs")
            need(len(np.unique(anchors))==len(anchors),f"Visual DP duplicate {name} anchors")
            need(np.isin(meta["episodes"][anchors],splits[name]).all(),f"Visual DP {name} split leakage")
        offline=read(directory/"test_metrics.json")
        need(offline["split"]=="test" and offline["checkpoint_sha256"]==digest(directory/"best.pt"),
             "Visual DP offline test checkpoint/split mismatch")
        expected=np.flatnonzero(np.isin(meta["episodes"],splits["test"]))
        rows=offline["per_window"]
        need(np.array_equal([r["anchor"] for r in rows],expected),"Visual DP offline test is incomplete or repeats/reorders frames")
        need(offline["windows"]==len(rows) and offline["episodes"]==21,"Visual DP offline count mismatch")
        need([r["episode"] for r in offline["per_episode"]]==splits["test"],"Visual DP offline episode IDs mismatch")
        for row in rows:
            anchor=row["anchor"]
            need(row["episode"]==int(meta["episodes"][anchor]) and row["frame"]==int(meta["frames"][anchor]),
                 "Visual DP offline frame/episode alignment mismatch")
            need(all(np.isfinite(row[k]) and row[k]>=0 for k in ["first_action_l2","valid_horizon_action_l2","hold_baseline_action_l2"]),
                 "Visual DP offline action errors invalid")
        need(np.isclose(offline["mean_first_action_l2"],np.mean([r["first_action_l2"] for r in rows]),atol=1e-10),
             "Visual DP offline first-action mean differs")
        need(np.isclose(offline["mean_horizon_action_l2"],np.mean([r["valid_horizon_action_l2"] for r in rows]),atol=1e-10),
             "Visual DP offline sequence-action mean differs")
    return {"path":str(directory.relative_to(ROOT)),"epochs":result["epochs"],"best_epoch":result["best_epoch"],
            "checkpoint_sha256":result["best_checkpoint_sha256"],"train_samples":manifest["train_samples"],
            "validation_samples":manifest["validation_samples"],"offline_test_frames":len(rows),
            "offline_test_sha256":digest(directory/"test_metrics.json")}


def verify_log_recovery(base,test_ids):
    path=base/"log_collision_recovery.json"
    doc=read(path)
    need(doc["passed"] and doc["all_recovery_commands_returned_zero"],"Logging recovery not completed")
    protected=read(resolve(doc["protected_before_manifest"]))
    for relative,expected in protected.items():
        need(digest(ROOT/relative)==expected,f"Original artifact changed during logging recovery: {relative}")
    field_count=0
    maximum_difference=0.0
    commands=[]
    need({r["run"] for r in doc["runs"]}=={"contact","flowaction","seed1","seed2"},"Logging recovery missing a model")
    for run in doc["runs"]:
        comparison=read(resolve(run["comparison_file"]))
        original,fresh=resolve(comparison["original_directory"]),resolve(comparison["fresh_directory"])
        command=read(resolve(run["command_record"]))
        need(command["returncode"]==0 and run["returncode"]==0,"Logging recovery process failed")
        need("research/evaluate_closed_loop.py" in command["command"],"Recovery command was not closed-loop execution")
        output_index=command["command"].index("--output-dir")+1
        need(resolve(command["command"][output_index]).resolve()==fresh.resolve(),"Recovery command points to wrong output")
        need(resolve(command["stdout"]).exists() and resolve(command["stderr"]).exists(),"Fresh subprocess logs missing")
        commands.append({"record":run["command_record"],"returncode":0,
                         "stdout_sha256":digest(resolve(command["stdout"])),
                         "stderr_sha256":digest(resolve(command["stderr"])),
                         "stderr_bytes":resolve(command["stderr"]).stat().st_size})
        for ep in test_ids:
            with np.load(original/f"episode_{ep:06d}.npz",allow_pickle=False) as a, \
                 np.load(fresh/f"episode_{ep:06d}.npz",allow_pickle=False) as b:
                need(set(a.files)==set(b.files),f"Recovery {run['run']} episode {ep}: fields differ")
                for key in a.files:
                    old,new=a[key],b[key]
                    need(old.shape==new.shape and old.dtype==new.dtype,f"Recovery {run['run']} episode {ep}/{key}: schema differs")
                    if key=="coverage":
                        error=float(np.max(np.abs(old-new)))
                        maximum_difference=max(maximum_difference,error)
                        need(error<=1e-12,f"Recovery {run['run']} episode {ep}: coverage differs materially")
                    else:
                        need(np.array_equal(old,new),f"Recovery {run['run']} episode {ep}/{key}: content differs")
                    field_count+=1
    need(field_count==doc["array_fields_compared"],"Recovery reported field count differs")
    return {"passed":True,"report_sha256":digest(path),"original_files_unchanged":len(protected),
            "rerun_episodes":4*len(test_ids),"compared_array_fields":field_count,
            "coverage_max_absolute_difference":maximum_difference,"commands":commands,
            "defect":doc["defect"],"repair":doc["repair"],
            "interpretation":"Fresh reproducibility evidence supplements overwritten historical logs; reruns are not additional independent test samples."}


def verify_stage2_samples(base,window_doc):
    directory=base/"stage2_samples"
    manifest=read(directory/"manifest.json")
    need(manifest["status"]=="passed" and manifest["samples"]==9,"Stage-2 sample export incomplete")
    need(manifest["output_sha256"]==digest(directory/"samples.npz"),"Stage-2 sample archive changed")
    expected=[]
    for split,records in window_doc["windows"].items():
        for index in [0,len(records)//2,len(records)-1]:
            expected.append((split,*records[index]))
    with np.load(directory/"samples.npz",allow_pickle=False) as samples:
        need(samples["geometry"].shape==(9,64,13) and samples["flow"].shape==(9,64,8,2),"Stage-2 sample feature shape mismatch")
        for i,(split,ep,t) in enumerate(expected):
            need(samples["split_name"][i]==split and np.array_equal(samples["ids"][i],[ep,t]),"Stage-2 display sample changed split or index")
            source=ROOT/f"data/derived/replayed_contacts/episode_{ep:06d}.npz"
            need(digest(source)==manifest["rows"][i]["source_npz_sha256"],"Stage-2 sample source changed")
            with np.load(source,allow_pickle=False) as raw:
                need(np.array_equal(samples["states_world"][i],raw["states"][t:t+9]),"Stage-2 sample states differ")
                need(np.array_equal(samples["actions_world"][i],raw["actions"][t:t+8]),"Stage-2 sample actions differ")
                need(np.allclose(samples["contact_soft_target"][i],raw["contact_soft_labels"][t:t+8].max(0),atol=1e-7),
                     "Stage-2 sample is not an eight-interval label union")
                angle=raw["states"][t,4]
                r=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
                expected_action=(raw["actions"][t:t+8]-raw["states"][t,:2])@r/100
                need(np.allclose(samples["action_target"][i],expected_action,rtol=1e-6,atol=1e-6),"Stage-2 action normalization differs")
                flow=(samples["points_world"][i,1:]-samples["points_world"][i,0])@r/30
                need(np.allclose(samples["flow"][i],flow.transpose(1,0,2),rtol=1e-6,atol=1e-6),"Stage-2 flow anchor/normalization differs")
    return {"passed":True,"samples":9,"splits":{"train":3,"validation":3,"test":3},
            "archive_sha256":digest(directory/"samples.npz"),"manifest_sha256":digest(directory/"manifest.json")}


def verify_runtime_metadata(base):
    before_path=base/"runtime_metadata_before_repair.json"
    after_path=base/"runtime_metadata_after_repair.json"
    before,after=read(before_path),read(after_path)
    need(before["all_selected_source_records_match"],"Pre-repair active package source records did not match")
    need(after["duplicate_metadata"]=={},"Duplicate distribution metadata remains after repair")
    need(after["all_selected_source_records_match"],"Reinstalled active package source records do not match")
    need(Path(before["python"]).resolve()==Path(after["python"]).resolve(),"Runtime audit used different environments")
    checks=[]
    for name,expected in [("filelock","3.32.3"),("fsspec","2026.7.0")]:
        old=before["active_package_record_checks"][name]
        new=after["active_package_record_checks"][name]
        need(old["selected_version"]==new["selected_version"]==expected,f"Runtime repair changed active {name} version")
        need(not old["mismatches"] and not new["mismatches"],f"Runtime {name} source mismatch")
        need(old["checked_source_files"]==new["checked_source_files"]>0,f"Runtime {name} checked-file count differs")
        checks.append({"package":name,"version":expected,"source_files_checked":new["checked_source_files"]})
    target_path=base/"runtime_repair_targets.json"
    targets=json.loads(target_path.read_text(encoding="utf-8-sig"))
    need(len(targets)==6,"Runtime repair backup target list differs")
    for target in targets:
        need((base/"runtime_repair_backup"/Path(target).name).is_dir(),f"Runtime original package backup missing: {target}")
    return {"passed":True,"before_sha256":digest(before_path),"after_sha256":digest(after_path),
            "backup_target_manifest_sha256":digest(target_path),"preserved_original_directories":len(targets),
            "duplicates_before":before["duplicate_metadata"],"duplicates_after":{},"active_versions_unchanged":checks,
            "interpretation":"Duplicate metadata was repaired after training using the same selected package versions; active package sources matched their wheel RECORD entries before and after repair."}


def verify_visual_reverification(base,test_ids):
    comparison=base/"fresh_verification/visual_dp_comparison.json"
    need(comparison.exists(),"Visual DP fresh-process comparison record missing")
    original=base/"visual_dp_test"
    fresh=base/"fresh_verification/visual_dp_episode12"
    episode=test_ids[0]
    before,after=read(original/"summary.json"),read(fresh/"summary.json")
    need(after["episodes"]==[episode],"Visual DP reproducibility run must repeat only the first test episode")
    need(before["checkpoint_sha256"]==after["checkpoint_sha256"],"Visual DP reproducibility checkpoint differs")
    files={}
    with np.load(original/f"episode_{episode:06d}.npz",allow_pickle=False) as a, \
         np.load(fresh/f"episode_{episode:06d}.npz",allow_pickle=False) as b:
        need(set(a.files)==set(b.files),"Visual DP reproducibility array fields differ")
        for key in a.files:
            need(a[key].shape==b[key].shape and a[key].dtype==b[key].dtype,f"Visual DP reproducibility schema differs: {key}")
            error=float(np.max(np.abs(a[key].astype(float)-b[key].astype(float)))) if a[key].size else 0.
            need(np.allclose(a[key],b[key],atol=1e-12,rtol=0) if key=="coverage" else np.array_equal(a[key],b[key]),
                 f"Visual DP reproducibility array differs: {key}, maximum absolute difference {error}")
            files[key]={"exactly_equal":bool(np.array_equal(a[key],b[key])),"maximum_absolute_difference":error}
    return {"passed":True,"episode_index":episode,"comparison_report_sha256":digest(comparison),
            "original_trajectory_sha256":digest(original/f"episode_{episode:06d}.npz"),
            "fresh_trajectory_sha256":digest(fresh/f"episode_{episode:06d}.npz"),
            "arrays":files,"new_statistical_samples":0}


def verify_visual_contract(base,splits):
    path=base/"visual_dp_contract_audit/result.json"
    report=read(path)
    command_path=base/"logs/visual_dp_contract_audit_v2.command.json"
    command=read(command_path)
    need(report["passed"] and report["protected_files_unchanged"] and command["returncode"]==0,
         "Visual DP interface audit did not pass")
    for relative,expected in report["protected_files_sha256"].items():
        need(digest(ROOT/relative)==expected,f"Visual DP interface-audited source/weights changed: {relative}")
    windows=report["window_boundary_checks"]
    need(len(windows)==24,"Visual DP contract audit window count differs")
    for window in windows:
        need(window["split"] in ["train","validation"] and window["episode"] in splits[window["split"]],
             "Visual DP contract audit unexpectedly used held-out test demonstrations")
    need(len(report["exact_rgb_video_cache_frames_checked"])==4,"Visual DP raw RGB cache checks missing")
    need(report["action_normalization_roundtrip_max_error"]==0 and report["batch_vs_online_packing_max_action_error"]==0,
         "Visual DP normalization or online/batch packing differs")
    spy=report["online_history_spy"]
    need(spy["matched"] and spy["at_steps"]==[0,8,16] and spy["calls"]==3,"Visual DP observation chunk timing differs")
    replay=report["validation_episode11_replay_checks"]
    need(len(replay)==10 and max(r["pusher_xy_error"] for r in replay)<1e-4,
         "Visual DP replay pusher alignment differs")
    need(min(r["visible_mask_iou"] for r in replay)>.95,"Visual DP replay image-mask comparison differs")
    need(report["output_shape"]==[8,2],"Visual DP online output chunk shape differs")
    for stream in ["stdout","stderr"]:
        need(resolve(command[stream]).is_file(),f"Visual DP contract {stream} log missing")
    return {"passed":True,"report_sha256":digest(path),"command_record_sha256":digest(command_path),
            "protected_source_and_weight_files":len(report["protected_files_sha256"]),
            "checked_windows":len(windows),"checked_rgb_frames":4,"new_test_samples":0,
            "replay_pusher_max_error":max(r["pusher_xy_error"] for r in replay),
            "replay_visible_mask_min_iou":min(r["visible_mask_iou"] for r in replay),
            "scope":report["scope"],"limitations":report["limitations"]}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir",type=Path,default=ROOT/"runs/stages234_20261022")
    parser.add_argument("--output",type=Path)
    parser.add_argument("--partial",action="store_true",help="Interim audit only; does not certify full completion")
    args=parser.parse_args()
    base=args.run_dir.resolve()
    output=args.output.resolve() if args.output else base/"final_acceptance.json"
    if output.exists():
        raise FileExistsError(output)
    report={"status":"failed","created_at_utc":datetime.now(timezone.utc).isoformat(),
            "script_sha256":digest(__file__),"command":sys.argv,"partial":args.partial,"errors":[]}
    start=time.perf_counter()
    try:
        splits_doc=read(ROOT/"data/derived/splits.json")
        splits=splits_doc["splits"]
        flat=sum(list(splits.values()),[])
        need(len(flat)==len(set(flat))==206,"Train/validation/test split overlap")
        need([len(splits[k]) for k in ["train","validation","test"]]==[164,21,21],"Unexpected split counts")
        split_hash=digest(ROOT/"data/derived/splits.json")
        windows=read(ROOT/"data/derived/windows.json")
        expected_ids=np.array(windows["windows"]["test"])
        report["stage2_sample_checks"]=verify_stage2_samples(base,windows)
        audited=read(base/"data_audit.json")
        need(audited["status"]=="passed","Fresh data audit failed")
        for relative,sha in audited["protected_files_sha256"].items():
            need(digest(ROOT/relative)==sha,f"Protected stage-1 artifact changed: {relative}")
        report["protected_stage1_files_unchanged"]=len(audited["protected_files_sha256"])
        checks=read(base/"controller_checks.json")
        need(checks["passed"] and checks["npz_and_parquet_readers_blocked_during_planning"],"Online isolation check missing")
        need(read(base/"contact_feature_checks.json")["status"]=="passed","Contact feature check missing")
        need(read(base/"visual_window_checks.json")["status"]=="passed","Visual window check missing")
        freeze_files=list(base.glob("controller_freeze*.json"))
        need(bool(freeze_files),"No pre-test controller freeze")
        freeze_path=max(freeze_files,key=lambda p:p.stat().st_mtime_ns)
        frozen=read(freeze_path)
        report["controller_freeze"]=str(freeze_path.relative_to(ROOT))
        closed=[]
        for path in sorted(base.rglob("summary.json")):
            doc=read(path)
            if doc.get("split")=="test" and "results" in doc:
                expected_episodes=splits["test"]
                if path.parent==base/"fresh_verification/visual_dp_episode12":
                    need(doc["method"]=="visual_dp","Unexpected method in visual reproducibility directory")
                    expected_episodes=splits["test"][:1]
                checked=verify_closed_loop(path,expected_episodes,split_hash,frozen)
                checked["evaluation_kind"]="primary" if path.parent.parent==base else "reverification"
                if path.parent==base/"contact_no_flow_test":
                    checked["evaluation_kind"]="supplementary_diagnostic"
                    protocol=base/"contact_no_flow_diagnostic_protocol.json"
                    need(protocol.exists(),"Post-hoc no-flow diagnostic has no declared protocol")
                    checked["diagnostic_protocol_sha256"]=digest(protocol)
                closed.append(checked)
        report["closed_loop"]=closed
        primary=[r for r in closed if r["evaluation_kind"]=="primary"]
        report["primary_closed_loop_run_count"]=len(primary)
        report["reverification_closed_loop_run_count"]=sum(r["evaluation_kind"]=="reverification" for r in closed)
        report["supplementary_diagnostic_count"]=sum(r["evaluation_kind"]=="supplementary_diagnostic" for r in closed)
        methods={r["method"] for r in primary}
        if not args.partial:
            need(methods=={"rule","learned_contact","flow_action","visual_dp"},f"Incomplete closed-loop methods: {methods}")
            contact_checkpoints={Path(r["checkpoint"]).parent.name for r in primary if r["method"]=="learned_contact"}
            need({"contact_main","contact_seed1","contact_seed2"}<=contact_checkpoints,"Three contact-training seeds not all evaluated")
        offline=[]
        for name in EXPECTED_MODELS:
            directory=base/name
            if args.partial and not (directory/"test_metrics.json").exists():
                continue
            offline.append(verify_contact_model(directory,expected_ids,split_hash))
        report["offline_models"]=offline
        visual_directories={resolve(r["checkpoint"]).parent for r in closed if r["method"]=="visual_dp"}
        report["visual_training"]=[verify_visual_training(d,splits,split_hash) for d in sorted(visual_directories)]
        if (base/"log_collision_recovery.json").exists() or not args.partial:
            report["log_collision_recovery"]=verify_log_recovery(base,splits["test"])
        if (base/"runtime_metadata_after_repair.json").exists() or not args.partial:
            report["runtime_metadata_repair"]=verify_runtime_metadata(base)
        if (base/"fresh_verification/visual_dp_comparison.json").exists() or not args.partial:
            report["visual_dp_reverification"]=verify_visual_reverification(base,splits["test"])
        if (base/"visual_dp_contract_audit/result.json").exists() or not args.partial:
            report["visual_dp_contract_audit"]=verify_visual_contract(base,splits)
        report["status"]="partial_passed" if args.partial else "passed"
        report["scope"]="Saved experiment consistency and declared leakage guards; not proof of scientific superiority or real-world generalization."
    except Exception as exc:
        report["errors"].append(f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report["elapsed_seconds"]=time.perf_counter()-start
        output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
        print(json.dumps({"status":report["status"],"errors":report["errors"],"elapsed_seconds":report["elapsed_seconds"]}),flush=True)


if __name__=="__main__":
    main()
