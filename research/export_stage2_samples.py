"""Export nine real offline model samples without loading the full dataset."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np

from contact_learning import features, rotation

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,default=ROOT/"runs/stages234_20261022/stage2_samples")
    args=parser.parse_args()
    output=args.output_dir
    if output.exists():
        raise FileExistsError(output)
    windows=ROOT/"data/derived/windows.json"
    manifest=json.loads(windows.read_text())
    selected=[]
    for split,records in manifest["windows"].items():
        for index in [0,len(records)//2,len(records)-1]:
            selected.append((split,*records[index]))
    data={key:[] for key in ["ids","split_name","geometry","flow","contact_soft_target","contact_presence",
                            "distance_to_future_contact","action_target","anchor_state","states_world",
                            "points_world","future_flow_world","actions_world","per_interval_soft_labels",
                            "contact_label_valid","transition_has_video_endpoint"]}
    source_rows=[]
    event_rows=[]
    for sample,(split,ep,t) in enumerate(selected):
        path=ROOT/f"data/derived/replayed_contacts/episode_{ep:06d}.npz"
        with np.load(path,allow_pickle=False) as z:
            state=z["states"][t]
            points=z["points_world"][t:t+9]
            flow=points[1:]-points[0]
            geo,anchor_flow=features(state,flow,64)
            events=z["contact_events"]
            chosen=events[(events["action_index"]>=t)&(events["action_index"]<t+8)&events["on_exterior"]]
            local=z["local_boundary_points"]
            distance=(np.linalg.norm(local[:,None]-chosen["point_local"][None],axis=-1).min(1)
                      if len(chosen) else np.full(64,512.))
            target=np.exp(-.5*(distance/5)**2) if len(chosen) else np.zeros(64)
            np.testing.assert_allclose(target,z["contact_soft_labels"][t:t+8].max(0),rtol=1e-6,atol=1e-7)
            assert z["contact_label_valid"][t:t+8].all() and z["transition_has_video_endpoint"][t:t+8].all()
            transformed_actions=(z["actions"][t:t+8]-state[:2])@rotation(state[4])/100
            values={"ids":np.array([ep,t],np.int32),"split_name":split,"geometry":geo[0],"flow":anchor_flow[0],
                    "contact_soft_target":target.astype(np.float32),"contact_presence":bool(len(chosen)),
                    "distance_to_future_contact":distance.astype(np.float32),"action_target":transformed_actions.astype(np.float32),
                    "anchor_state":state,"states_world":z["states"][t:t+9],"points_world":points,
                    "future_flow_world":flow,"actions_world":z["actions"][t:t+8],
                    "per_interval_soft_labels":z["contact_soft_labels"][t:t+8],
                    "contact_label_valid":z["contact_label_valid"][t:t+8],
                    "transition_has_video_endpoint":z["transition_has_video_endpoint"][t:t+8]}
            for key,value in values.items():
                data[key].append(value)
            extended=np.empty(len(chosen),dtype=np.dtype([("sample_index","i4")]+chosen.dtype.descr))
            extended["sample_index"]=sample
            for name in chosen.dtype.names:
                extended[name]=chosen[name]
            event_rows.append(extended)
            source_rows.append({"sample_index":sample,"split":split,"episode_index":ep,"start_action_index":t,
                                "contact_events":len(chosen),"source_npz_sha256":sha(path)})
    arrays={k:np.asarray(v) for k,v in data.items()}
    arrays["contact_events"]=np.concatenate(event_rows)
    output.mkdir(parents=True)
    np.savez_compressed(output/"samples.npz",**arrays)
    # Reopen the actual saved artifact in this process, with no object pickles.
    with np.load(output/"samples.npz",allow_pickle=False) as saved:
        assert saved["geometry"].shape==(9,64,13)
        assert saved["flow"].shape==(9,64,8,2)
        assert saved["action_target"].shape==(9,8,2)
        assert set(saved["split_name"])=={"train","validation","test"}
        for name in saved.files:
            assert np.array_equal(saved[name],arrays[name])
    report={"status":"passed","samples":9,"selection":"first, middle, last frozen window from each split; display only",
            "source_windows_sha256":sha(windows),"source_splits_sha256":sha(ROOT/"data/derived/splits.json"),
            "feature_implementation_sha256":sha(ROOT/"research/contact_learning.py"),"script_sha256":sha(__file__),
            "output_sha256":sha(output/"samples.npz"),"rows":source_rows,
            "arrays":{k:{"shape":list(v.shape),"dtype":str(v.dtype)} for k,v in arrays.items()},
            "schema_document":"docs/STAGES234_DATA_SCHEMA.md",
            "future_data_warning":"These are OFFLINE supervised samples. Future demonstration motion and actions must not be passed to a closed-loop policy."}
    (output/"manifest.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"status":"passed","samples":9,"output":str(output.resolve()),
                      "contact_event_rows":len(arrays["contact_events"])}),flush=True)


if __name__=="__main__":
    main()
