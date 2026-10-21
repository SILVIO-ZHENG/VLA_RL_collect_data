"""Read-only numerical/provenance audit. Writes only into a separate report directory."""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pyarrow.parquet as pq

ARMS = list(range(7)) + list(range(8, 15))


def stats(x):
    x = np.asarray(x).reshape(-1)
    x = x[np.isfinite(x)]
    if not len(x):
        return {"n": 0}
    return dict(n=len(x), mean=float(x.mean()), **{
        name: float(np.quantile(x, q)) for name, q in
        [("min", 0), ("p50", .5), ("p95", .95), ("p99", .99), ("max", 1)]})


def runs(mask):
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False].astype(np.int8)))
    return list(zip(edges[::2], edges[1::2]))


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)


def audit(root, out, urdf):
    out.mkdir(parents=True, exist_ok=True)
    info = json.loads((root / "meta/info.json").read_text())
    meta = pq.read_table(root / "meta/episodes").to_pylist()
    tab = pq.read_table(root / "data")
    arrays = {}
    for col in ["action", "observation.state"]:
        v = tab[col].combine_chunks()
        assert np.all(np.diff(v.offsets.to_numpy()) == 24)
        arrays[col] = v.values.to_numpy().reshape(-1, 24)
    a, s = arrays["action"], arrays["observation.state"]
    ep, fi, ts, ix, done, task = [tab[c].combine_chunks().to_numpy(zero_copy_only=False)
        for c in ["episode_index", "frame_index", "timestamp", "index", "next.done", "task_index"]]
    cuts = np.flatnonzero(np.r_[True, ep[1:] != ep[:-1], True])
    spans = list(zip(cuts[:-1], cuts[1:]))
    assert len(spans) == len(meta)
    fps = info["fps"]
    report = {"dataset": str(root), "frames": len(a), "episodes": len(spans), "fps": fps,
              "definitions": {"open_m": .019, "closed_m": .006, "no_movement_span_m": .0005,
                "short_pulse_max_s_exclusive": .5, "action_mapping": "physical_opening=0.025-action",
                "limits_source": str(urdf), "derivatives": "within episode, delta*30, delta2*900, delta3*27000"}}
    print("Loaded", len(a), "frames", flush=True)
    loc = lambda i: {"episode": int(ep[i]), "frame": int(fi[i]), "time_s": float(ts[i])}
    mask1 = ep[1:] == ep[:-1]
    mask2 = ep[2:] == ep[:-2]
    mask3 = ep[3:] == ep[:-3]
    report["integrity"] = dict(
        finite_action=bool(np.isfinite(a).all()), finite_state=bool(np.isfinite(s).all()),
        index_mismatch=int(np.sum(ix != np.arange(len(ix)))),
        episode_ids_contiguous=bool(np.array_equal(np.unique(ep), np.arange(len(spans)))),
        timestamp_error_max_s=float(np.max(np.abs(ts-fi/fps))),
        dt_stats_s=stats(np.diff(ts)[mask1]), task_counts=dict(collections.Counter(map(str, task))))
    bad_meta = []
    for (b, e), m in zip(spans, meta):
        if (not np.array_equal(fi[b:e], np.arange(e-b)) or m["length"] != e-b or
            m["dataset_from_index"] != b or m["dataset_to_index"] != e or
            m["episode_index"] != ep[b] or not done[e-1] or done[b:e-1].any()):
            bad_meta.append(int(ep[b]))
    report["integrity"]["bad_episode_metadata_or_done"] = bad_meta
    report["episode_length_s"] = stats(np.diff(cuts)/fps)
    report["gripper_ranges"] = {name: {str(j): stats(x[:, j]) for j in [7,15]} for name,x in arrays.items()}
    names = info["features"]["action"]["names"]
    root_xml = ET.parse(urdf).getroot()
    limits = {}
    for side, ids, prefix in [("l",range(7),20),("r",range(8,15),60)]:
        for k, j in enumerate(ids,1):
            elem=root_xml.find(f"./joint[@name='idx{prefix+k}_arm_{side}_joint{k}']/limit")
            limits[j] = tuple(float(elem.attrib[t]) for t in ["lower","upper","velocity"])
    dynamic = []; suspect = []; overshoot = []
    for name,x in arrays.items():
        for j in ARMS:
            lo,hi,vel=limits[j]; q=x[:,j].astype(float)
            v=np.diff(q)*fps; ac=np.diff(q,2)*fps**2; jerk=np.diff(q,3)*fps**3
            excess=np.maximum(lo-q,q-hi)
            d={"source":name,"joint":names[j],"position":stats(q),"speed_abs":stats(abs(v[mask1])),
               "acceleration_abs":stats(abs(ac[mask2])),"jerk_abs":stats(abs(jerk[mask3])),
               "urdf_limits":[lo,hi,vel],"outside_position_1mrad_frames":int((excess>.001).sum()),
               "outside_position_10mrad_frames":int((excess>.01).sum()),
               "inside_deployment_10mrad_margin_violations":int(((q<lo+.01)|(q>hi-.01)).sum()),
               "over_urdf_velocity_intervals":int(((abs(v)>vel)&mask1).sum()),
               "max_position_excess_rad":float(max(0,excess.max()))}
            dynamic.append(d)
            # Ranked diagnostic points; threshold is screening, not a hardware limit.
            valid=np.where(mask1,abs(v),-1)
            for i in np.argsort(valid)[-5:][::-1]+1:
                suspect.append(dict(source=name,joint=names[j],kind="speed_peak",value=float(v[i-1]),**loc(i)))
            if (excess>.001).any():
                i=int(excess.argmax());overshoot.append(dict(source=name,joint=names[j],value=float(q[i]),excess_rad=float(excess[i]),**loc(i)))
    report["dynamics"] = dynamic
    write_csv(out/"关节动态峰值.csv",suspect)
    write_csv(out/"关节位置越界样例.csv",overshoot)
    # Align source before any comparison to the pre-SG data.
    manifest=json.loads((root/"smoothing_manifest.json").read_text())
    rawroot=Path(manifest["source_dataset"])
    rawtab=pq.read_table(rawroot/"data",columns=["action","observation.state","episode_index","frame_index"])
    assert np.array_equal(rawtab["episode_index"].combine_chunks().to_numpy(),ep)
    assert np.array_equal(rawtab["frame_index"].combine_chunks().to_numpy(),fi)
    raw=rawtab["action"].combine_chunks().values.to_numpy().reshape(-1,24)
    rawstate=rawtab["observation.state"].combine_chunks().values.to_numpy().reshape(-1,24)
    residual=np.max(abs(a[:,ARMS]-raw[:,ARMS]),axis=1)
    report["smoothing"]={"source":str(rawroot),"state_exactly_preserved":bool(np.array_equal(s,rawstate)),
        "grippers_exactly_preserved":bool(np.array_equal(a[:,[7,15]],raw[:,[7,15]])),
        "nonarm_exactly_preserved":bool(np.array_equal(a[:,[7,15]+list(range(16,24))],raw[:,[7,15]+list(range(16,24))])),
        "max_joint_change_per_frame_rad":stats(residual),
        "thresholds":[{"rad":v,"frames":int((residual>v).sum()),"episodes":len(np.unique(ep[residual>v]))} for v in [.01,.03,.05,.1]]}
    smooth_events=[]
    for b,e in spans:
        for l,r in runs(residual[b:e]>.03):
            i=b+l+int(np.argmax(residual[b+l:b+r]));j=ARMS[int(np.argmax(abs(a[i,ARMS]-raw[i,ARMS])))]
            smooth_events.append(dict(**loc(i),joint=names[j],start_frame=int(l),end_frame_exclusive=int(r),
                raw_action=float(raw[i,j]),smooth_action=float(a[i,j]),state=float(s[i,j]),max_change_rad=float(residual[i])))
    write_csv(out/"SG大幅改动区间.csv",smooth_events)
    # Errors, stuck-feedback candidates and activity segmentation.
    error=np.max(abs(a[:,ARMS]-s[:,ARMS]),axis=1)
    report["tracking"]={"max_arm_error_rad":stats(error),"over_0.1_frames":int((error>.1).sum()),
        "over_0.3_frames":int((error>.3).sum()),"over_0.3_episodes":len(np.unique(ep[error>.3]))}
    tracking_events=[]; episode_rows=[]; hashes=collections.defaultdict(list); frozen=[]
    for b,e in spans:
        for l,r in runs(error[b:e]>.3):
            if r-l>=3:
                i=b+l+int(np.argmax(error[b+l:b+r]));j=ARMS[int(np.argmax(abs(a[i,ARMS]-s[i,ARMS])))]
                tracking_events.append(dict(**loc(i),joint=names[j],start_frame=int(l),end_frame_exclusive=int(r),duration_s=(r-l)/fps,max_error_rad=float(error[i])))
        # Joint trace freeze with changing commands: candidates only, can also be control disabled.
        sd=np.max(abs(np.diff(s[b:e,ARMS],axis=0)),axis=1)
        ad=np.max(abs(np.diff(a[b:e,ARMS],axis=0)),axis=1)
        for l,r in runs(sd<=1e-7):
            if r-l>=15:
                asp=float(np.max(np.ptp(a[b+l:b+r+1,ARMS],axis=0)))
                if asp>.02:frozen.append(dict(**loc(b+l),end_frame_exclusive=int(r+1),duration_s=(r-l+1)/fps,action_span_rad=asp))
        episode_rows.append(dict(episode=int(ep[b]),frames=int(e-b),duration_s=(e-b)/fps,
            action_motion_fraction=float((ad>.001).mean()),state_motion_fraction=float((sd>.001).mean()),
            max_arm_error_rad=float(error[b:e].max()),max_sg_change_rad=float(residual[b:e].max()),
            start_left_opening_m=float(s[b,7]),end_left_opening_m=float(s[e-1,7]),
            start_right_opening_m=float(s[b,15]),end_right_opening_m=float(s[e-1,15]),
            last_step_arm_rad=float(ad[-1]),first_step_arm_rad=float(ad[0])))
        hashes[hashlib.sha256(a[b:e].tobytes()+s[b:e].tobytes()).hexdigest()].append(int(ep[b]))
    report["tracking"]["sustained_error_over_0.3_3frames_events"]=len(tracking_events)
    report["tracking"]["exact_arm_feedback_freeze_candidates"]=len(frozen)
    report["duplicate_full_episodes"]=[v for v in hashes.values() if len(v)>1]
    write_csv(out/"动作反馈明显偏离区间.csv",tracking_events)
    write_csv(out/"反馈冻结候选.csv",frozen)
    # Full, episode-bounded gripper-event inventory, including censored events.
    events=[]
    for b,e in spans:
        for side,j in [("L",7),("R",15)]:
            command=None;pending=None
            for i in range(b,e):
                goal=.025-float(a[i,j]);sv=float(s[i,j])
                new="C" if goal<=.006 else ("O" if goal>=.019 else command)
                if command is None:command=new;continue
                if new != command:
                    if pending:
                        pending.update(outcome="reversed",end_frame=int(fi[i]),end_time_s=float(ts[i]))
                        events.append(pending)
                    pending=dict(**loc(i),side=side,direction=command+">"+new,start_state_m=sv,
                        min_state_m=sv,max_state_m=sv,outcome=None,end_frame=None,end_time_s=None)
                    command=new
                if pending:
                    pending["min_state_m"]=min(pending["min_state_m"],sv)
                    pending["max_state_m"]=max(pending["max_state_m"],sv)
                    if (command=="C" and sv<=.006) or (command=="O" and sv>=.019):
                        pending.update(outcome="reached",end_frame=int(fi[i]),end_time_s=float(ts[i]))
                        events.append(pending);pending=None
            if pending:
                pending.update(outcome="episode_end",end_frame=int(fi[e-1]),end_time_s=float(ts[e-1]))
                events.append(pending)
    for v in events:
        v["duration_s"]=v["end_time_s"]-v["time_s"]
        v["state_span_m"]=v["max_state_m"]-v["min_state_m"]
        v["strict_short_no_response"]=(v["direction"]=="O>C" and v["start_state_m"]>=.019 and
            v["outcome"]=="reversed" and v["duration_s"]<.5 and v["state_span_m"]<.0005)
    write_csv(out/"夹爪全部事件.csv",events)
    noise=[v for v in events if v["strict_short_no_response"]]
    grouping=collections.defaultdict(list)
    for v in noise:grouping[(v["episode"],v["frame"],v["end_frame"])].append(v)
    report["gripper"]={"all_endpoint_events":len(events),"strict_noise_side_events":len(noise),
        "strict_noise_episodes":len(set(v["episode"] for v in noise)),"unique_noise_pulses":len(grouping),
        "bilateral_same_start_end_pulses":sum(len(v)==2 for v in grouping.values()),
        "long_unresolved_close_examples":[v for v in events if v["direction"]=="O>C" and
           v["start_state_m"]>=.019 and v["outcome"]!="reached" and v["duration_s"]>1][:20]}
    nc=collections.Counter(v["episode"] for v in noise)
    for er in episode_rows:er["strict_gripper_noise_events"]=nc[er["episode"]]
    write_csv(out/"episode审计汇总.csv",episode_rows)
    write_csv(out/"夹爪短脉冲候选.csv",noise)
    # Sampling distribution, persistent plateaus and normalization metadata.
    split=json.loads((root/"stats_view_manifest.json").read_text())
    train=np.isin(ep,split["train_episode_indices"]); valid=~train
    normalization=json.loads((root/"meta/stats.json").read_text())
    report["split"]={"train_episodes":len(split["train_episode_indices"]),"eval_episodes":len(split["eval_episode_indices"]),
        "train_frames":int(train.sum()),"eval_frames":int(valid.sum()),"info_splits":info.get("splits"),
        "train_noise_side_events":sum(v["episode"] in set(split["train_episode_indices"]) for v in noise),
        "eval_noise_side_events":sum(v["episode"] in set(split["eval_episode_indices"]) for v in noise)}
    norm=[]
    for name,x in arrays.items():
        ns=normalization[name]
        for j in range(24):
            q1=float(np.asarray(ns["q01"]).reshape(-1)[j]);q99=float(np.asarray(ns["q99"]).reshape(-1)[j])
            q=x[:,j];tr=q[train];ev=q[valid]
            norm.append(dict(source=name,joint=names[j],train_mean=float(tr.mean()),eval_mean=float(ev.mean()),
                q01=q1,q99=q99,quantile_span=q99-q1,train_min=float(tr.min()),train_max=float(tr.max()),
                eval_outside_train_minmax_fraction=float(((ev<tr.min()-1e-6)|(ev>tr.max()+1e-6)).mean()),
                train_outside_stats_q01q99_fraction=float(((tr<q1-1e-6)|(tr>q99+1e-6)).mean()),
                stats_mean_error=float(abs(tr.mean()-np.asarray(ns["mean"]).reshape(-1)[j]))))
    write_csv(out/"训练验证分布与归一化.csv",norm)
    report["normalization"]={"max_mean_error":max(v["stats_mean_error"] for v in norm),
        "near_zero_quantile_span":[v for v in norm if v["quantile_span"]<1e-5],
        "eval_outside_train_range_over_5pct":[v for v in norm if v["eval_outside_train_minmax_fraction"]>.05]}
    # Lag estimation uses moving arm derivatives, not gripper polarity. Stratified episodes.
    lagrows=[]
    for n in np.linspace(0,len(spans)-1,120,dtype=int):
        b,e=spans[n]
        for side,ids in [("L",range(7)),("R",range(8,15))]:
            av=np.diff(raw[b:e,ids],axis=0).astype(float);sv=np.diff(s[b:e,ids],axis=0).astype(float)
            moving=np.max(abs(av),axis=1)>.001
            if moving.sum()<30:continue
            scores=[]
            for lag in range(-10,16):
                start=max(0,-lag);stop=min(len(av),len(sv)-lag)
                x=av[start:stop];y=sv[start+lag:stop+lag];m=moving[start:stop]
                x=x[m];y=y[m];den=np.linalg.norm(x)*np.linalg.norm(y)
                scores.append(float(np.sum(x*y)/den) if den>0 else 0)
            k=int(np.argmax(scores));lagrows.append(dict(episode=int(ep[b]),side=side,best_lag_frames=k-10,
                best_score=scores[k],zero_lag_score=scores[10],positive_lag_meaning="state follows action"))
    write_csv(out/"手臂动作反馈时移抽样.csv",lagrows)
    report["arm_lag_sample"]={"episode_count":120,"best_lag_frames":stats([v["best_lag_frames"] for v in lagrows]),
        "histogram":dict(collections.Counter(str(v["best_lag_frames"]) for v in lagrows)),
        "strong_matches_count":sum(v["best_score"]>.8 for v in lagrows),"note":"Derivative correlation of raw pre-SG action; not hardware latency measurement"}
    (out/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in ["dynamics","normalization","gripper_ranges"]},ensure_ascii=False,indent=2),flush=True)


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--dataset",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--urdf",type=Path,required=True)
    args=p.parse_args();audit(args.dataset,args.output,args.urdf)
