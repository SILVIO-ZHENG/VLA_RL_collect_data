"""Supplemental plots and attribution; no dataset mutations."""
from pathlib import Path
import collections
import csv
import json
import numpy as np
import pyarrow.parquet as pq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R=Path('/home/luoxiaolong/workspace/datasets/logistics_lerobot_20260819_20260915_per_package_v30')
O=Path('outputs/diagnostics/dataset-audit-20260923')
M=json.loads((R/'smoothing_manifest.json').read_text())
t=pq.read_table(R/'data',columns=['episode_index','frame_index','action','observation.state'])
ep=t['episode_index'].combine_chunks().to_numpy();fi=t['frame_index'].combine_chunks().to_numpy()
a=t['action'].combine_chunks().values.to_numpy().reshape(-1,24)
s=t['observation.state'].combine_chunks().values.to_numpy().reshape(-1,24)
rt=pq.read_table(Path(M['source_dataset'])/'data',columns=['action'])
raw=rt['action'].combine_chunks().values.to_numpy().reshape(-1,24)
arms=list(range(7))+list(range(8,15));report={}
for j in [5,13]:
    lo=-1.0123;hi=1.0123
    rb=(raw[:,j]<lo-.001)|(raw[:,j]>hi+.001)
    sb=(a[:,j]<lo-.001)|(a[:,j]>hi+.001)
    edge=np.minimum(abs(raw[:,j]-lo),abs(raw[:,j]-hi))<1e-6
    margin=(a[:,j]<lo+.01)|(a[:,j]>hi-.01)
    report['left_wrist6' if j==5 else 'right_wrist6']={
        'raw_at_exact_urdf_limit_frames':int(edge.sum()),'raw_at_exact_urdf_limit_pct':float(edge.mean()*100),
        'raw_urdf_excess_gt_1mrad_frames':int(rb.sum()),'sg_urdf_excess_gt_1mrad_frames':int(sb.sum()),
        'sg_new_excess_frames':int((sb&~rb).sum()),'sg_excess_episodes':len(np.unique(ep[sb])),
        'deployment_margin_conflict_frames':int(margin.sum()),'deployment_margin_conflict_pct':float(margin.mean()*100)}
# Sudden exact action=feedback replacement: hypothesis, not proof of source fallback.
mode_rows=[]
for side,ids in [('L',range(7)),('R',range(8,15))]:
    eq=np.all(raw[:,ids]==s[:,ids],axis=1)
    moving=np.r_[False,np.max(abs(np.diff(s[:,ids],axis=0)),axis=1)>.001] & np.r_[False,ep[1:]==ep[:-1]]
    enter=np.flatnonzero(eq[1:]&~eq[:-1]&(ep[1:]==ep[:-1]))+1
    suspected=[i for i in enter if moving[i] and np.max(abs(raw[i-1,list(ids)]-s[i-1,list(ids)]))>.05]
    report[side+'_raw_exact_state']={'all_frames':int(eq.sum()),'pct':float(eq.mean()*100),
        'moving_exact_frames':int((eq&moving).sum()),'enter_from_error_gt_0_05_during_motion':len(suspected),
        'affected_episodes':len(np.unique(ep[suspected]))}
    for i in suspected:
        mode_rows.append(dict(episode=int(ep[i]),frame=int(fi[i]),time_s=float(fi[i]/30),side=side,
            previous_max_error_rad=float(np.max(abs(raw[i-1,list(ids)]-s[i-1,list(ids)]))),
            max_target_step_rad=float(np.max(abs(raw[i,list(ids)]-raw[i-1,list(ids)])))))
with (O/'动作突变为反馈候选.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,list(mode_rows[0]));w.writeheader();w.writerows(mode_rows)
# Noise phase relative to cut boundary; state movement and arm movement during pulses.
noise=list(csv.DictReader((O/'夹爪短脉冲候选.csv').open()))
meta=pq.read_table(R/'meta/episodes').to_pylist();lookup={m['episode_index']:m for m in meta}
ph=collections.Counter();moving_pulses=0;still_pulses=0
for row in noise:
    m=lookup[int(row['episode'])];b=m['dataset_from_index']+int(row['frame']);e=m['dataset_from_index']+int(row['end_frame'])
    span=np.max(np.ptp(s[b:e,arms],axis=0))
    moving_pulses+=span>.02;still_pulses+=span<=.001
    if int(row['frame'])<30:ph['first_1s']+=1
    elif int(row['end_frame'])>=m['length']-30:ph['last_1s']+=1
    else:ph['middle']+=1
report['short_pulse_context']={'phase_side_events':dict(ph),'arm_span_over_0_02':int(moving_pulses),'arm_span_under_0_001':int(still_pulses)}
# End censored events must not be counted as failed grasps.
events=list(csv.DictReader((O/'夹爪全部事件.csv').open()))
censored=[v for v in events if v['direction']=='O>C' and float(v['start_state_m'])>=.019 and v['outcome']=='episode_end']
report['censored_close_at_end']={'side_events':len(censored),'episodes':len(set(v['episode'] for v in censored)),
    'duration_lt_0_5s':sum((int(v['end_frame'])-int(v['frame']))/30<.5 for v in censored),
    'no_response_span_lt_0_5mm':sum(float(v['state_span_m'])<.0005 for v in censored)}
# Plots with unit-correct gripper mapping and marked event.
fig,axs=plt.subplots(3,1,figsize=(13,10),constrained_layout=True)
cases=[(2130,12,13.4,14.6),(2807,13,8.35,9.35),(3276,5,0,0.7)]
for ax,(eid,j,t0,t1) in zip(axs,cases):
    sel=(ep==eid)&(fi>=round(t0*30))&(fi<=round(t1*30));xx=fi[sel]/30
    ax.plot(xx,raw[sel,j],'.--',label='Original action',alpha=.8)
    ax.plot(xx,a[sel,j],'.-',label='Training action (SG 11)',linewidth=2)
    ax.plot(xx,s[sel,j],label='Measured state',alpha=.8)
    if j in [5,13]:
        ax.axhline(-1.0123,color='red',linestyle=':',label='Current URDF lower limit')
        ax.axhline(-1.0023,color='orange',linestyle=':',label='URDF + 0.01 rad deployment margin')
    ax.set_title(f'Episode {eid}, '+('right joint 5' if j==12 else ('right joint 6' if j==13 else 'left joint 6')))
    ax.set_xlabel('Episode time (s)');ax.set_ylabel('Position (rad)');ax.grid(alpha=.25);ax.legend(loc='best')
fig.savefig(O/'关节异常与SG过冲.png',dpi=160);plt.close(fig)
fig,axs=plt.subplots(2,1,figsize=(12,7),constrained_layout=True)
for ax,(eid,t0,t1) in zip(axs,[(0,19.7,21.1),(2,9.8,11.0)]):
    sel=(ep==eid)&(fi>=round(t0*30))&(fi<=round(t1*30))
    for side,j,style in [('L',7,'-'),('R',15,'--')]:
        ax.step(fi[sel]/30,(.025-a[sel,j])*1000,where='post',linestyle=style,label=f'{side} target opening')
        ax.plot(fi[sel]/30,s[sel,j]*1000,linestyle=style,label=f'{side} measured opening')
    ax.set_title(f'Episode {eid}');ax.set_xlabel('Episode time (s)');ax.set_ylabel('Physical opening (mm)');ax.grid(alpha=.25);ax.legend()
fig.savefig(O/'夹爪短脉冲复核.png',dpi=160);plt.close(fig)
(O/'followup.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps(report,ensure_ascii=False,indent=2))
