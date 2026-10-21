"""Audit gripper feedback spikes and render one evidence example."""
from pathlib import Path
import json,csv
import numpy as np
import pyarrow.parquet as pq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R=Path('/home/luoxiaolong/workspace/datasets/logistics_lerobot_20260819_20260915_per_package_v30')
P=Path('outputs/diagnostics/dataset-audit-20260923')
t=pq.read_table(R/'data',columns=['action','observation.state','episode_index','frame_index'])
a=t['action'].combine_chunks().values.to_numpy().reshape(-1,24)
s=t['observation.state'].combine_chunks().values.to_numpy().reshape(-1,24)
ep=t['episode_index'].combine_chunks().to_numpy();fi=t['frame_index'].combine_chunks().to_numpy()
mask1=ep[1:]==ep[:-1];mask2=ep[2:]==ep[:-2]
report={};rows=[];all_eps=set()
for side,j in [('L',7),('R',15)]:
    q=s[:,j];d=np.diff(q)
    jumps=np.flatnonzero((abs(d)>.01)&mask1)+1
    spike=(abs(q[1:-1]-q[:-2])>.01)&(abs(q[1:-1]-q[2:])>.01)&(abs(q[2:]-q[:-2])<.0005)&mask2
    ids=np.flatnonzero(spike)+1
    report[side]={'jump_over_10mm_intervals':len(jumps),'jump_episodes':len(np.unique(ep[jumps])),
        'isolated_return_spikes':len(ids),'spike_episodes':len(np.unique(ep[ids])),
        'near_0_025mm_spikes':int((abs(q[ids]-.000025)<1e-7).sum())}
    all_eps.update(map(int,ep[ids]))
    for i in ids:rows.append(dict(episode=int(ep[i]),frame=int(fi[i]),time_s=float(fi[i]/30),side=side,
        prev_state_mm=float(q[i-1]*1000),spike_state_mm=float(q[i]*1000),next_state_mm=float(q[i+1]*1000),
        prev_target_opening_mm=float((.025-a[i-1,j])*1000),target_opening_mm=float((.025-a[i,j])*1000),next_target_opening_mm=float((.025-a[i+1,j])*1000)))
report['union_spike_episodes']=len(all_eps)
with (P/'夹爪反馈单帧尖峰.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
(P/'state_spikes.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
fig,ax=plt.subplots(figsize=(11,4),constrained_layout=True)
sel=(ep==0)&(fi>=193)&(fi<=217)
ax.step(fi[sel]/30,(.025-a[sel,15])*1000,where='post',label='Right target physical opening')
ax.plot(fi[sel]/30,s[sel,15]*1000,'.-',label='Right measured opening')
ax.set(title='Episode 0: single-frame gripper feedback spike',xlabel='Episode time (s)',ylabel='Opening (mm)');ax.legend();ax.grid(alpha=.25)
fig.savefig(P/'夹爪反馈单帧尖峰.png',dpi=160)
print(json.dumps(report,ensure_ascii=False,indent=2))
