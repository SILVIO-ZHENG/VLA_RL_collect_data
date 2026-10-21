"""Resolve censored gripper events against the original merged episodes."""
from pathlib import Path
import json
import csv
import collections
import numpy as np
import pyarrow.parquet as pq

P=Path('outputs/diagnostics/dataset-audit-20260923')
R=Path('/home/luoxiaolong/workspace/datasets/logistics_lerobot_no_drop_20260819_20260915_merged_v30')
mappings={int(x['episode']):x for x in csv.DictReader((P/'episode原始来源映射.csv').open())}
events=list(csv.DictReader((P/'夹爪全部事件.csv').open()))
censored=[v for v in events if v['direction']=='O>C' and float(v['start_state_m'])>=.019 and v['outcome']=='episode_end']
t=pq.read_table(R/'data',columns=['action','observation.state'])
a=t['action'].combine_chunks().values.to_numpy().reshape(-1,24)
s=t['observation.state'].combine_chunks().values.to_numpy().reshape(-1,24)
meta=pq.read_table(R/'meta/episodes').to_pylist()
rows=[]
for v in censored:
    eid=int(v['episode']);m=mappings[eid];orig=meta[int(m['source_episode'])]
    j=7 if v['side']=='L' else 15
    b=orig['dataset_from_index']+int(m['source_start_frame'])+int(v['frame'])
    boundary=orig['dataset_from_index']+int(m['source_end_frame_exclusive'])
    end=orig['dataset_to_index'];finish=end;outcome='original_episode_end'
    for i in range(b,end):
        # A one-frame feedback glitch is not a completed close.
        if i+1<end and s[i,j]<=.006+1e-9 and s[i+1,j]<=.006+1e-9:
            finish=i;outcome='reached';break
        if .025-a[i,j]>=.019-1e-9:finish=i;outcome='reversed';break
    span=float(np.ptp(s[b:max(b+1,finish),j]));duration=(finish-b)/30
    rows.append(dict(episode=eid,side=v['side'],frame=int(v['frame']),time_s=float(v['time_s']),
        source_episode=int(m['source_episode']),source_event_start_frame=b-orig['dataset_from_index'],
        outcome_in_source=outcome,total_duration_s=duration,state_span_m=span,
        resolved_after_boundary_frames=finish-boundary,short_no_response=outcome=='reversed' and duration<.5 and span<.0005))
summary={'censored_events':len(rows),'source_outcomes':dict(collections.Counter(r['outcome_in_source'] for r in rows)),
    'additional_strict_short_no_response_side_events':sum(r['short_no_response'] for r in rows),
    'additional_no_response_reversals_any_duration':sum(r['outcome_in_source']=='reversed' and r['state_span_m']<.0005 for r in rows),
    'examples':rows[:12]}
with (P/'跨切分边界夹爪复核.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
(P/'boundaries.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
print(json.dumps(summary,ensure_ascii=False,indent=2))
