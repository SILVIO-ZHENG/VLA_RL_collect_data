"""Video metadata plus reproducible decoded samples and split provenance audit."""
import csv
import json
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq

ROOT=Path('/home/luoxiaolong/workspace/datasets/logistics_lerobot_20260819_20260915_per_package_v30')
OUT=Path('outputs/diagnostics/dataset-audit-20260923')
OUT.mkdir(parents=True,exist_ok=True)
parent=ROOT.parent
meta=pq.read_table(ROOT/'meta/episodes').to_pylist()
rawroot=parent/'logistics_lerobot_no_drop_20260819_20260915_per_package_residual_clean_v30'
rm=json.loads((rawroot/'residual_episode_filter_manifest.json').read_text())
kept=[i for i in range(rm['source_total_episodes']) if i not in set(rm['removed_episode_indices'])]
# First deletion is recorded in DATASET_HANDOFF; verify via every episode length and sampled states.
curated_kept=[i for i in range(3786) if i not in {2,668,674}]
manifest=json.loads((parent/'logistics_lerobot_no_drop_20260819_20260915_per_package_v30/conversion_manifest.json').read_text())
source=manifest['episodes']
mapping=[]
for i,k in enumerate(kept):
    src=source[curated_kept[k]]
    assert src['output_frames']==meta[i]['length'],(i,src['output_frames'],meta[i]['length'])
    mapping.append(dict(episode=i,original_split_episode=curated_kept[k],source_episode=src['source_episode_index'],
        source_start_frame=src['source_frame_start'],source_end_frame_exclusive=src['source_frame_end_exclusive']))
merged=parent/'logistics_lerobot_no_drop_20260819_20260915_merged_v30'
mt=pq.read_table(merged/'data',columns=['observation.state','episode_index','frame_index'])
ms=mt['observation.state'].combine_chunks().values.to_numpy().reshape(-1,24)
mm=pq.read_table(merged/'meta/episodes').to_pylist()
t=pq.read_table(ROOT/'data',columns=['observation.state'])
ss=t['observation.state'].combine_chunks().values.to_numpy().reshape(-1,24)
verified=0
for m in mapping:
    target=meta[m['episode']];offset=mm[m['source_episode']]['dataset_from_index']+m['source_start_frame']
    original=ms[offset:offset+target['length']]
    current=ss[target['dataset_from_index']:target['dataset_to_index']]
    assert np.array_equal(original,current),m
    verified+=1
train={m['source_episode'] for m in mapping if m['episode']<3405}
valid={m['source_episode'] for m in mapping if m['episode']>=3405}
shared=sorted(train&valid)
provenance={'length_and_full_state_verified_episodes':verified,'shared_original_train_eval_episodes':shared,
    'shared_source_mappings':[m for m in mapping if m['source_episode'] in shared],
    'limitation':'Original raw capture timestamps/source-session paths are absent from final dataset; upstream conversion may remove idle intervals.'}
with (OUT/'episode原始来源映射.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,list(mapping[0]));w.writeheader();w.writerows(mapping)
(OUT/'provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2))
print('Provenance',json.dumps(provenance,ensure_ascii=False),flush=True)
videos=[]
for p in sorted((ROOT/'videos').glob('*/*/*.mp4')):
    with av.open(str(p)) as c:
        st=c.streams.video[0]
        videos.append(dict(path=str(p.relative_to(ROOT)),frames=st.frames,width=st.width,height=st.height,
            fps=float(st.average_rate),duration_s=float(st.duration*st.time_base),codec=st.codec_context.name))
errors=[]
byfile={v['path']:v for v in videos}
for m in meta:
    for cam in ['cam_head','cam_left_wrist','cam_right_wrist']:
        pre='videos/observation.images.'+cam+'/'
        path=f"videos/observation.images.{cam}/chunk-{m[pre+'chunk_index']:03d}/file-{m[pre+'file_index']:03d}.mp4"
        start,end=m[pre+'from_timestamp'],m[pre+'to_timestamp']
        if abs((end-start)*30-m['length'])>.01 or start<0 or end>byfile[path]['duration_s']+.034:
            errors.append({'episode':m['episode_index'],'camera':cam,'path':path})
# Sample 3s windows, two per episode at ~1/4 and ~3/4, all cameras. Not full decoding.
samples=[]
sample_eps=[0,2,143,602,812,1084,1150,1359,2088,2130,2688,3200,3404,3405,3584]
for eid in sample_eps:
    m=meta[eid]
    for frac in [.25,.75]:
        offset=max(0,min(m['length']/30-3,frac*m['length']/30))
        for cam in ['cam_head','cam_left_wrist','cam_right_wrist']:
            pre='videos/observation.images.'+cam+'/'
            p=ROOT/f"videos/observation.images.{cam}/chunk-{m[pre+'chunk_index']:03d}/file-{m[pre+'file_index']:03d}.mp4"
            start=m[pre+'from_timestamp']+offset
            times=[]; means=[];delta=[];exact=[];prev=None
            try:
                with av.open(str(p)) as c:
                    st=c.streams.video[0];c.seek(int(start/st.time_base),stream=st,backward=True)
                    for frame in c.decode(video=0):
                        tm=float(frame.pts*frame.time_base)
                        if tm<start-1/60:continue
                        if tm>=start+3-1/60:break
                        img=frame.to_ndarray(format='gray');means.append(float(img.mean()));times.append(tm)
                        if prev is not None:
                            exact.append(bool(np.array_equal(img,prev)))
                            delta.append(float(np.mean(abs(img.astype(float)-prev))))
                        prev=img
                samples.append(dict(episode=eid,camera=cam,start_s=offset,frames=len(times),
                    gap_gt_1_5_frames=int(np.sum(np.diff(times)>.05)),repeated_exact_frames=sum(exact),
                    dark_frames=sum(v<5 for v in means),mean_frame_difference=float(np.mean(delta)),error=None))
            except Exception as exc:samples.append(dict(episode=eid,camera=cam,start_s=offset,error=str(exc)))
    print('Decoded sample episode',eid,flush=True)
report={'videos':videos,'metadata_interval_errors':errors,'decoded_samples':samples,
    'scope':'48 MP4 headers/episode intervals; 15 episodes x 2 three-second windows x 3 cameras; not a source-timestamp synchronization proof'}
(OUT/'media.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print('Media',len(videos),'samples',len(samples),'frames',sum(v.get('frames',0) for v in samples),
    'errors',sum(v.get('error') is not None for v in samples),'metadata_errors',len(errors),flush=True)
