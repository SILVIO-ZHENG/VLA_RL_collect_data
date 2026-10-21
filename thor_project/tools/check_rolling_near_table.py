"""Offline conservative-envelope audit; tabletop changes exist ONLY in memory."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import time

import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--client-dir',type=Path,required=True)
    p.add_argument('--profile',type=Path,required=True)
    p.add_argument('--pose-log',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--geometry',choices=('bounds','legacy'),default='bounds')
    a=p.parse_args()
    sys.path.insert(0,str(a.client_dir.resolve()))
    from robot_client.safety_layer import load_safety_layer, SurfaceConstraints
    from robot_client.async_safety import check_rolling_window, check_legacy_rolling_window, hold_displacement_coefficients, braking_sample
    from robot_client.constants import GDK_BODY_JOINTS, GDK_BODY_JOINT_LIMITS
    with a.output.open('x') as output:
        layer=load_safety_layer(a.profile,asset_root=a.client_dir/'robot_client').realtime_guard(path_samples=2)
        with a.pose_log.open() as f:
            q=np.asarray(json.loads(next(f))['safe_action'],dtype=np.float32)
        coefficients=hold_displacement_coefficients(layer)
        z=layer._evaluate(q).minimum_z.z
        lower,upper=np.empty(24),np.empty(24)
        for index,name in zip((*range(7),*range(8,15),*range(16,24)),GDK_BODY_JOINTS,strict=True):
            low,high=GDK_BODY_JOINT_LIMITS[name];lower[index]=low+.01;upper[index]=high-.01
        lower[[7,15]],upper[[7,15]]=0,np.float32(.025)
        acceleration=np.full(24,2.);acceleration[[7,15]]=.2;acceleration[16:]=1.
        vmax=np.full(24,.6);vmax[[7,15]]=.05;vmax[16:]=.3
        original=layer.surface
        rows=[]
        for gap_mm in (2.5,5,10,20):
            surface=replace(original,height=z-gap_mm/1000,hard_margin=.002,warning_margin=.005,xy_bounds=None)
            layer.surfaces=SurfaceConstraints((surface,));layer.surface=surface
            for index in (*range(7),*range(8,15)):
                for sign in (-1,1):
                    v=np.zeros(24);v[index]=sign*.05
                    targets=[(q+v*.01*i).astype(np.float32) for i in range(1,11)]
                    spec=dict(acceleration=acceleration,max_velocity=vmax,dt=.01,lower=lower,upper=upper,
                              target_velocities=[v.copy() for _ in targets])
                    start=time.perf_counter()
                    result=(check_legacy_rolling_window(layer,q,q,targets,targets[-1],q)
                            if a.geometry=='legacy' else
                            check_rolling_window(layer,q,q,targets,targets[-1],q,spec,coefficients))
                    check_ms=(time.perf_counter()-start)*1000
                    reserve=dict(anchor=targets[-1],velocity=v,acceleration=acceleration,dt=.01,steps=3)
                    end,_=braking_sample(reserve,3)
                    points=[q+(end-q)*alpha for alpha in np.linspace(0,1,51)]
                    sampled=min(layer._evaluate(point).minimum_z.z for point in points)
                    # Subtract the global displacement bound to the nearest
                    # sample. This covers BETWEEN samples, not just 51 poses.
                    error_bound=float(np.dot(coefficients,np.abs(end-q)))/100
                    lower_z=sampled-error_bound-1e-6
                    in_range=bool(np.all(np.minimum(q,end)>=lower) and np.all(np.maximum(q,end)<=upper))
                    certified_clear=lower_z>surface.height+surface.hard_margin and in_range
                    rows.append(dict(gap_mm=gap_mm,joint=index,speed=float(v[index]),check_ms=check_ms,
                                     rejected=result['decision'].hard_limit_triggered,
                                     refined_continuous_clear=certified_clear,
                                     refined_clearance_mm=(lower_z-surface.height-surface.hard_margin)*1000))
        summary=dict(physical_commands=0,table_modified_on_robot=False,geometry=a.geometry,cases=len(rows),
                     conservative_rejections=sum(r['rejected'] and r['refined_continuous_clear'] for r in rows),
                     rows=rows)
        json.dump(summary,output,indent=2)
        print(json.dumps({k:v for k,v in summary.items() if k!='rows'}))


if __name__=='__main__':main()
