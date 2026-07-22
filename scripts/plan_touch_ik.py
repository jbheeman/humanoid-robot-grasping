#!/usr/bin/env python3
"""Plan one collision-checked tucked-to-intercept trajectory outside Isaac."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from object_tracking.arm_tracking.ik_solver import G1RightArmIK

p = argparse.ArgumentParser()
p.add_argument("--urdf", type=Path, required=True)
p.add_argument("--start-q", required=True)
p.add_argument("--tuck-base", required=True)
p.add_argument("--intercept-base", required=True)
p.add_argument("--ee-frame", default="right_hand_palm_link")
p.add_argument("--extension-frames", type=int, default=25)
p.add_argument("--hold-frames", type=int, default=18)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()

def vector(text):
    return np.asarray([float(x) for x in text.split(",")], dtype=np.float64)

def solve_line(ik, q, start, end, steps):
    values=[]
    rotation=ik.forward_kinematics(q)[:3,:3].copy()
    for xyz in np.linspace(start, end, steps+1)[1:]:
        target=np.eye(4); target[:3,:3]=rotation; target[:3,3]=xyz
        result=ik.solve(target,q)
        if not result.ok:
            raise SystemExit(f"ik:{result.reason}")
        q=np.asarray(result.q_rad,dtype=np.float64); values.append(q.copy())
    return values,q

if a.extension_frames < 12 or a.hold_frames < 15:
    raise SystemExit("extension-frames must be >=12 and hold-frames >=15")
ik=G1RightArmIK(a.urdf, orientation_weight=0.0, position_tolerance_m=0.012,
                discontinuity_limit_rad=1.20, ee_frame_name=a.ee_frame)
q=vector(a.start_q); tuck=vector(a.tuck_base); intercept=vector(a.intercept_base)
start=ik.forward_kinematics(q)[:3,3]
tuck_path,q=solve_line(ik,q,start,tuck,18)
tuck_q=q.copy()
actual_tuck=ik.forward_kinematics(tuck_q)[:3,3]
extension_m=float(np.linalg.norm(intercept-actual_tuck))
if not 0.15 <= extension_m <= 0.32:
    raise SystemExit(f"extension_distance:{extension_m:.4f}")
path,q=solve_line(ik,tuck_q,actual_tuck,intercept,a.extension_frames)
path.extend([q.copy()]*a.hold_frames)
path=np.asarray(path,dtype=np.float32)
ee=np.asarray([ik.forward_kinematics(x)[:3,3] for x in path],dtype=np.float32)
a.output.parent.mkdir(parents=True,exist_ok=True)
np.savez_compressed(a.output,q=path,ee_xyz=ee,tuck_q=tuck_q.astype(np.float32),
                    tuck_xyz=actual_tuck.astype(np.float32),extension_m=np.float32(extension_m))
