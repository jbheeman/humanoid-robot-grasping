#!/usr/bin/env python3
"""Browser-based, no-actuation D435I localization collector."""

from __future__ import annotations

import argparse
import base64
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
import time

import cv2
import numpy as np
import pyrealsense2 as rs


HTML = """<!doctype html><meta name=viewport content='width=device-width,initial-scale=1'>
<title>G1 localization tuner</title><style>body{font:16px sans-serif;max-width:1000px;margin:auto;background:#111;color:#eee}#stage{position:relative;width:100%;max-width:960px}#f{display:block;width:100%;cursor:crosshair;user-select:none;-webkit-user-drag:none}#overlay{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}input{width:7em;margin:3px}button{padding:8px;margin:4px}#msg{white-space:pre-wrap}</style>
<h2>D435I localization tuner (no actuation)</h2><p>Click the plushie's <b>top-left</b> corner, then its <b>bottom-right</b> corner. The box will be drawn automatically. Measure from the center of the waist yaw joint (robot torso/pelvis center), not the chest shell. X=forward, Y=robot-left, Z=up; enter metres.</p>
<div id=stage><img id=f draggable=false src=/frame.jpg><canvas id=overlay></canvas></div><div><label>name <input id=n value=pose1></label><label>torso X (forward) <input id=x type=number step=.001></label><label>torso Y (left +) <input id=y type=number step=.001></label><label>torso Z (up) <input id=z type=number step=.001></label><button onclick=save()>Save sample</button><button onclick=resetBox()>Reset box</button></div><div id=msg>Click top-left, then bottom-right.</div>
<script>const W=960,H=540;let pts=[],box=null;const stage=document.getElementById('stage'),f=document.getElementById('f'),ov=document.getElementById('overlay'),ctx=ov.getContext('2d'),msg=document.getElementById('msg');function xy(e){let r=f.getBoundingClientRect();return [Math.max(0,Math.min(W,(e.clientX-r.left)*W/r.width)),Math.max(0,Math.min(H,(e.clientY-r.top)*H/r.height))]}function draw(){ov.width=f.clientWidth;ov.height=f.clientHeight;ctx.clearRect(0,0,ov.width,ov.height);if(!box)return;let sx=ov.width/W,sy=ov.height/H;ctx.strokeStyle='#00ff66';ctx.lineWidth=3;ctx.strokeRect(box[0]*sx,box[1]*sy,(box[2]-box[0])*sx,(box[3]-box[1])*sy)}f.addEventListener('click',e=>{e.preventDefault();if(pts.length>=2)pts=[];pts.push(xy(e));if(pts.length===1){msg.textContent='Now click the plushie bottom-right corner.';box=null;draw()}else{let[a,b]=pts;box=[Math.min(a[0],b[0]),Math.min(a[1],b[1]),Math.max(a[0],b[0]),Math.max(a[1],b[1])];msg.textContent='Box selected: '+box.map(v=>v.toFixed(1)).join(', ');draw()}});window.addEventListener('resize',draw);function resetBox(){pts=[];box=null;msg.textContent='Click top-left, then bottom-right.';draw()}setInterval(()=>{if(pts.length!==1)f.src='/frame.jpg?t='+Date.now()},250);async function save(){if(!box)return msg.textContent='Select two corners first';let r=await fetch('/sample',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({name:n.value,bbox_xyxy:box,torso_truth_m:[+x.value,+y.value,+z.value]})});msg.textContent=await r.text();if(r.ok){n.value='pose'+(Date.now()%100000);resetBox()}}f.addEventListener('load',draw);</script>"""


class Collector:
    def __init__(self, dataset: Path, serial: str) -> None:
        self.dataset, self.serial = dataset, serial
        self.lock = threading.Lock(); self.rgb = None; self.z16 = None; self.intr = None
        self.pipeline = rs.pipeline(); cfg = rs.config(); cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.color, 960, 540, rs.format.rgb8, 60)
        cfg.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 60)
        profile = self.pipeline.start(cfg); self.align = rs.align(rs.stream.color)
        self.device = profile.get_device(); self.scale = float(self.device.first_depth_sensor().get_depth_scale())
        self.intr = self.align.process(self.pipeline.wait_for_frames(1000)).get_color_frame().profile.as_video_stream_profile().get_intrinsics()
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self) -> None:
        while True:
            frames = self.align.process(self.pipeline.wait_for_frames(1000)); c,d=frames.get_color_frame(),frames.get_depth_frame()
            if c and d:
                with self.lock: self.rgb=np.asanyarray(c.get_data()).copy(); self.z16=np.asanyarray(d.get_data()).astype('<u2',copy=True)

    def frame_jpeg(self) -> bytes:
        with self.lock: rgb=None if self.rgb is None else self.rgb.copy()
        if rgb is None: return b''
        ok,data=cv2.imencode('.jpg',cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,85]); return data.tobytes() if ok else b''

    def save(self, value: dict) -> dict:
        name=str(value['name']).strip(); x,y,x2,y2=[int(v) for v in value['bbox_xyxy']]
        torso=[float(v) for v in value['torso_truth_m']]
        with self.lock: rgb=self.rgb.copy(); z=self.z16.copy()
        left,right=max(0,x),min(960,x2); top,bottom=max(0,y),min(540,y2); cx,cy=(left+right)//2,(top+bottom)//2
        l,r=max(0,int(cx-(right-left)*.3)),min(960,int(cx+(right-left)*.3)); t,b=max(0,int(cy-(bottom-top)*.3)),min(540,int(cy+(bottom-top)*.3)); vals=z[t:b,l:r].astype(float)*self.scale; vals=vals[(vals>=.12)&(vals<=4)]
        if len(vals)<8: raise ValueError('not enough valid depth in box')
        data=json.loads(self.dataset.read_text()) if self.dataset.exists() else {'samples':[]}; data.setdefault('samples',[])
        if any(s.get('name')==name for s in data['samples']): raise ValueError('duplicate sample name')
        out=self.dataset.parent/(self.dataset.stem+'_frames'); out.mkdir(parents=True,exist_ok=True); np.save(out/f'{name}_rgb.npy',rgb); np.save(out/f'{name}_aligned_depth_z16.npy',z)
        data.update({'camera_serial':self.serial,'rgb_profile':{'width':960,'height':540,'fps':60,'format':'rgb8'},'depth_profile':{'width':960,'height':540,'fps':60,'format':'z16','aligned_to_rgb':True},'depth_scale':self.scale,'rgb_intrinsics':{'width':960,'height':540,'fx':self.intr.fx,'fy':self.intr.fy,'ppx':self.intr.ppx,'ppy':self.intr.ppy,'distortion_model':str(self.intr.model).split('.')[-1].lower(),'coefficients':list(self.intr.coeffs)}})
        data['samples'].append({'name':name,'bbox_xyxy':[x,y,x2,y2],'torso_truth_m':torso,'median_aligned_depth_m':float(np.median(vals)),'depth_valid_fraction':float(len(vals)/max(1,(r-l)*(b-t))),'rgb_file':str(out/f'{name}_rgb.npy'),'aligned_depth_file':str(out/f'{name}_aligned_depth_z16.npy'),'captured_unix_s':time.time()}); self.dataset.parent.mkdir(parents=True,exist_ok=True); self.dataset.write_text(json.dumps(data,indent=2)+'\n'); return {'ok':True,'name':name,'median_depth_m':float(np.median(vals))}


def main() -> int:
    p=argparse.ArgumentParser(); p.add_argument('dataset',type=Path); p.add_argument('--serial',default='254322072511'); p.add_argument('--host',default='0.0.0.0'); p.add_argument('--port',type=int,default=8001); a=p.parse_args(); c=Collector(a.dataset,a.serial)
    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.startswith('/frame.jpg'): body,typ=c.frame_jpeg(),'image/jpeg'
            else: body,typ=HTML.encode(),'text/html; charset=utf-8'
            self.send_response(200); self.send_header('Content-Type',typ); self.send_header('Content-Length',str(len(body))); self.end_headers(); self.wfile.write(body)
        def do_POST(self):
            try: result=c.save(json.loads(self.rfile.read(int(self.headers.get('Content-Length','0'))))); body=json.dumps(result).encode(); code=200
            except Exception as e: body=json.dumps({'ok':False,'error':str(e)}).encode(); code=400
            self.send_response(code); self.send_header('Content-Type','application/json'); self.send_header('Content-Length',str(len(body))); self.end_headers(); self.wfile.write(body)
        def log_message(self,*args): pass
    print(f'Open http://<robot-ip>:{a.port}/',flush=True); ThreadingHTTPServer((a.host,a.port),H).serve_forever(); return 0

if __name__=='__main__': raise SystemExit(main())
