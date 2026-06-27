#!/usr/bin/env python3
# verify_moldova.py [N] — VERIFICARE riguroasă: scorează N tumuli RAN Moldova (din /tmp/moldova_tumuli.json)
# pe stratul național z16, rețeta ANTRENATĂ (homog), cu PEAK-SEARCH ±60m (coord RAN imprecise) +
# multi-window. Control = puncte random >250m, TOT peak-searched (= maxime locale = hard-neg). Onest.
import math,subprocess,os,random,json
import numpy as np
from PIL import Image,ImageFilter
import torch,torch.nn as nn
random.seed(11)
H=os.path.expanduser('~/lidar-match');dev=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
N=int(os.sys.argv[1]) if len(os.sys.argv)>1 else 120
R=6378137.0;ORIG=-20037508.342787;ORIGY=20037508.342787;Z=16;MPP=2*math.pi*R/(256*2**Z)
ORG="wCvLzGFkz06gCfBg";svc="1m";TDIR="/tmp/nat_tiles";os.makedirs(TDIR,exist_ok=True)
def merc(lo,la): return R*math.radians(lo), R*math.log(math.tan(math.pi/4+math.radians(la)/2))
def tile(col,row):
    fn=f"{TDIR}/{svc}_{Z}_{col}_{row}.png"
    if not os.path.exists(fn): subprocess.run(["curl","-s","--max-time","20","-o",fn,f"https://tiles.arcgis.com/tiles/{ORG}/arcgis/rest/services/{svc}/MapServer/tile/{Z}/{row}/{col}"],check=False)
    try: return Image.open(fn).convert('L')
    except: return None
class Net(nn.Module):
    def __init__(s):
        super().__init__();s.c=nn.Sequential(nn.Conv2d(1,16,3,2,1),nn.ReLU(),nn.Conv2d(16,32,3,2,1),nn.ReLU(),nn.Conv2d(32,64,3,2,1),nn.ReLU(),nn.AdaptiveAvgPool2d(1));s.f=nn.Linear(64,1)
    def forward(s,x): return s.f(s.c(x).flatten(1)).squeeze(1)
net=Net().to(dev);net.load_state_dict(torch.load(f'{H}/combined_cnn.pt',map_location=dev,weights_only=True));net.eval()
def homog(a):
    a2=np.asarray(Image.fromarray(a).filter(ImageFilter.GaussianBlur(0.8)),np.uint8);cdf=np.bincount(a2.ravel(),minlength=256).astype(np.float64).cumsum()
    return (cdf[a2]/cdf[-1]*255).astype(np.uint8) if cdf[-1]>0 else a2
def fetch(clon,clat,km):
    cx,cy=merc(clon,clat);half=km*1000/2/MPP
    x0=(cx-ORIG)/MPP-half;y0=(ORIGY-cy)/MPP-half;W=int(2*half);mos=Image.new('L',(W,W),0)
    for col in range(int(x0//256),int((x0+W)//256)+1):
        for row in range(int(y0//256),int((y0+W)//256)+1):
            t=tile(col,row)
            if t: mos.paste(t,(col*256-int(x0),row*256-int(y0)))
    return np.asarray(mos,np.float32),x0,y0
def stamp(mosA,px,py,clat,wm):
    MPPg=MPP*math.cos(math.radians(clat));hw=int(wm/2/MPPg);w=mosA[py-hw:py+hw,px-hw:px+hw]
    if w.shape!=(2*hw,2*hw) or w.std()<0.3: return None
    lo,hi=np.percentile(w,2),np.percentile(w,98);a=np.clip((w-lo)/(hi-lo+1e-6),0,1)
    f=max(1,int(round(2.0/MPPg)));Hh,Ww=a.shape;a=a[:Hh//f*f,:Ww//f*f].reshape(Hh//f,f,Ww//f,f).mean((1,3))
    return homog(np.asarray(Image.fromarray((a*255).astype('uint8')).resize((128,128)),np.uint8))
def peakscore(mosA,px,py,clat):
    MPPg=MPP*math.cos(math.radians(clat));st=int(30/MPPg);rng=int(60/MPPg)
    S=[]
    for dy in range(-rng,rng+1,st):
        for dx in range(-rng,rng+1,st):
            for wm in (48,64):
                s=stamp(mosA,px+dx,py+dy,clat,wm)
                if s is not None: S.append(s)
    if not S: return None
    X=torch.tensor(np.array(S,dtype=np.uint8))
    with torch.no_grad(): v=torch.sigmoid(net(X.unsqueeze(1).float().to(dev)/255.)).cpu().numpy()
    return float(v.max())
def ll_to_px(lo,la,x0,y0):
    x,y=merc(lo,la);return (x-ORIG)/MPP-x0,(ORIGY-y)/MPP-y0
T=json.load(open('/tmp/moldova_tumuli.json'));random.shuffle(T);T=T[:N]
pos=[];neg=[];done=0
for t in T:
    mosA,x0,y0=fetch(t['lon'],t['lat'],1.6)
    if (mosA>0).mean()<0.3: continue
    px,py=ll_to_px(t['lon'],t['lat'],x0,y0);ps=peakscore(mosA,int(px),int(py),t['lat'])
    if ps is None: continue
    pos.append(ps);done+=1
    # 2 hard controls: random >250m from center, peak-searched
    W=mosA.shape[0];cnt=0;tries=0
    while cnt<2 and tries<40:
        tries+=1;qx=random.randint(80,W-80);qy=random.randint(80,W-80)
        if (qx-px)**2+(qy-py)**2 < (250/(MPP*math.cos(math.radians(t['lat']))))**2: continue
        ns=peakscore(mosA,qx,qy,t['lat'])
        if ns is not None: neg.append(ns);cnt+=1
def pct(a,thr): return 100*np.mean(np.array(a)>=thr)
def auroc(p,n):
    c=sum((a>b)+0.5*(a==b) for a in p for b in n);return c/(len(p)*len(n))
pos=np.array(pos);neg=np.array(neg)
print(f"N tumuli scorați: {len(pos)} | controale hard: {len(neg)}")
print(f"TUMULI  : median {np.median(pos):.3f} | recall@0.3 {pct(pos,0.3):.0f}% @0.5 {pct(pos,0.5):.0f}% @0.7 {pct(pos,0.7):.0f}%")
print(f"CONTROL : median {np.median(neg):.3f} | %@0.3 {pct(neg,0.3):.0f}% @0.5 {pct(neg,0.5):.0f}% @0.7 {pct(neg,0.7):.0f}%")
print(f"AUROC (tumul vs hard-control peak-searched): {auroc(pos,neg):.3f}")
