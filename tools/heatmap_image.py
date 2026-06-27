#!/usr/bin/env python3
# heatmap_image.py IMG MODEL [MPP=0.5] [SCALES_m=48,64,80] [STEP_M=6] — HEATMAP dens multi-scară pe o
# IMAGINE hillshade dată (nu tile geo). Replică pipeline-ul de stampă (fereastra metri-teren -> stretch
# percentile -> downsample la 2m efectiv -> 128 -> homog -> scor), ia MAX peste scări. -> review/heatmap_image.png
import sys,os,math
import numpy as np
from PIL import Image,ImageFilter,ImageDraw,ImageFont
import torch,torch.nn as nn
H=os.path.expanduser('~/lidar-match');dev=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
IMG=sys.argv[1];MODEL=sys.argv[2]
MPP=float(sys.argv[3]) if len(sys.argv)>3 else 0.5
SCALES=[float(x) for x in (sys.argv[4].split(',') if len(sys.argv)>4 else "48,64,80".split(','))]
STEP_M=float(sys.argv[5]) if len(sys.argv)>5 else 6.0
im=Image.open(IMG).convert('L');A=np.asarray(im,np.float32);Hh,Ww=A.shape
print(f"imagine {Ww}x{Hh}px @ {MPP}m/px (~{Ww*MPP:.0f}m teren) | scări {SCALES}m | model {os.path.basename(MODEL)}",flush=True)
def homog(a):
    a2=np.asarray(Image.fromarray(a).filter(ImageFilter.GaussianBlur(0.8)),np.uint8);cdf=np.bincount(a2.ravel(),minlength=256).astype(np.float64).cumsum()
    return (cdf[a2]/cdf[-1]*255).astype(np.uint8) if cdf[-1]>0 else a2
def stamp(cx,cy,win,content):
    h=win//2;w=A[cy-h:cy+h,cx-h:cx+h]
    if w.shape!=(2*h,2*h) or w.std()<3: return None
    lo,hi=np.percentile(w,2),np.percentile(w,98);a=np.clip((w-lo)/(hi-lo+1e-6),0,1)
    a2=np.asarray(Image.fromarray((a*255).astype('uint8')).resize((content,content)),np.uint8)  # ~2m efectiv
    return homog(np.asarray(Image.fromarray(a2).resize((128,128)),np.uint8))
class Net(nn.Module):
    def __init__(s):
        super().__init__();s.c=nn.Sequential(nn.Conv2d(1,16,3,2,1),nn.ReLU(),nn.Conv2d(16,32,3,2,1),nn.ReLU(),nn.Conv2d(32,64,3,2,1),nn.ReLU(),nn.AdaptiveAvgPool2d(1));s.f=nn.Linear(64,1)
    def forward(s,x): return s.f(s.c(x).flatten(1)).squeeze(1)
net=Net().to(dev);net.load_state_dict(torch.load(MODEL,map_location=dev,weights_only=True));net.eval()
step=max(2,int(STEP_M/MPP))
gxs=list(range(0,Ww,step));gys=list(range(0,Hh,step))
best=np.zeros((len(gys),len(gxs)),np.float32)
for M in SCALES:
    win=int(round(M/MPP));content=max(8,int(round(M/2.0)));h=win//2
    batch=[];pos=[]
    for iy,cy in enumerate(gys):
        for ix,cx in enumerate(gxs):
            if cx-h<0 or cy-h<0 or cx+h>Ww or cy+h>Hh: continue
            s=stamp(cx,cy,win,content)
            if s is not None: batch.append(s);pos.append((iy,ix))
    if not batch: print(f"  scară {M:.0f}m (win {win}px): fără stampe");continue
    X=torch.tensor(np.array(batch,dtype=np.uint8));sc=[]
    with torch.no_grad():
        for k in range(0,len(X),512): sc.extend(torch.sigmoid(net(X[k:k+512].unsqueeze(1).float().to(dev)/255.)).cpu().numpy().tolist())
    g=np.zeros_like(best)
    for (iy,ix),v in zip(pos,sc): g[iy,ix]=v
    best=np.maximum(best,g)
    print(f"  scară {M:.0f}m (win {win}px): mediană {np.median(sc):.3f} | %>=0.7 {(np.array(sc)>=0.7).mean()*100:.1f}% | %>=0.9 {(np.array(sc)>=0.9).mean()*100:.1f}%",flush=True)
# NOTĂ: filtrul de coerență NU se aplică aici — pe hillshade-ul randat (imagine) coerența gradientului taie și
# movilele (Balloo 10.4→6.0% vs Groningen 14.5→10.9%). Coerența trebuie calculată pe DEM, în detectoarele geo
# (rescore_eval/ran_pass/scan5 au DEM-ul). Vezi tuning 24.06: DEM coh22>0.70 e cel validat.
v=best[best>0]
print(f"COMBINAT (max pe scări): mediană {np.median(v):.3f} | %>=0.7 {(v>=0.7).mean()*100:.1f}% | %>=0.9 {(v>=0.9).mean()*100:.1f}%",flush=True)
def jet(s):
    r=np.clip(1.5-np.abs(4*s-3),0,1);g=np.clip(1.5-np.abs(4*s-2),0,1);b=np.clip(1.5-np.abs(4*s-1),0,1)
    return np.stack([r,g,b],-1)
field=np.asarray(Image.fromarray((best*255).astype('uint8')).resize((Ww,Hh),Image.BICUBIC),np.float32)/255.
bg=np.clip((A-np.percentile(A,2))/(np.percentile(A,98)-np.percentile(A,2)+1e-6),0,1)
rgb=(np.stack([bg,bg,bg],-1)*255).astype(np.float32)
col=jet(field)*255;alpha=(np.clip((field-0.5)/0.5,0,1)*0.75)[...,None]   # arată doar >=0.5, ca heatmap-ul de referință
out=(rgb*(1-alpha)+col*alpha).astype(np.uint8)
os.makedirs(f"{H}/review",exist_ok=True)
Image.fromarray(out).save(f"{H}/review/heatmap_image.png")
print(f"-> {H}/review/heatmap_image.png")
