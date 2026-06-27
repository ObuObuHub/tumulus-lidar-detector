#!/usr/bin/env python3
# test_moldova_tumuli.py — scorează tumuli RAN CONFIRMAȚI din Moldova (Botoșani, au coord) pe stratul
# național z16, sub 2 rețete: CLEAN (norm globală, fără homog) vs HOMOG (per-window 2-98 + homog).
# pozitivi vs control random -> vede dacă RECALL generalizează „la înălțime" pe 5m curat.
import math,subprocess,os,random
import numpy as np
from PIL import Image,ImageFilter
import torch,torch.nn as nn
random.seed(7)
H=os.path.dirname(os.path.dirname(os.path.abspath(__file__)));dev=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
R=6378137.0;ORIG=-20037508.342787;ORIGY=20037508.342787;Z=16;MPP=2*math.pi*R/(256*2**Z)
ORG="wCvLzGFkz06gCfBg";svc="1m";TDIR="/tmp/nat_tiles";os.makedirs(TDIR,exist_ok=True)
def merc(lo,la): return R*math.radians(lo), R*math.log(math.tan(math.pi/4+math.radians(la)/2))
def tile(col,row):
    fn=f"{TDIR}/{svc}_{Z}_{col}_{row}.png"
    if not os.path.exists(fn): subprocess.run(["curl","-s","--max-time","25","-o",fn,f"https://tiles.arcgis.com/tiles/{ORG}/arcgis/rest/services/{svc}/MapServer/tile/{Z}/{row}/{col}"],check=False)
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
def fetch_mosaic(clon,clat,km):
    cx,cy=merc(clon,clat);half=km*1000/2/MPP
    x0=(cx-ORIG)/MPP-half;y0=(ORIGY-cy)/MPP-half;W=int(2*half);mos=Image.new('L',(W,W),0)
    for col in range(int(x0//256),int((x0+W)//256)+1):
        for row in range(int(y0//256),int((y0+W)//256)+1):
            t=tile(col,row)
            if t: mos.paste(t,(col*256-int(x0),row*256-int(y0)))
    return np.asarray(mos,np.float32),x0,y0
def ll_to_px(lo,la,x0,y0):
    x,y=merc(lo,la);return (x-ORIG)/MPP-x0,(ORIGY-y)/MPP-y0
def score(mosA,px,py,clat,glo,ghi,recipe,wins=(40,56,72)):
    MPPg=MPP*math.cos(math.radians(clat));best=0.0
    for wm in wins:
        hw=int(wm/2/MPPg);w=mosA[py-hw:py+hw,px-hw:px+hw]
        if w.shape!=(2*hw,2*hw): continue
        if recipe=='clean':
            if (w>0).mean()<0.5: continue
            a=np.clip((w-glo)/(ghi-glo+1e-6),0,1);s=np.asarray(Image.fromarray((a*255).astype('uint8')).resize((128,128)),np.uint8)
        else:
            if w.std()<0.3: continue
            lo,hi=np.percentile(w,2),np.percentile(w,98);a=np.clip((w-lo)/(hi-lo+1e-6),0,1)
            f=max(1,int(round(2.0/MPPg)));Hh,Ww=a.shape;a=a[:Hh//f*f,:Ww//f*f].reshape(Hh//f,f,Ww//f,f).mean((1,3))
            s=homog(np.asarray(Image.fromarray((a*255).astype('uint8')).resize((128,128)),np.uint8))
        with torch.no_grad(): v=float(torch.sigmoid(net(torch.tensor(s[None,None],dtype=torch.float32).to(dev)/255.)).item())
        best=max(best,v)
    return best
# tumuli RAN CONFIRMAȚI cu coord (Botoșani, în acoperirea națională)
POS=[("Concesti-Cuzoaia",26.61935,48.12012),("Concesti-SE",26.5797,48.13954),
     ("Tataraseni-Movila",26.68799,48.07725),("Tataraseni-Mileanca-I",26.68508,48.07974),
     ("Tataraseni-Mileanca-II",26.6885,48.07665)]
def auroc(pos,neg):
    if not pos or not neg: return float('nan')
    c=sum((p>n)+0.5*(p==n) for p in pos for n in neg);return c/(len(pos)*len(neg))
for recipe in ('clean','homog'):
    pscores=[];allneg=[]
    print(f"\n===== rețetă {recipe.upper()} =====")
    for nm,lo,la in POS:
        mosA,x0,y0=fetch_mosaic(lo,la,2.0)
        v=mosA[mosA>0];glo,ghi=(np.percentile(v,1),np.percentile(v,99)) if v.size else (0,255)
        px,py=ll_to_px(lo,la,x0,y0);ps=score(mosA,int(px),int(py),la,glo,ghi,recipe)
        # control: 12 random points >180m from center, valide
        negs=[];tries=0
        while len(negs)<12 and tries<200:
            tries+=1;dx=random.uniform(-900,900);dy=random.uniform(-900,900)
            if dx*dx+dy*dy<180*180: continue
            qx=int(px+dx/(MPP*math.cos(math.radians(la))));qy=int(py+dy/(MPP*math.cos(math.radians(la))))
            ns=score(mosA,qx,qy,la,glo,ghi,recipe)
            if ns is not None: negs.append(ns)
        pscores.append(ps);allneg+=negs
        print(f"  {nm:24s} tumul={ps:.3f} | control med={np.median(negs):.3f} max={max(negs):.3f}")
    print(f"  --> POZITIVI med {np.median(pscores):.3f} | CONTROL med {np.median(allneg):.3f} | recall@0.7 {sum(p>=0.7 for p in pscores)}/{len(pscores)} | AUROC {auroc(pscores,allneg):.3f}")
