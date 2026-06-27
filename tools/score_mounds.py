#!/usr/bin/env python3
# score_mounds.py MODEL [lon_min lon_max lat_min lat_max] — scorează movilele REALE din labels.csv (verdict=mound)
# dintr-un bbox (default = AR_MDH/Arad) cu modelul dat, pe stampe hillshade MDH (IDENTIC cu arad_diag/scanarea).
# Raportează distribuția scorurilor + recall@0.6/0.5 = cât de bine vede modelul movilele reale (gardă anti-suprimare).
import sys,os,math,subprocess,csv
import numpy as np
from PIL import Image,ImageFilter
import torch,torch.nn as nn
H=os.path.dirname(os.path.dirname(os.path.abspath(__file__)));dev=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
MODEL=sys.argv[1]
LOMIN,LOMAX,LAMIN,LAMAX=(float(sys.argv[2]),float(sys.argv[3]),float(sys.argv[4]),float(sys.argv[5])) if len(sys.argv)>5 else (20.67,22.77,45.86,46.70)
R=6378137.0;ORIG=-20037508.342787;ORIGY=20037508.342787;Z=18;MPP=2*math.pi*R/(256*2**Z)
MDH=[("AR_MDH_tif",(20.67,45.86,22.77,46.70)),("BH_MDH_tif",(21.37,46.36,22.83,47.61)),("HD_MDH_tif",(22.32,45.23,23.60,46.37)),("AB_MDH_tif",(22.66,45.44,23.82,46.59))]
ORG="Q2Kmg0bQDn3rySgn";TDIR="/tmp/mdh_tiles_L18";os.makedirs(TDIR,exist_ok=True)
def pick(lo,la):
    for svc,(a,b,c,d) in MDH:
        if a<=lo<=c and b<=la<=d: return svc
def tile(svc,col,row):
    fn=f"{TDIR}/{svc}_{col}_{row}.png"
    if not os.path.exists(fn): subprocess.run(["curl","-s","--max-time","25","-o",fn,f"https://tiles.arcgis.com/tiles/{ORG}/arcgis/rest/services/{svc}/MapServer/tile/{Z}/{row}/{col}"],check=False)
    try: return Image.open(fn).convert('L')
    except: return None
def homog(a):
    a2=np.asarray(Image.fromarray(a).filter(ImageFilter.GaussianBlur(0.8)),np.uint8);cdf=np.bincount(a2.ravel(),minlength=256).astype(np.float64).cumsum()
    return (cdf[a2]/cdf[-1]*255).astype(np.uint8) if cdf[-1]>0 else a2
def stamp(lo,la,meters=80,eff=2.0,out=128):
    svc=pick(lo,la)
    if not svc: return None
    half=meters/2/MPP;x=R*math.radians(lo);y=R*math.log(math.tan(math.pi/4+math.radians(la)/2))
    px=(x-ORIG)/MPP;py=(ORIGY-y)/MPP;x0=px-half;y0=py-half;W=int(2*half);cv=Image.new('L',(W,W),0);ok=False
    for col in range(int(x0//256),int((x0+W)//256)+1):
        for row in range(int(y0//256),int((y0+W)//256)+1):
            t=tile(svc,col,row)
            if t: cv.paste(t,(col*256-int(x0),row*256-int(y0)));ok=True
    if not ok: return None
    a=np.asarray(cv,np.float32)
    if a.std()<0.5: return None
    lo2,hi2=np.percentile(a,2),np.percentile(a,98);a=np.clip((a-lo2)/(hi2-lo2+1e-6),0,1)
    f=max(1,int(round(eff/MPP)));Hh,Ww=a.shape;a=a[:Hh//f*f,:Ww//f*f].reshape(Hh//f,f,Ww//f,f).mean((1,3))
    return homog(np.asarray(Image.fromarray((a*255).astype('uint8')).resize((out,out)),np.uint8))
class Net(nn.Module):
    def __init__(s):
        super().__init__();s.c=nn.Sequential(nn.Conv2d(1,16,3,2,1),nn.ReLU(),nn.Conv2d(16,32,3,2,1),nn.ReLU(),nn.Conv2d(32,64,3,2,1),nn.ReLU(),nn.AdaptiveAvgPool2d(1));s.f=nn.Linear(64,1)
    def forward(s,x): return s.f(s.c(x).flatten(1)).squeeze(1)
net=Net().to(dev);net.load_state_dict(torch.load(MODEL,map_location=dev,weights_only=True));net.eval()
rows=[r for r in csv.DictReader(open(f'{H}/labeled/labels.csv')) if r.get('verdict')=='mound']
pts=[(float(r['lon']),float(r['lat'])) for r in rows if LOMIN<=float(r['lon'])<=LOMAX and LAMIN<=float(r['lat'])<=LAMAX]
scores=[]
for lo,la in pts:
    s=stamp(lo,la)
    if s is None: continue
    with torch.no_grad():
        sc=float(torch.sigmoid(net(torch.tensor(s).unsqueeze(0).unsqueeze(0).float().to(dev)/255.)).item())
    scores.append(sc)
scores=np.array(scores)
print(f"model {os.path.basename(MODEL)} | {len(scores)} movile reale scorate (bbox {LOMIN},{LAMIN}-{LOMAX},{LAMAX})")
for t in (0.9,0.6,0.5,0.3):
    print(f"  scor>={t}: {int((scores>=t).sum())}/{len(scores)} ({100*(scores>=t).mean():.0f}%)")
print(f"  median={np.median(scores):.3f} mean={scores.mean():.3f} min={scores.min():.3f}")
