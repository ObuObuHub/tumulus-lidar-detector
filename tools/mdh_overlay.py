#!/usr/bin/env python3
# mdh_overlay.py CLON CLAT KM [MODEL] [THRESH] — overlay: hillshade MDH al zonei + TOATE detecțiile modelului
# (cercuri colorate pe scor). Ca să comparăm direct cu ground-truth-ul lui Andrei (ce-ar trebui detectat).
import sys,os,math,subprocess
import numpy as np
from PIL import Image,ImageFilter,ImageDraw
import torch,torch.nn as nn
H=os.path.expanduser('~/lidar-match');dev=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
CLON=float(sys.argv[1]);CLAT=float(sys.argv[2]);KM=float(sys.argv[3]) if len(sys.argv)>3 else 1.5
MODEL=sys.argv[4] if len(sys.argv)>4 else f'{H}/combined_cleanpool.pt';TH=float(sys.argv[5]) if len(sys.argv)>5 else 0.6
R=6378137.0;ORIG=-20037508.342787;ORIGY=20037508.342787;Z=18;MPP=2*math.pi*R/(256*2**Z)  # L18=0.6m/px
MDH=[("AR_MDH_tif",(20.67,45.86,22.77,46.70)),("BH_MDH_tif",(21.37,46.36,22.83,47.61)),("HD_MDH_tif",(22.32,45.23,23.60,46.37)),("AB_MDH_tif",(22.66,45.44,23.82,46.59))]
ORG="Q2Kmg0bQDn3rySgn";TDIR="/tmp/mdh_tiles_L18";os.makedirs(TDIR,exist_ok=True)
def pick(lo,la):
    for svc,(a,b,c,d) in MDH:
        if a<=lo<=c and b<=la<=d: return svc
def merc(lo,la): return R*math.radians(lo), R*math.log(math.tan(math.pi/4+math.radians(la)/2))
def tilepx(svc,col,row):
    fn=f"{TDIR}/{svc}_{col}_{row}.png"
    if not os.path.exists(fn): subprocess.run(["curl","-s","--max-time","25","-o",fn,f"https://tiles.arcgis.com/tiles/{ORG}/arcgis/rest/services/{svc}/MapServer/tile/{Z}/{row}/{col}"],check=False)
    try: return Image.open(fn).convert('L')
    except: return None
svc=pick(CLON,CLAT);cx,cy=merc(CLON,CLAT)
half=KM*1000/2; pxc=(cx-ORIG)/MPP; pyc=(ORIGY-cy)/MPP; hp=half/MPP
x0=pxc-hp;y0=pyc-hp;W=int(2*hp)
# mozaic hillshade MDH
mos=Image.new('L',(W,W),0)
for col in range(int(x0//256),int((x0+W)//256)+1):
    for row in range(int(y0//256),int((y0+W)//256)+1):
        t=tilepx(svc,col,row)
        if t: mos.paste(t,(col*256-int(x0),row*256-int(y0)))
mosA=np.asarray(mos,np.float32)
print(f"mozaic {W}x{W}px ({KM}km, ~{MPP:.2f}m/px), svc {svc}",flush=True)
def homog(a):
    a2=np.asarray(Image.fromarray(a).filter(ImageFilter.GaussianBlur(0.8)),np.uint8);cdf=np.bincount(a2.ravel(),minlength=256).astype(np.float64).cumsum()
    return (cdf[a2]/cdf[-1]*255).astype(np.uint8) if cdf[-1]>0 else a2
def stamp_px(px,py,meters=80,eff=2.0,out=128):
    hw=int(meters/2/MPP);w=mosA[py-hw:py+hw,px-hw:px+hw]
    if w.shape!=(2*hw,2*hw) or w.std()<0.5: return None
    lo,hi=np.percentile(w,2),np.percentile(w,98);a=np.clip((w-lo)/(hi-lo+1e-6),0,1)
    f=max(1,int(round(eff/MPP)));Hh,Ww=a.shape;a=a[:Hh//f*f,:Ww//f*f].reshape(Hh//f,f,Ww//f,f).mean((1,3))
    return homog(np.asarray(Image.fromarray((a*255).astype('uint8')).resize((out,out)),np.uint8))
class Net(nn.Module):
    def __init__(s):
        super().__init__();s.c=nn.Sequential(nn.Conv2d(1,16,3,2,1),nn.ReLU(),nn.Conv2d(16,32,3,2,1),nn.ReLU(),nn.Conv2d(32,64,3,2,1),nn.ReLU(),nn.AdaptiveAvgPool2d(1));s.f=nn.Linear(64,1)
    def forward(s,x): return s.f(s.c(x).flatten(1)).squeeze(1)
net=Net().to(dev);net.load_state_dict(torch.load(MODEL,map_location=dev,weights_only=True));net.eval()
step=int(40/MPP);hw=int(40/MPP);batch=[];pos=[]
for py in range(hw,W-hw,step):
    for px in range(hw,W-hw,step):
        s=stamp_px(px,py)
        if s is not None: batch.append(s);pos.append((px,py))
X=torch.tensor(np.array(batch,dtype=np.uint8));sc=[]
with torch.no_grad():
    for k in range(0,len(X),512): sc.extend(torch.sigmoid(net(X[k:k+512].unsqueeze(1).float().to(dev)/255.)).cpu().numpy().tolist())
sc=np.array(sc)
# NMS pe detecții >=TH
idx=np.argsort(-sc);kept=[]
for k in idx:
    if sc[k]<TH: break
    px,py=pos[k]
    if any((px-q[0])**2+(py-q[1])**2 < (60/MPP)**2 for q in kept): continue
    kept.append((px,py,float(sc[k])))
print(f"  {int((sc>=TH).sum())} celule >={TH} | {len(kept)} detecții (NMS)",flush=True)
ov=Image.fromarray(mosA.astype('uint8')).convert('RGB');dr=ImageDraw.Draw(ov)
for px,py,s in kept:
    r=int(40/MPP);col=(255,40,40) if s>=0.9 else (255,170,40)
    dr.ellipse([px-r,py-r,px+r,py+r],outline=col,width=2)
out=f"{H}/review/mdh_overlay.png";ov.resize((min(W,1100),int(min(W,1100)*W/W))).save(out)
print(f"  -> {out} ({len(kept)} detecții marcate; roșu>=0.9, portocaliu>={TH})")
