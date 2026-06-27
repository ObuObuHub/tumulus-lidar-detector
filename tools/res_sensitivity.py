#!/usr/bin/env python3
# res_sensitivity.py [MODEL] [COORDS] — testează sensibilitatea la REZOLUȚIA SURSEI în banda fină.
# Pt fiecare movilă confirmată: ia fereastra 0.5m, o DEGRADEAZĂ la rez. simulate (downsample la r apoi înapoi),
# apoi rulează pipeline-ul standard (hs + downsample 2m + homog + 128) și scorează. Vede UNDE cade scorul.
import os,sys,math,subprocess,csv
import numpy as np
from PIL import Image,ImageFilter
import torch,torch.nn as nn
H=os.path.dirname(os.path.dirname(os.path.abspath(__file__)));dev=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
CACHE="/tmp/laki3";CS=0.5;TPX=2000
MODEL=sys.argv[1] if len(sys.argv)>1 else f'{H}/combined_cnn.pt'
COORDS=sys.argv[2] if len(sys.argv)>2 else '/tmp/catane_gt.csv'
APP=os.environ.get("QGIS_APP","/Applications/QGIS-final-4_0_3.app/Contents")
ENV=dict(os.environ,DYLD_FRAMEWORK_PATH=f"{APP}/Frameworks",PROJ_DATA=f"{APP}/Resources/qgis/proj",PROJ_LIB=f"{APP}/Resources/qgis/proj",GDAL_DATA=f"{APP}/Resources/qgis/gdal")
GTb=f"{APP}/MacOS/gdaltransform"
def trans(pts,s,t):
    inp="\n".join(f"{a} {b}" for a,b in pts)+"\n";r=subprocess.run([GTb,"-s_srs",s,"-t_srs",t],input=inp,capture_output=True,text=True,env=ENV)
    return [tuple(map(float,l.split()[:2])) for l in r.stdout.strip().split("\n") if l.split()]
def hs(dem,cs,azs=(315,45,135,225,270,0),alt=35):
    gy,gx=np.gradient(dem,cs);sl=np.arctan(np.hypot(gx,gy));asp=np.arctan2(-gy,gx);o=np.zeros_like(dem);ar=math.radians(alt)
    for az in azs: azr=math.radians(360-az+90);o+=np.clip(np.sin(ar)*np.cos(sl)+np.cos(ar)*np.sin(sl)*np.cos(azr-asp),0,1)
    return o/len(azs)
def downs(a,fc):
    Hh,Ww=a.shape;return a[:Hh//fc*fc,:Ww//fc*fc].reshape(Hh//fc,fc,Ww//fc,fc).mean((1,3))
def homog(a):
    a2=np.asarray(Image.fromarray(a).filter(ImageFilter.GaussianBlur(0.8)),np.uint8);cdf=np.bincount(a2.ravel(),minlength=256).astype(np.float64).cumsum()
    return (cdf[a2]/cdf[-1]*255).astype(np.uint8) if cdf[-1]>0 else a2
class Net(nn.Module):
    def __init__(s):
        super().__init__();s.c=nn.Sequential(nn.Conv2d(1,16,3,2,1),nn.ReLU(),nn.Conv2d(16,32,3,2,1),nn.ReLU(),nn.Conv2d(32,64,3,2,1),nn.ReLU(),nn.AdaptiveAvgPool2d(1));s.f=nn.Linear(64,1)
    def forward(s,x): return s.f(s.c(x).flatten(1)).squeeze(1)
net=Net().to(dev);net.load_state_dict(torch.load(MODEL,map_location=dev,weights_only=True));net.eval()
hw=int(40/CS)  # 80m window @0.5m
def load_win(est,nord):
    e0=int((est-60)//1000);e1=int((est+60)//1000);n0=int((nord-60)//1000);n1=int((nord+60)//1000)
    xll=e0*1000;ytop=(n1+1)*1000;W=(e1-e0+1)*TPX;Hh=(n1-n0+1)*TPX;mos=np.full((Hh,W),np.nan,np.float32);got=False
    for nk in range(n0,n1+1):
        for ek in range(e0,e1+1):
            p=f"{CACHE}/{nk}_{ek}.npy"
            if os.path.exists(p): d=np.load(p);got=True;ox=int((ek*1000-xll)/CS);oy=int((ytop-(nk+1)*1000)/CS);mos[oy:oy+TPX,ox:ox+TPX]=d[:TPX,:TPX]
    if not got: return None
    px=int((est-xll)/CS);py=int((ytop-nord)/CS);w=mos[py-hw:py+hw,px-hw:px+hw]
    return w if w.shape==(2*hw,2*hw) else None
def degrade(w,res_m):
    # simulează sursa la res_m: block-downsample la res_m apoi înapoi la grila 0.5m
    if res_m<=CS+1e-6: return w
    fc=max(1,int(round(res_m/CS)))
    small=downs(w,fc)
    return np.asarray(Image.fromarray(small).resize((w.shape[1],w.shape[0]),Image.BILINEAR))
def score_at(w,res_m):
    wd=degrade(np.nan_to_num(w,nan=np.nanmedian(w)),res_m)
    f=int(round(2.0/CS));d2=downs(wd,f);h=hs(d2,CS*f);lo,hi=np.percentile(h,2),np.percentile(h,98)
    if hi-lo<1e-6: return None
    im=homog(np.asarray(Image.fromarray(np.clip((h-lo)/(hi-lo)*255,0,255).astype('uint8')).resize((128,128)),np.uint8))
    with torch.no_grad(): return float(torch.sigmoid(net(torch.tensor(im[None,None],dtype=torch.float32).to(dev)/255.)).item())
rows=list(csv.DictReader(open(COORDS)))
en=trans([(float(r['lon']),float(r['lat'])) for r in rows],"EPSG:4326","EPSG:3844")
RES=[0.5,1.0,1.5,2.0,3.0,5.0]
bym={r:[] for r in RES}
for r,(e,n) in zip(rows,en):
    w=load_win(e,n)
    if w is None: continue
    for rr in RES:
        s=score_at(w,rr)
        if s is not None: bym[rr].append(s)
print(f"MODEL {os.path.basename(MODEL)} | {len(bym[0.5])} movile | scor mediu vs REZOLUȚIA sursei simulate:")
base=np.mean(bym[0.5]) if bym[0.5] else 1
for rr in RES:
    a=np.array(bym[rr]);print(f"  {rr:>4.1f}m: scor mediu {a.mean():.3f} | median {np.median(a):.3f} | %>=0.7 {100*(a>=0.7).mean():.0f}%  ({100*a.mean()/base:.0f}% din nativ)")
