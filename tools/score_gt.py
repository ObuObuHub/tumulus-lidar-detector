#!/usr/bin/env python3
# score_gt.py GT_CSV [MODEL] — scorează fiecare movilă GT (lon,lat) cu modelul (recipe neg_stamp + homog,
# max peste offset-uri ±30m ca scanarea) și listează sortat. Cea mai mică = movila pe care o ratează modelul.
import os,sys,math,subprocess,csv
import numpy as np
from PIL import Image,ImageFilter
import torch,torch.nn as nn
H=os.path.dirname(os.path.dirname(os.path.abspath(__file__)));dev=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
GTf=sys.argv[1];MODEL=sys.argv[2] if len(sys.argv)>2 else f'{H}/combined_cnn.pt'
CACHE="/tmp/laki3";CS=0.5;f=int(round(2.0/CS));WPX=int(80/CS)
APP=os.environ.get("QGIS_APP","/Applications/QGIS-final-4_0_3.app/Contents")
ENV=dict(os.environ,DYLD_FRAMEWORK_PATH=f"{APP}/Frameworks",PROJ_DATA=f"{APP}/Resources/qgis/proj",PROJ_LIB=f"{APP}/Resources/qgis/proj",GDAL_DATA=f"{APP}/Resources/qgis/gdal")
GT=f"{APP}/MacOS/gdaltransform"
def trans(pts):
    inp="".join(f"{a} {b}\n" for a,b in pts)
    r=subprocess.run([GT,"-s_srs","EPSG:4326","-t_srs","EPSG:3844"],input=inp,capture_output=True,text=True,env=ENV)
    return [(float(x.split()[0]),float(x.split()[1])) for x in r.stdout.strip().split("\n")]
def hs(dem,cs,azs=(315,45,135,225,270,0),alt=35):
    gy,gx=np.gradient(dem,cs);sl=np.arctan(np.hypot(gx,gy));asp=np.arctan2(-gy,gx);o=np.zeros_like(dem);ar=math.radians(alt)
    for az in azs: azr=math.radians(360-az+90);o+=np.clip(np.sin(ar)*np.cos(sl)+np.cos(ar)*np.sin(sl)*np.cos(azr-asp),0,1)
    return o/len(azs)
def downs(a,fc):
    Hh,Ww=a.shape;return a[:Hh//fc*fc,:Ww//fc*fc].reshape(Hh//fc,fc,Ww//fc,fc).mean((1,3))
def _histeq(a):
    h=np.bincount(a.ravel(),minlength=256).astype(np.float64);cdf=h.cumsum();return a if cdf[-1]==0 else (cdf[a]/cdf[-1]*255).astype(np.uint8)
def stamp(T,px,py):
    hw=WPX//2;w=T[py-hw:py+hw,px-hw:px+hw]
    if w.shape!=(WPX,WPX) or np.isnan(w).mean()>0.05: return None
    d2=downs(np.nan_to_num(w,nan=np.nanmedian(w)),f);h=hs(d2,CS*f);lo,hi=np.percentile(h,2),np.percentile(h,98)
    if hi-lo<1e-6: return None
    raw=np.asarray(Image.fromarray(np.clip((h-lo)/(hi-lo)*255,0,255).astype('uint8')).resize((128,128)),np.uint8)
    return _histeq(np.asarray(Image.fromarray(raw).filter(ImageFilter.GaussianBlur(0.8)),np.uint8))
class Net(nn.Module):
    def __init__(s):
        super().__init__();s.c=nn.Sequential(nn.Conv2d(1,16,3,2,1),nn.ReLU(),nn.Conv2d(16,32,3,2,1),nn.ReLU(),nn.Conv2d(32,64,3,2,1),nn.ReLU(),nn.AdaptiveAvgPool2d(1));s.f=nn.Linear(64,1)
    def forward(s,x): return s.f(s.c(x).flatten(1)).squeeze(1)
net=Net().to(dev);net.load_state_dict(torch.load(MODEL,map_location=dev,weights_only=True));net.eval()
def sb(sts):
    xb=torch.tensor(np.array(sts,dtype=np.uint8)).unsqueeze(1).float().to(dev)/255.
    with torch.no_grad(): return torch.sigmoid(net(xb)).cpu().numpy()
rows=list(csv.DictReader(open(GTf)))
pts=[(float(r['lon']),float(r['lat'])) for r in rows]
en=trans(pts)
cache={}
def tile(nk,ek):
    if (nk,ek) not in cache:
        p=f"{CACHE}/{nk}_{ek}.npy";cache[(nk,ek)]=np.load(p) if os.path.exists(p) else None
    return cache[(nk,ek)]
res=[]
for (lon,lat),(e,n) in zip(pts,en):
    best=0.0;sts=[]
    for de in range(-30,31,15):
        for dn in range(-30,31,15):
            ee=e+de;nn=n+dn;nk=int(nn//1000);ek=int(ee//1000);T=tile(nk,ek)
            if T is None: continue
            px=int(round((ee-ek*1000)/CS));py=int(round(((nk+1)*1000-nn)/CS))
            s=stamp(T,px,py)
            if s is not None: sts.append(s)
    if sts: best=float(sb(sts).max())
    res.append((best,lon,lat))
res.sort()
print(f"GT scoruri ({MODEL.split('/')[-1]}), sortate crescător:")
for sc,lon,lat in res: print(f"  {sc:.3f}  {lat:.5f},{lon:.5f}  {'<-- RATAT' if sc<0.5 else ''}")
print(f"min {res[0][0]:.3f} | <0.5: {sum(1 for r in res if r[0]<0.5)} | <0.7: {sum(1 for r in res if r[0]<0.7)}")
