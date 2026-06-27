#!/usr/bin/env python3
# center_sensitivity.py MODEL COORDS_CSV — testează invarianța la translație: pt fiecare movilă (lon,lat),
# scorează modelul pe o grilă de offset-uri (±30m, pas 6m) și raportează cât scade scorul descentrat.
# Profil mediu scor vs offset + „halfwidth" (offset unde scorul scade la jumătate). Pt baseline vs jitter.
import os,sys,math,subprocess,csv
import numpy as np
from PIL import Image,ImageFilter
import torch,torch.nn as nn
H=os.path.dirname(os.path.dirname(os.path.abspath(__file__)));dev=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
CACHE="/tmp/laki3";CS=0.5;TPX=2000
MODEL=sys.argv[1];COORDS=sys.argv[2]
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
f=int(round(2.0/CS));hw=int(40/CS)
def load_win(est,nord,half_m=110):
    e0=int((est-half_m)//1000);e1=int((est+half_m)//1000);n0=int((nord-half_m)//1000);n1=int((nord+half_m)//1000)
    xll=e0*1000;ytop=(n1+1)*1000;W=(e1-e0+1)*TPX;Hh=(n1-n0+1)*TPX;mos=np.full((Hh,W),np.nan,np.float32);got=False
    for nk in range(n0,n1+1):
        for ek in range(e0,e1+1):
            p=f"{CACHE}/{nk}_{ek}.npy"
            if os.path.exists(p): d=np.load(p);got=True;ox=int((ek*1000-xll)/CS);oy=int((ytop-(nk+1)*1000)/CS);mos[oy:oy+TPX,ox:ox+TPX]=d[:TPX,:TPX]
    return (mos,xll,ytop) if got else None
OFFS=list(range(-30,31,6))  # metri
def profile(est,nord):
    win=load_win(est,nord)
    if win is None: return None
    mos,xll,ytop=win;ims=[];keys=[]
    for dy in OFFS:
        for dx in OFFS:
            e=est+dx;n=nord+dy;px=int((e-xll)/CS);py=int((ytop-n)/CS);w=mos[py-hw:py+hw,px-hw:px+hw]
            if w.shape!=(2*hw,2*hw) or np.isnan(w).mean()>0.05: continue
            d2=downs(np.nan_to_num(w,nan=np.nanmedian(w)),f);h=hs(d2,CS*f);lo,hi=np.percentile(h,2),np.percentile(h,98)
            if hi-lo<1e-6: continue
            ims.append(homog(np.asarray(Image.fromarray(np.clip((h-lo)/(hi-lo)*255,0,255).astype('uint8')).resize((128,128)),np.uint8)));keys.append((dx,dy))
    if not ims: return None
    X=torch.tensor(np.array(ims),dtype=torch.float32).unsqueeze(1).to(dev)/255.
    with torch.no_grad(): sc=torch.sigmoid(net(X)).cpu().numpy()
    return dict(zip(keys,sc.tolist()))
rows=list(csv.DictReader(open(COORDS)))
en=trans([(float(r['lon']),float(r['lat'])) for r in rows],"EPSG:4326","EPSG:3844")
# aggregate score by offset radius
bands={0:[],6:[],12:[],18:[],24:[],30:[]}
center=[];ring15=[];ring30=[]
for r,(e,n) in zip(rows,en):
    p=profile(e,n)
    if p is None: continue
    c=p.get((0,0),np.nan)
    if np.isnan(c): continue
    center.append(c)
    for (dx,dy),s in p.items():
        rad=round(math.hypot(dx,dy))
        b=min(bands.keys(),key=lambda k:abs(k-rad)); bands[b].append(s)
    r15=[s for (dx,dy),s in p.items() if 12<=math.hypot(dx,dy)<=18]
    r30=[s for (dx,dy),s in p.items() if 27<=math.hypot(dx,dy)<=33]
    if r15: ring15.append(np.mean(r15))
    if r30: ring30.append(np.mean(r30))
print(f"MODEL {os.path.basename(MODEL)} | {len(center)} movile")
print(f"  scor mediu la CENTRU: {np.mean(center):.3f}")
print(f"  scor mediu inel ~15m: {np.mean(ring15):.3f}  ({100*np.mean(ring15)/np.mean(center):.0f}% din centru)")
print(f"  scor mediu inel ~30m: {np.mean(ring30):.3f}  ({100*np.mean(ring30)/np.mean(center):.0f}% din centru)")
print("  profil scor vs offset-radius:")
for b in sorted(bands):
    if bands[b]: print(f"    ~{b:2d}m: {np.mean(bands[b]):.3f}")
