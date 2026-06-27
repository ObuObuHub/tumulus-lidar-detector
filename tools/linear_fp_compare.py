#!/usr/bin/env python3
# linear_fp_compare.py MODEL CLON CLAT KM — sweep pe zonă cached cu MODEL, numără detecțiile LINIARE
# (coerență direcțională mare = șanț/canal) vs compacte printre cele cu scor mare. Pt A/B clean vs random.
import os,sys,math,subprocess
import numpy as np
from PIL import Image,ImageFilter
import torch,torch.nn as nn
H=os.path.dirname(os.path.dirname(os.path.abspath(__file__)));dev=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
MODEL=sys.argv[1];CLON=float(sys.argv[2]);CLAT=float(sys.argv[3]);KM=float(sys.argv[4]) if len(sys.argv)>4 else 6.0
CACHE="/tmp/laki3";CS=0.5;TPX=2000
APP=os.environ.get("QGIS_APP","/Applications/QGIS-final-4_0_3.app/Contents")
ENV=dict(os.environ,DYLD_FRAMEWORK_PATH=f"{APP}/Frameworks",PROJ_DATA=f"{APP}/Resources/qgis/proj",PROJ_LIB=f"{APP}/Resources/qgis/proj",GDAL_DATA=f"{APP}/Resources/qgis/gdal")
GT=f"{APP}/MacOS/gdaltransform"
def to_st(lo,la):
    r=subprocess.run([GT,"-s_srs","EPSG:4326","-t_srs","EPSG:3844"],input=f"{lo} {la}\n",capture_output=True,text=True,env=ENV);a,b,_=r.stdout.split();return float(a),float(b)
def dl(nk,ek):
    p=f"{CACHE}/{nk}_{ek}.npy";return np.load(p) if os.path.exists(p) else None
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
est,nord=to_st(CLON,CLAT);half=KM*1000/2
e0=int((est-half)//1000);e1=int((est+half)//1000);n0=int((nord-half)//1000);n1=int((nord+half)//1000)
xll0=e0*1000;ytop0=(n1+1)*1000;W=(e1-e0+1)*TPX;Hh=(n1-n0+1)*TPX
mos=np.full((Hh,W),np.nan,np.float32)
for nk in range(n0,n1+1):
    for ek in range(e0,e1+1):
        d=dl(nk,ek)
        if d is None: continue
        ox=int((ek*1000-xll0)/CS);oy=int((ytop0-(nk+1)*1000)/CS);mos[oy:oy+TPX,ox:ox+TPX]=d[:TPX,:TPX]
f=int(round(2.0/CS));hw=int(40/CS);step=int(80/CS)
def stamp(px,py):
    w=mos[py-hw:py+hw,px-hw:px+hw]
    if w.shape!=(2*hw,2*hw) or np.isnan(w).mean()>0.05: return None,None
    d2=downs(w,f);h=hs(d2,CS*f);lo,hi=np.percentile(h,2),np.percentile(h,98)
    if hi-lo<1e-6: return None,None
    raw=np.asarray(Image.fromarray(np.clip((h-lo)/(hi-lo)*255,0,255).astype('uint8')).resize((128,128)),np.uint8)
    # coerenta direcțională (liniar?) pe fereastra 2m
    gy,gx=np.gradient(d2);Jxx=(gx*gx).mean();Jyy=(gy*gy).mean();Jxy=(gx*gy).mean()
    coh=math.sqrt((Jxx-Jyy)**2+4*Jxy*Jxy)/(Jxx+Jyy+1e-9)
    return homog(raw),coh
cells=[];cohs=[]
ys=list(range(hw,Hh-hw,step));xs=list(range(hw,W-hw,step))
for py in ys:
    for px in xs:
        s,coh=stamp(px,py)
        if s is not None: cells.append(s);cohs.append(coh)
X=torch.tensor(np.array(cells,dtype=np.uint8))
sc=[]
with torch.no_grad():
    for i in range(0,len(X),512): sc.extend(torch.sigmoid(net(X[i:i+512].unsqueeze(1).float().to(dev)/255.)).cpu().numpy().tolist())
sc=np.array(sc);cohs=np.array(cohs)
for T in (0.7,0.8,0.9):
    hi=sc>=T;lin=hi&(cohs>=0.55);comp=hi&(cohs<0.55)
    print(f"  @{T}: high-score {int(hi.sum())} | LINIARE {int(lin.sum())} | compacte {int(comp.sum())}")
print(f"  ({len(cells)} celule scanate, model {os.path.basename(MODEL)})")
