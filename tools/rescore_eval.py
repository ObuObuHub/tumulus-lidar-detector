#!/usr/bin/env python3
# rescore_eval.py CLON CLAT KM — re-scorează celulele din /tmp/eval_map.csv cu modelul CURENT (homogenizat)
# -> /tmp/eval_rescore.csv (idx,score). Pt a măsura precizia/recall după retrain pe marcajele lui Andrei.
import os,sys,math,subprocess,csv
import numpy as np
from PIL import Image,ImageFilter
import torch,torch.nn as nn
H=os.path.expanduser('~/lidar-match');dev=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
CLON=float(sys.argv[1]);CLAT=float(sys.argv[2]);KM=float(sys.argv[3]) if len(sys.argv)>3 else 5.0
CACHE="/tmp/laki3";CS=0.5;TPX=2000
APP="/Applications/QGIS-final-4_0_3.app/Contents"
ENV=dict(os.environ,DYLD_FRAMEWORK_PATH=f"{APP}/Frameworks",PROJ_DATA=f"{APP}/Resources/qgis/proj",PROJ_LIB=f"{APP}/Resources/qgis/proj",GDAL_DATA=f"{APP}/Resources/qgis/gdal")
GT=f"{APP}/MacOS/gdaltransform"
def trans(p,s,t):
    r=subprocess.run([GT,"-s_srs",s,"-t_srs",t],input=f"{p[0]} {p[1]}\n",capture_output=True,text=True,env=ENV);a,b,_=r.stdout.split();return float(a),float(b)
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
est,nord=trans((CLON,CLAT),"EPSG:4326","EPSG:3844");half=KM*1000/2
e0=int((est-half)//1000);e1=int((est+half)//1000);n0=int((nord-half)//1000);n1=int((nord+half)//1000)
xll0=e0*1000;ytop0=(n1+1)*1000;W=(e1-e0+1)*TPX;Hh=(n1-n0+1)*TPX
mos=np.full((Hh,W),np.nan,np.float32)
for nk in range(n0,n1+1):
    for ek in range(e0,e1+1):
        d=dl(nk,ek)
        if d is None: continue
        ox=int((ek*1000-xll0)/CS);oy=int((ytop0-(nk+1)*1000)/CS);mos[oy:oy+TPX,ox:ox+TPX]=d[:TPX,:TPX]
f=int(round(2.0/CS));hw=int(40/CS)
class Net(nn.Module):
    def __init__(s):
        super().__init__();s.c=nn.Sequential(nn.Conv2d(1,16,3,2,1),nn.ReLU(),nn.Conv2d(16,32,3,2,1),nn.ReLU(),nn.Conv2d(32,64,3,2,1),nn.ReLU(),nn.AdaptiveAvgPool2d(1));s.f=nn.Linear(64,1)
    def forward(s,x): return s.f(s.c(x).flatten(1)).squeeze(1)
net=Net().to(dev);net.load_state_dict(torch.load(f'{H}/combined_cnn.pt',weights_only=True));net.eval()
def stamp(e,n):
    px=int((e-xll0)/CS);py=int((ytop0-n)/CS);w=mos[py-hw:py+hw,px-hw:px+hw]
    if w.shape!=(2*hw,2*hw) or np.isnan(w).mean()>0.05: return None
    d2=downs(w,f);h=hs(d2,CS*f);lo,hi=np.percentile(h,2),np.percentile(h,98)
    if hi-lo<1e-6: return None
    raw=np.asarray(Image.fromarray(np.clip((h-lo)/(hi-lo)*255,0,255).astype('uint8')).resize((128,128)),np.uint8)
    return homog(raw)
# FILTRU COERENȚĂ DIRECȚIONALĂ (post-filtru tunat 24.06): suprimă detecția dacă e direcțională (arătură/șanț).
# coh22>0.70 → Catane 91%→100% (taie idx34), 0 movile pierdute. Activat cu env COHFILT (prag, ex 0.70).
COHFILT=float(os.environ['COHFILT']) if os.environ.get('COHFILT') else None
def coherence(e,n,rad_m=22):
    px=int((e-xll0)/CS);py=int((ytop0-n)/CS);r=int(rad_m/CS)
    w=mos[py-r:py+r,px-r:px+r]
    if w.shape!=(2*r,2*r) or np.isnan(w).mean()>0.1: return 0.0
    w=np.nan_to_num(w,nan=np.nanmedian(w));gy,gx=np.gradient(w)
    Jxx=(gx*gx).mean();Jyy=(gy*gy).mean();Jxy=(gx*gy).mean();den=Jxx+Jyy
    return math.sqrt((Jxx-Jyy)**2+4*Jxy*Jxy)/den if den>1e-12 else 0.0
rows=list(csv.DictReader(open('/tmp/eval_map.csv')))
out=open('/tmp/eval_rescore.csv','w');w=csv.writer(out);w.writerow(['idx','score'])
ims=[];idxs=[];ens=[]
for r in rows:
    e,n=trans((float(r['lon']),float(r['lat'])),"EPSG:4326","EPSG:3844");im=stamp(e,n)
    if im is None: w.writerow([r['idx'],"NA"]);continue
    ims.append(im);idxs.append(r['idx']);ens.append((e,n))
xb=torch.tensor(np.array(ims)).unsqueeze(1).float().to(dev)/255.
with torch.no_grad(): sc=torch.sigmoid(net(xb)).cpu().numpy()
ncut=0
for i,s,(e,n) in zip(idxs,sc,ens):
    fs=float(s)
    if COHFILT is not None and coherence(e,n)>COHFILT: fs=0.0;ncut+=1
    w.writerow([i,f"{fs:.3f}"])
out.close()
msg=f"-> /tmp/eval_rescore.csv ({len(idxs)} celule re-scorate cu modelul curent"
if COHFILT is not None: msg+=f"; filtru coh>{COHFILT}: {ncut} suprimate"
print(msg+")")
