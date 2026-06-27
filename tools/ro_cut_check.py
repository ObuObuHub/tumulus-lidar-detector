#!/usr/bin/env python3
# ro_cut_check.py — găsește candidații RO broad (model_found+greens) pe care filtrul coh>0.70 îi taie,
# listează (sursă, scor model, coerență) + randează board hillshade ca să judecăm: tumul real pe pantă vs FP direcțional.
import os,math,subprocess,csv,glob,random
import numpy as np
from PIL import Image,ImageFilter,ImageDraw,ImageFont
import torch,torch.nn as nn
H=os.path.expanduser('~/lidar-match');dev=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
CACHE="/tmp/laki3";CS=0.5;TPX=2000
APP="/Applications/QGIS-final-4_0_3.app/Contents"
ENV=dict(os.environ,DYLD_FRAMEWORK_PATH=f"{APP}/Frameworks",PROJ_DATA=f"{APP}/Resources/qgis/proj",PROJ_LIB=f"{APP}/Resources/qgis/proj",GDAL_DATA=f"{APP}/Resources/qgis/gdal")
GT=f"{APP}/MacOS/gdaltransform"
def trans_many(pts):
    inp="".join(f"{lo} {la}\n" for lo,la in pts)
    r=subprocess.run([GT,"-s_srs","EPSG:4326","-t_srs","EPSG:3844"],input=inp,capture_output=True,text=True,env=ENV)
    return [tuple(map(float,l.split()[:2])) for l in r.stdout.strip().split("\n")]
def hs(dem,cs,azs=(315,45,135,225,270,0),alt=35):
    gy,gx=np.gradient(dem,cs);sl=np.arctan(np.hypot(gx,gy));asp=np.arctan2(-gy,gx);o=np.zeros_like(dem);ar=math.radians(alt)
    for az in azs: azr=math.radians(360-az+90);o+=np.clip(np.sin(ar)*np.cos(sl)+np.cos(ar)*np.sin(sl)*np.cos(azr-asp),0,1)
    return o/len(azs)
def downs(a,fc):
    Hh,Ww=a.shape;return a[:Hh//fc*fc,:Ww//fc*fc].reshape(Hh//fc,fc,Ww//fc,fc).mean((1,3))
def homog(a):
    a2=np.asarray(Image.fromarray(a).filter(ImageFilter.GaussianBlur(0.8)),np.uint8);cdf=np.bincount(a2.ravel(),minlength=256).astype(np.float64).cumsum()
    return (cdf[a2]/cdf[-1]*255).astype(np.uint8) if cdf[-1]>0 else a2
_tc={}
def tile(nk,ek):
    k=(nk,ek)
    if k not in _tc:
        if len(_tc)>60: _tc.clear()
        p=f"{CACHE}/{nk}_{ek}.npy";_tc[k]=np.load(p) if os.path.exists(p) else None
    return _tc[k]
def window(est,nord,half):
    eks=sorted({int((est-half*CS)//1000),int((est+half*CS)//1000)});nks=sorted({int((nord-half*CS)//1000),int((nord+half*CS)//1000)})
    e0=min(eks);n1=max(nks);xll=e0*1000;ytop=(n1+1)*1000
    W=(max(eks)-e0+1)*TPX;Hh=(n1-min(nks)+1)*TPX;mos=np.full((Hh,W),np.nan,np.float32);got=False
    for nk in nks:
        for ek in eks:
            d=tile(nk,ek)
            if d is None: continue
            got=True;ox=int((ek*1000-xll)/CS);oy=int((ytop-(nk+1)*1000)/CS);mos[oy:oy+TPX,ox:ox+TPX]=d[:TPX,:TPX]
    if not got: return None
    px=int((est-xll)/CS);py=int((ytop-nord)/CS);return mos[py-half:py+half,px-half:px+half]
f=int(round(2.0/CS));hw=int(40/CS)
class Net(nn.Module):
    def __init__(s):
        super().__init__();s.c=nn.Sequential(nn.Conv2d(1,16,3,2,1),nn.ReLU(),nn.Conv2d(16,32,3,2,1),nn.ReLU(),nn.Conv2d(32,64,3,2,1),nn.ReLU(),nn.AdaptiveAvgPool2d(1));s.f=nn.Linear(64,1)
    def forward(s,x): return s.f(s.c(x).flatten(1)).squeeze(1)
net=Net().to(dev);net.load_state_dict(torch.load(f'{H}/combined_cnn.pt',weights_only=True));net.eval()
def in_laki(lo,la): return 22.85<=lo<=24.05 and 43.78<=la<=44.55
broad=[]
for r in csv.DictReader(open('labeled/labels.csv')):
    lo,la=float(r['lon']),float(r['lat'])
    if in_laki(lo,la) and r['source']!='gold_ran': broad.append((lo,la,r['source']))
for fn in glob.glob('labeled/batch*_greens.csv'):
    for r in csv.DictReader(open(fn)):
        lo,la=float(r['lon']),float(r['lat'])
        if in_laki(lo,la): broad.append((lo,la,'green'))
seen=set();uniq=[]
for lo,la,s in broad:
    k=(round(lo,5),round(la,5))
    if k in seen: continue
    seen.add(k);uniq.append((lo,la,s))
EN=trans_many([(lo,la) for lo,la,_ in uniq])
cut=[]
for (lo,la,src),(e,n) in zip(uniq,EN):
    r=int(22/CS);cw=window(e,n,r)
    if cw is None or cw.shape!=(2*r,2*r) or np.isnan(cw).mean()>0.1: continue
    cw=np.nan_to_num(cw,nan=np.nanmedian(cw));gy,gx=np.gradient(cw);Jxx=(gx*gx).mean();Jyy=(gy*gy).mean();Jxy=(gx*gy).mean();den=Jxx+Jyy
    coh=math.sqrt((Jxx-Jyy)**2+4*Jxy*Jxy)/den if den>1e-12 else 0.0
    if coh<=0.70: continue
    w=window(e,n,hw)
    sc=0.0
    if w is not None and w.shape==(2*hw,2*hw) and np.isnan(w).mean()<0.05:
        d2=downs(np.nan_to_num(w,nan=np.nanmedian(w)),f);h=hs(d2,CS*f);lo2,hi2=np.percentile(h,2),np.percentile(h,98)
        raw=np.asarray(Image.fromarray(np.clip((h-lo2)/(hi2-lo2+1e-9)*255,0,255).astype('uint8')).resize((128,128)),np.uint8)
        with torch.no_grad(): sc=float(torch.sigmoid(net(torch.tensor(homog(raw)[None,None],dtype=torch.float32).to(dev)/255.)).item())
    cut.append((lo,la,src,sc,coh,e,n))
cut.sort(key=lambda x:-x[3])
print(f"candidați broad tăiați de coh>0.70: {len(cut)}",flush=True)
for lo,la,src,sc,coh,e,n in cut: print(f"  {src:16} {la:.5f},{lo:.5f}  scor={sc:.2f}  coh={coh:.2f}",flush=True)
# board hillshade (fereastră 80m centrată) pt cei tăiați cu scor>=0.7
show=[c for c in cut if c[3]>=0.7][:24]
if show:
    cols=6;rows=(len(show)+cols-1)//cols;cell=150;lab=16
    bd=Image.new('RGB',(cols*cell,rows*(cell+lab)),(20,20,20));dr=ImageDraw.Draw(bd)
    try: fnt=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial Bold.ttf',12)
    except: fnt=ImageFont.load_default()
    for i,(lo,la,src,sc,coh,e,n) in enumerate(show):
        w=window(e,n,int(40/CS))
        if w is None: continue
        h=hs(np.nan_to_num(w,nan=np.nanmedian(w)),CS);im=np.clip((h-np.percentile(h,2))/(np.percentile(h,98)-np.percentile(h,2)+1e-9)*255,0,255).astype('uint8')
        thumb=Image.fromarray(im).resize((cell,cell));r0=i//cols;c0=i%cols;x=c0*cell;y=r0*(cell+lab)
        bd.paste(thumb,(x,y+lab));dr.text((x+2,y+2),f"{src[:6]} s{sc:.2f} c{coh:.2f}",fill=(120,255,120),font=fnt)
    bd.save(f"{H}/review/ro_cut_check.png");print(f"-> review/ro_cut_check.png ({len(show)} tăiați cu scor>=0.7)",flush=True)
