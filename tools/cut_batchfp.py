#!/usr/bin/env python3
# cut_batchfp.py — taie stampe hard-negative din coordonatele FP marcate de Andrei (labeled/batchN_hardneg.csv),
# recipe IDENTIC cu extract_marked neg_stamp (80m -> downsample 2m -> hillshade 6-dir -> percentile 2-98 -> 128 RAW).
# EXCLUDE zona Catane (test held-out) anti-leakage. Per-punct: încarcă DOAR tile-urile locale (memory-safe).
# -> dataset_neg_batchmarks/batchfp_NNNNN.png + manifest.
import os,sys,math,subprocess,csv,glob
import numpy as np
from PIL import Image
H=os.path.expanduser('~/lidar-match');CACHE="/tmp/laki3";CS=0.5;TPX=2000
OUT=f"{H}/dataset_neg_batchmarks";os.makedirs(OUT,exist_ok=True)
APP="/Applications/QGIS-final-4_0_3.app/Contents"
ENV=dict(os.environ,DYLD_FRAMEWORK_PATH=f"{APP}/Frameworks",PROJ_DATA=f"{APP}/Resources/qgis/proj",PROJ_LIB=f"{APP}/Resources/qgis/proj",GDAL_DATA=f"{APP}/Resources/qgis/gdal")
GT=f"{APP}/MacOS/gdaltransform"
# Catane held-out box (excludem) — centru 23.4181/43.9141, marjă generoasă
CAT=(23.36,43.86,23.48,43.97)  # lon0,lat0,lon1,lat1
def in_catane(lo,la): return CAT[0]<=lo<=CAT[2] and CAT[1]<=la<=CAT[3]
# 1) adună coords FP
coords=[]
for f in sorted(glob.glob(f'{H}/labeled/batch*_hardneg.csv')):
    for r in csv.DictReader(open(f)):
        lo,la=float(r['lon']),float(r['lat'])
        if in_catane(lo,la): continue
        coords.append((lo,la,os.path.basename(f).split('_')[0],r['idx']))
print(f"FP totale (excl. Catane): {len(coords)}",flush=True)
# 2) transform batch -> est,nord
inp="".join(f"{lo} {la}\n" for lo,la,_,_ in coords)
r=subprocess.run([GT,"-s_srs","EPSG:4326","-t_srs","EPSG:3844"],input=inp,capture_output=True,text=True,env=ENV)
EN=[tuple(map(float,l.split()[:2])) for l in r.stdout.strip().split("\n")]
def hs(dem,cs,azs=(315,45,135,225,270,0),alt=35):
    gy,gx=np.gradient(dem,cs);sl=np.arctan(np.hypot(gx,gy));asp=np.arctan2(-gy,gx);o=np.zeros_like(dem);ar=math.radians(alt)
    for az in azs: azr=math.radians(360-az+90);o+=np.clip(np.sin(ar)*np.cos(sl)+np.cos(ar)*np.sin(sl)*np.cos(azr-asp),0,1)
    return o/len(azs)
def downs(a,fc):
    Hh,Ww=a.shape;return a[:Hh//fc*fc,:Ww//fc*fc].reshape(Hh//fc,fc,Ww//fc,fc).mean((1,3))
_tc={}
def get_tile(nk,ek):
    k=(nk,ek)
    if k not in _tc:
        if len(_tc)>40: _tc.clear()
        p=f"{CACHE}/{nk}_{ek}.npy";_tc[k]=np.load(p) if os.path.exists(p) else None
    return _tc[k]
f=int(round(2.0/CS));wpx=int(80/CS)  # 160px window, downsample factor 4 -> 2m
def neg_stamp(est,nord):
    eks=sorted({int((est-50)//1000),int((est+50)//1000)});nks=sorted({int((nord-50)//1000),int((nord+50)//1000)})
    e0=min(eks);n1=max(nks);xll=e0*1000;ytop=(n1+1)*1000
    W=(max(eks)-e0+1)*TPX;Hh=(n1-min(nks)+1)*TPX
    mos=np.full((Hh,W),np.nan,np.float32);got=False
    for nk in nks:
        for ek in eks:
            d=get_tile(nk,ek)
            if d is None: continue
            got=True;ox=int((ek*1000-xll)/CS);oy=int((ytop-(nk+1)*1000)/CS);mos[oy:oy+TPX,ox:ox+TPX]=d[:TPX,:TPX]
    if not got: return None
    px=int((est-xll)/CS);py=int((ytop-nord)/CS);w=mos[py-wpx//2:py+wpx//2,px-wpx//2:px+wpx//2]
    if w.shape!=(wpx,wpx) or np.isnan(w).mean()>0.05: return None
    d2=downs(np.nan_to_num(w,nan=np.nanmedian(w)),f);h=hs(d2,CS*f);lo,hi=np.percentile(h,2),np.percentile(h,98)
    if hi-lo<1e-6: return None
    return np.asarray(Image.fromarray(np.clip((h-lo)/(hi-lo)*255,0,255).astype('uint8')).resize((128,128)),np.uint8)
mf=open(f'{OUT}/manifest.csv','w');mw=csv.writer(mf);mw.writerow(['file','est','nord','lon','lat','batch','idx']);nw=0;sk=0
for (lo,la,bt,ix),(e,n) in zip(coords,EN):
    st=neg_stamp(e,n)
    if st is None: sk+=1; continue
    fn=f"batchfp_{nw:05d}.png";Image.fromarray(st).save(f"{OUT}/{fn}");mw.writerow([f"dataset_neg_batchmarks/{fn}",f"{e:.1f}",f"{n:.1f}",lo,la,bt,ix]);nw+=1
    if nw%200==0: print(f"  {nw} tăiate...",flush=True)
mf.close()
print(f"GATA: {nw} stampe hard-neg scrise în dataset_neg_batchmarks/ ({sk} skip acoperire/edge)",flush=True)
