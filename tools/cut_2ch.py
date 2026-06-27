#!/usr/bin/env python3
# cut_2ch.py IN_CSV OUT_NPZ
# Pt fiecare coord (lon,lat[,label]) taie o stampa 2-canal din LAKI3 0.5m (cache /tmp/laki3, download-on-miss):
#   canal 0 = HILLSHADE multidir (RECETA IDENTICA stage-1: win 80m, downsample 2m, 6-dir, stretch 2-98, resize128, homog)
#   canal 1 = SLRM ADEVARAT (elev - boxblur(elev,30m) pe DEM 2m, stretch 2-98, resize128, homog) — independent de iluminare
# Salveaza OUT_NPZ: X uint8 (N,2,128,128), y int (1 pos / 0 neg / -1 necunoscut), lon, lat, ok(bool). Batch gdaltransform.
import os,sys,subprocess,math,csv,zipfile
import numpy as np
from PIL import Image,ImageFilter
H=os.path.dirname(os.path.dirname(os.path.abspath(__file__)));CACHE="/tmp/laki3";CS=0.5;TPX=2000
os.makedirs(CACHE,exist_ok=True)
APP=os.environ.get("QGIS_APP","/Applications/QGIS-final-4_0_3.app/Contents")
ENV=dict(os.environ,DYLD_FRAMEWORK_PATH=f"{APP}/Frameworks",PROJ_DATA=f"{APP}/Resources/qgis/proj",PROJ_LIB=f"{APP}/Resources/qgis/proj",GDAL_DATA=f"{APP}/Resources/qgis/gdal")
GTb=f"{APP}/MacOS/gdaltransform"
def trans_batch(pts,s,t):
    inp="\n".join(f"{a} {b}" for a,b in pts)+"\n"
    r=subprocess.run([GTb,"-s_srs",s,"-t_srs",t],input=inp,capture_output=True,text=True,env=ENV)
    out=[]
    for l in r.stdout.strip().split("\n"):
        p=l.split();out.append((float(p[0]),float(p[1])) if len(p)>=2 else (None,None))
    return out
def load_one(nkm,ekm):
    NUME=f"{nkm}_{ekm}";npy=f"{CACHE}/{NUME}.npy"
    if os.path.exists(npy):
        try:return np.load(npy)
        except:pass
    z=f"{CACHE}/{NUME}.zip"
    if not os.path.exists(z):
        subprocess.run(["curl","-s","--max-time","120","-o",z,f"https://geoportal.ancpi.ro/laki3_mnt/zip/{NUME}.zip"],check=False)
    try: zf=zipfile.ZipFile(z)
    except:
        if os.path.exists(z):os.remove(z)
        return None
    asc=[n for n in zf.namelist() if n.lower().endswith('.asc')]
    if not asc: return None
    raw=zf.read(asc[0]).decode('latin-1').replace(',','.')
    lines=raw.split('\n');hdr={};i=0
    while i<len(lines):
        p=lines[i].split()
        if len(p)>=2 and p[0].lower() in ('ncols','nrows','xllcorner','yllcorner','cellsize','nodata_value'):
            hdr[p[0].lower()]=float(p[1]);i+=1
        else:break
    nc=int(hdr['ncols']);nr=int(hdr['nrows']);nd=hdr.get('nodata_value',-9999)
    data=np.fromstring(' '.join(lines[i:]),sep=' ',dtype=np.float32)[:nc*nr].reshape(nr,nc)
    data[data==nd]=np.nan;np.save(npy,data);return data
def downs(a,fc):
    Hh,Ww=a.shape;return a[:Hh//fc*fc,:Ww//fc*fc].reshape(Hh//fc,fc,Ww//fc,fc).mean((1,3))
def hs(dem,cs,azs=(315,45,135,225,270,0),alt=35):
    gy,gx=np.gradient(dem,cs);sl=np.arctan(np.hypot(gx,gy));asp=np.arctan2(-gy,gx);o=np.zeros_like(dem);ar=math.radians(alt)
    for az in azs: azr=math.radians(360-az+90);o+=np.clip(np.sin(ar)*np.cos(sl)+np.cos(ar)*np.sin(sl)*np.cos(azr-asp),0,1)
    return o/len(azs)
def boxblur1(a,r):
    Hh,Ww=a.shape;ii=np.zeros((Hh+1,Ww+1),np.float64);ii[1:,1:]=np.cumsum(np.cumsum(a,0),1)
    ys=np.arange(Hh);xs=np.arange(Ww);y0=np.clip(ys-r,0,Hh);y1=np.clip(ys+r+1,0,Hh);x0=np.clip(xs-r,0,Ww);x1=np.clip(xs+r+1,0,Ww)
    A=ii[y1][:,x1]-ii[y0][:,x1]-ii[y1][:,x0]+ii[y0][:,x0];cnt=((y1-y0)[:,None]*(x1-x0)[None,:]).astype(np.float64)
    return (A/cnt).astype(np.float32)
def homog(a):
    a2=np.asarray(Image.fromarray(a).filter(ImageFilter.GaussianBlur(0.8)),np.uint8);cdf=np.bincount(a2.ravel(),minlength=256).astype(np.float64).cumsum()
    return (cdf[a2]/cdf[-1]*255).astype(np.uint8) if cdf[-1]>0 else a2
def stretch128(field):
    lo,hi=np.percentile(field,2),np.percentile(field,98)
    if hi-lo<1e-9: return None
    u=np.clip((field-lo)/(hi-lo)*255,0,255).astype('uint8')
    return homog(np.asarray(Image.fromarray(u).resize((128,128)),np.uint8))
HW=int(40/CS)  # 80m window @0.5m
def window(est,nord):
    e0=int((est-50)//1000);e1=int((est+50)//1000);n0=int((nord-50)//1000);n1=int((nord+50)//1000)
    xll0=e0*1000;ytop0=(n1+1)*1000;W=(e1-e0+1)*TPX;Hh=(n1-n0+1)*TPX
    mos=np.full((Hh,W),np.nan,np.float32);got=0
    for nk in range(n0,n1+1):
        for ek in range(e0,e1+1):
            d=load_one(nk,ek)
            if d is None:continue
            got+=1;ox=int((ek*1000-xll0)/CS);oy=int((ytop0-(nk+1)*1000)/CS);mos[oy:oy+TPX,ox:ox+TPX]=d[:TPX,:TPX]
    if got==0:return None
    px=int((est-xll0)/CS);py=int((ytop0-nord)/CS)
    return mos[py-HW:py+HW,px-HW:px+HW]
def stamp2ch(est,nord):
    w=window(est,nord)
    if w is None or w.shape!=(2*HW,2*HW) or np.isnan(w).mean()>0.05:return None
    w=np.nan_to_num(w,nan=np.nanmedian(w));f=int(round(2.0/CS));z=downs(w,f);cs2=CS*f
    ch0=stretch128(hs(z,cs2))
    slrm=z-boxblur1(boxblur1(boxblur1(z,int(30/cs2)),int(30/cs2)),int(30/cs2))
    ch1=stretch128(slrm)
    if ch0 is None or ch1 is None:return None
    return np.stack([ch0,ch1])
def main():
    inp=sys.argv[1];outp=sys.argv[2];rows=list(csv.DictReader(open(inp)))
    cols={c.lower():c for c in rows[0].keys()};lonc=cols.get('lon');latc=cols.get('lat')
    pts=[(float(r[lonc]),float(r[latc])) for r in rows]
    print(f"{len(pts)} puncte; transform...",flush=True);st=trans_batch(pts,"EPSG:4326","EPSG:3844")
    X=[];y=[];lons=[];lats=[];ok=[]
    for i,(r,(e,n)) in enumerate(zip(rows,st)):
        s=stamp2ch(e,n) if e is not None else None
        lab=r.get('label','').lower();yy=1 if lab.startswith('pos') else (0 if lab.startswith('neg') else -1)
        if s is None:
            X.append(np.zeros((2,128,128),np.uint8));ok.append(False)
        else:
            X.append(s);ok.append(True)
        y.append(yy);lons.append(float(r[lonc]));lats.append(float(r[latc]))
        if (i+1)%100==0:print(f"  {i+1}/{len(pts)} (ok {sum(ok)})",flush=True)
    np.savez_compressed(outp,X=np.array(X,np.uint8),y=np.array(y),lon=np.array(lons),lat=np.array(lats),ok=np.array(ok))
    print(f"-> {outp} | {sum(ok)}/{len(pts)} ok",flush=True)
if __name__=='__main__':main()
