#!/usr/bin/env python3
# dtm_slrm.py LON LAT OUT [METERS=300]
# SLRM ADEVARAT din DTM brut (inaltime), independent de iluminare.
# Sursa: ANCPI LAKI III MNT 0.5 m, download direct dala Stereo70 (NUME=Nord_km_Est_km).
# Pipeline: lon/lat -> Stereo70 (gdaltransform) -> dala -> download/cache -> parse .asc (virgula RO)
#           -> fereastra METERS -> SLRM = elev - GaussianBlur(elev, ~mound*1.5) -> stretch 2-98.
import sys,os,subprocess,math,zipfile
import numpy as np
from PIL import Image,ImageFilter
LON=float(sys.argv[1]); LAT=float(sys.argv[2]); OUT=sys.argv[3]
METERS=float(sys.argv[4]) if len(sys.argv)>4 else 300.0
MODE=sys.argv[5] if len(sys.argv)>5 else 'hs'   # hs=hillshade DTM (citibil) | slrm
APP=os.environ.get("QGIS_APP","/Applications/QGIS-final-4_0_3.app/Contents")
ENV=dict(os.environ, DYLD_FRAMEWORK_PATH=f"{APP}/Frameworks", PROJ_DATA=f"{APP}/Resources/qgis/proj",
         PROJ_LIB=f"{APP}/Resources/qgis/proj", GDAL_DATA=f"{APP}/Resources/qgis/gdal")
GT=f"{APP}/MacOS/gdaltransform"
CACHE="/tmp/laki3"; os.makedirs(CACHE,exist_ok=True)
def to_stereo(lon,lat):
    r=subprocess.run([GT,"-s_srs","EPSG:4326","-t_srs","EPSG:3844"],input=f"{lon} {lat}\n",
                     capture_output=True,text=True,env=ENV)
    e,n,_=r.stdout.split(); return float(e),float(n)
CS=0.5; TPX=2000  # 0.5 m, 2000 px/km
def boxblur1(a,r):
    H,W=a.shape
    ii=np.zeros((H+1,W+1),dtype=np.float64); ii[1:,1:]=np.cumsum(np.cumsum(a,axis=0),axis=1)
    ys=np.arange(H); xs=np.arange(W)
    y0=np.clip(ys-r,0,H); y1=np.clip(ys+r+1,0,H); x0=np.clip(xs-r,0,W); x1=np.clip(xs+r+1,0,W)
    A=ii[y1][:,x1]-ii[y0][:,x1]-ii[y1][:,x0]+ii[y0][:,x0]
    cnt=((y1-y0)[:,None]*(x1-x0)[None,:]).astype(np.float64)
    return (A/cnt).astype(np.float32)
def boxblur3(a,r):
    r=max(1,int(r*0.6))
    for _ in range(3): a=boxblur1(a,r)
    return a
def hillshade(dem,cs,azs=(315,45,135,225,270,0),alt=35):
    gy,gx=np.gradient(dem,cs); slope=np.arctan(np.hypot(gx,gy)); aspect=np.arctan2(-gy,gx)
    out=np.zeros_like(dem); altr=math.radians(alt)
    for az in azs:
        azr=math.radians(360-az+90)
        out+=np.clip(np.sin(altr)*np.cos(slope)+np.cos(altr)*np.sin(slope)*np.cos(azr-aspect),0,1)
    return out/len(azs)
def load_one(nkm,ekm):
    NUME=f"{nkm}_{ekm}"; npy=f"{CACHE}/{NUME}.npy"
    if os.path.exists(npy): return np.load(npy)
    z=f"{CACHE}/{NUME}.zip"
    if not os.path.exists(z):
        subprocess.run(["curl","-s","--max-time","90","-o",z,f"https://geoportal.ancpi.ro/laki3_mnt/zip/{NUME}.zip"],check=False)
    try: zf=zipfile.ZipFile(z)
    except: return None
    asc=[n for n in zf.namelist() if n.lower().endswith('.asc')]
    if not asc: return None
    raw=zf.read(asc[0]).decode('latin-1').replace(',','.')
    lines=raw.split('\n'); hdr={}; i=0
    while i<len(lines):
        p=lines[i].split()
        if len(p)>=2 and p[0].lower() in ('ncols','nrows','xllcorner','yllcorner','cellsize','nodata_value'):
            hdr[p[0].lower()]=float(p[1]); i+=1
        else: break
    nc=int(hdr['ncols']); nr=int(hdr['nrows']); nd=hdr.get('nodata_value',-9999)
    data=np.fromstring(' '.join(lines[i:]),sep=' ',dtype=np.float32)[:nc*nr].reshape(nr,nc)
    data[data==nd]=np.nan; np.save(npy,data); return data
def load_block(est,nord,half_m):
    # mozaic peste dalele care intersecteaza fereastra
    emin=est-half_m; emax=est+half_m; nmin=nord-half_m; nmax=nord+half_m
    e0=int(emin//1000); e1=int(emax//1000); n0=int(nmin//1000); n1=int(nmax//1000)
    xll0=e0*1000; ytop0=(n1+1)*1000
    Wt=(e1-e0+1); Ht=(n1-n0+1)
    canvas=np.full((Ht*TPX,Wt*TPX),np.nan,dtype=np.float32); got=[]
    for nkm in range(n0,n1+1):
        for ekm in range(e0,e1+1):
            d=load_one(nkm,ekm)
            if d is None: continue
            got.append(f"{nkm}_{ekm}")
            ox=(ekm*1000-xll0)/CS; oy=(ytop0-(nkm+1)*1000)/CS
            canvas[int(oy):int(oy)+TPX,int(ox):int(ox)+TPX]=d[:TPX,:TPX]
    return canvas,xll0,ytop0,got
def slrm(lon,lat,meters,out=600):
    est,nord=to_stereo(lon,lat)
    half=int((meters/2)/CS); marg=int(80/CS); hm=(meters/2)+80
    canvas,xll0,ytop0,got=load_block(est,nord,hm)
    if not got: return None,"no_tile"
    nr,nc=canvas.shape; cs=CS
    col=(est-xll0)/cs; row=(ytop0-nord)/cs
    c0=int(col-half-marg); c1=int(col+half+marg); r0=int(row-half-marg); r1=int(row+half+marg)
    if c0<0 or r0<0 or c1>nc or r1>nr: return None,f"edge({'+'.join(got)})"
    win=canvas[r0:r1,c0:c1].copy()
    NUME='+'.join(got)
    if np.isnan(win).mean()>0.3: return None,f"gap({NUME})"  # gaura de acoperire
    m=np.nanmean(win); win=np.where(np.isnan(win),m,win).astype(np.float32)
    if MODE=='hs':
        field=hillshade(win,cs)
    else:
        field=win-boxblur3(win,int(30/cs))  # SLRM: relief local
    field=field[marg:marg+2*half,marg:marg+2*half]
    lo,hi=np.percentile(field,2),np.percentile(field,98)
    sl=np.clip((field-lo)/(hi-lo+1e-6)*255,0,255).astype('uint8')
    return Image.fromarray(sl).resize((out,out)),NUME
img,info=slrm(LON,LAT,METERS)
if img is None: print("FAIL:",info); sys.exit(2)
img.save(OUT); print(f"SLRM-DTM saved: {OUT} | dala {info} | {METERS:.0f} m | 0.5m DTM brut")
