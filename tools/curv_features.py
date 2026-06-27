#!/usr/bin/env python3
# curv_features.py IN_CSV OUT_CSV
# Extrage trasaturi de CURBURA (din elevatie DTM LAKI3 0.5m, cache /tmp/laki3, download-on-miss)
# pentru fiecare coord din IN_CSV (trebuie sa aiba coloane lon,lat; restul coloanelor sunt pastrate).
# Trasaturi (DEM 2m, cap R=25m in jurul apexului din central 40m):
#   convex   = Laplacian mediu in cap (dom convex-up => NEGATIV).  Tumul real: puternic negativ.
#   convex_in= Laplacian mediu in cap interior R=12m (miezul domului).
#   rugoz    = std(Laplacian in cap)/relief.  Tumul = neted (mic); mușuroi natural = aspru (mare).
#   asim     = simetrie radiala (std azimutal elevatie pe inele)/relief.  Dom artificial = simetric (mic).
#   relief_m = inaltime apex peste percentila 10 in cap (m).
#   slrm_pk  = prominenta SLRM la apex (elev - blur(elev, 30m)) / relief.  Tumul = blob pozitiv curat.
#   mono     = fractia de raze pe care profilul radial scade monoton de la apex (dom => ~1).
# Batch gdaltransform (un singur apel). Fara scipy.
import os,sys,math,subprocess,csv,zipfile
import numpy as np
H=os.path.expanduser('~/lidar-match');CACHE="/tmp/laki3";CS=0.5;TPX=2000
os.makedirs(CACHE,exist_ok=True)
APP="/Applications/QGIS-final-4_0_3.app/Contents"
ENV=dict(os.environ,DYLD_FRAMEWORK_PATH=f"{APP}/Frameworks",PROJ_DATA=f"{APP}/Resources/qgis/proj",PROJ_LIB=f"{APP}/Resources/qgis/proj",GDAL_DATA=f"{APP}/Resources/qgis/gdal")
GTb=f"{APP}/MacOS/gdaltransform"
def trans_batch(pts,s,t):
    inp="\n".join(f"{a} {b}" for a,b in pts)+"\n"
    r=subprocess.run([GTb,"-s_srs",s,"-t_srs",t],input=inp,capture_output=True,text=True,env=ENV)
    out=[]
    for l in r.stdout.strip().split("\n"):
        p=l.split()
        out.append((float(p[0]),float(p[1])) if len(p)>=2 else (None,None))
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
def boxblur1(a,r):
    Hh,Ww=a.shape;ii=np.zeros((Hh+1,Ww+1),np.float64);ii[1:,1:]=np.cumsum(np.cumsum(a,0),1)
    ys=np.arange(Hh);xs=np.arange(Ww);y0=np.clip(ys-r,0,Hh);y1=np.clip(ys+r+1,0,Hh);x0=np.clip(xs-r,0,Ww);x1=np.clip(xs+r+1,0,Ww)
    A=ii[y1][:,x1]-ii[y0][:,x1]-ii[y1][:,x0]+ii[y0][:,x0];cnt=((y1-y0)[:,None]*(x1-x0)[None,:]).astype(np.float64)
    return (A/cnt).astype(np.float32)
WIN=100.0;HALF=int(WIN/CS)
def local_window(est,nord):
    e0=int((est-WIN)//1000);e1=int((est+WIN)//1000);n0=int((nord-WIN)//1000);n1=int((nord+WIN)//1000)
    xll0=e0*1000;ytop0=(n1+1)*1000;W=(e1-e0+1)*TPX;Hh=(n1-n0+1)*TPX
    mos=np.full((Hh,W),np.nan,np.float32);got=0
    for nk in range(n0,n1+1):
        for ek in range(e0,e1+1):
            d=load_one(nk,ek)
            if d is None:continue
            got+=1;ox=int((ek*1000-xll0)/CS);oy=int((ytop0-(nk+1)*1000)/CS);mos[oy:oy+TPX,ox:ox+TPX]=d[:TPX,:TPX]
    if got==0:return None
    px=int((est-xll0)/CS);py=int((ytop0-nord)/CS)
    return mos[py-HALF:py+HALF,px-HALF:px+HALF]
def analyze(est,nord):
    w=local_window(est,nord)
    if w is None or w.shape!=(2*HALF,2*HALF) or np.isnan(w).mean()>0.15:return None
    w=np.nan_to_num(w,nan=np.nanmedian(w))
    f=int(round(2.0/CS));z=downs(w,f);cs2=CS*f
    c0=z.shape[0]//2;rr=int(20/cs2);sub=z[c0-rr:c0+rr,c0-rr:c0+rr]
    off=np.unravel_index(np.argmax(sub),sub.shape);ay=c0-rr+off[0];ax=c0-rr+off[1]
    lap=(np.roll(z,1,0)+np.roll(z,-1,0)+np.roll(z,1,1)+np.roll(z,-1,1)-4*z)/(cs2*cs2)
    R=int(25/cs2);Rin=int(12/cs2);ys,xs=np.mgrid[0:z.shape[0],0:z.shape[1]];rad=np.hypot(ys-ay,xs-ax)
    cap=rad<=R;capin=rad<=Rin
    relief=float(z[cap].max()-np.percentile(z[cap],10))+1e-6
    convex=float(lap[cap].mean());convex_in=float(lap[capin].mean())
    rugoz=float(lap[cap].std()/relief)
    asim=[]
    for r0 in range(2,R,2):
        ring=(rad>=r0)&(rad<r0+2)&cap
        if ring.sum()>=6:asim.append(z[ring].std())
    asimv=float(np.mean(asim)/relief) if asim else 9.9
    # SLRM prominenta la apex (relief local fata de fundal 30m)
    bg=boxblur1(boxblur1(boxblur1(z,int(30/cs2)),int(30/cs2)),int(30/cs2));slrm=z-bg
    slrm_pk=float(slrm[capin].mean()/relief)
    # monotonie radiala: medie pe inele scade de la apex?
    prof=[]
    for r0 in range(0,R,1):
        ring=(rad>=r0)&(rad<r0+1)&cap
        if ring.sum()>=3:prof.append(z[ring].mean())
    prof=np.array(prof);mono=float(np.mean(np.diff(prof)<=0)) if len(prof)>2 else 0.0
    return dict(convex=convex,convex_in=convex_in,rugoz=rugoz,asim=asimv,relief_m=relief,slrm_pk=slrm_pk,mono=mono)
def main():
    inp=sys.argv[1];outp=sys.argv[2]
    rows=list(csv.DictReader(open(inp)))
    # detect lon/lat columns (case-insensitive)
    cols={c.lower():c for c in rows[0].keys()}
    lonc=cols.get('lon') or cols.get('longitude');latc=cols.get('lat') or cols.get('latitude')
    pts=[(float(r[lonc]),float(r[latc])) for r in rows]
    print(f"{len(pts)} puncte; transform batch...",flush=True)
    st=trans_batch(pts,"EPSG:4326","EPSG:3844")
    featcols=['convex','convex_in','rugoz','asim','relief_m','slrm_pk','mono']
    extra=[c for c in rows[0].keys()]
    with open(outp,'w',newline='') as fo:
        wr=csv.writer(fo);wr.writerow(extra+featcols)
        ok=0;na=0
        for i,(r,(e,n)) in enumerate(zip(rows,st)):
            a=analyze(e,n) if e is not None else None
            if a is None:
                na+=1;wr.writerow([r.get(c,'') for c in extra]+['NA']*len(featcols))
            else:
                ok+=1;wr.writerow([r.get(c,'') for c in extra]+[f"{a[k]:.5f}" for k in featcols])
            if (i+1)%100==0:print(f"  {i+1}/{len(pts)} (ok {ok}, NA {na})",flush=True)
        print(f"-> {outp} | ok {ok} | NA {na}",flush=True)
if __name__=='__main__':main()
