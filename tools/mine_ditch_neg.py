#!/usr/bin/env python3
# mine_ditch_neg.py CLON CLAT KM [N] [prefix] — minează AUTOMAT negative-șanț/arătură/canal.
# Criteriu de FORMĂ (sigur, fără otravă ca mining-ul pe scor): coerență direcțională mare SAU liniaritate mare
# = trăsătură liniară (șanț/dig/furrow) = NICIODATĂ tumul (compact). Exclud 150m de tumuli known (excludere
# suplimentară; forma oricum exclude tumulii necunoscuți). Cut recipe identic dataset_neg -> dataset_neg_ditch/.
import os,sys,math,subprocess,zipfile,csv,json,glob
import numpy as np
from PIL import Image
H=os.path.expanduser('~/lidar-match');CACHE="/tmp/laki3";CS=0.5;TPX=2000
CLON=float(sys.argv[1]);CLAT=float(sys.argv[2]);KM=float(sys.argv[3]) if len(sys.argv)>3 else 6.0
NMAX=int(sys.argv[4]) if len(sys.argv)>4 else 1500
PREFIX=sys.argv[5] if len(sys.argv)>5 else 'z'
COH_T=0.65; LIN_T=15.0; ENERGY_MIN=0.025   # praguri SIGURE (tumul compact nu le atinge)
APP="/Applications/QGIS-final-4_0_3.app/Contents"
ENV=dict(os.environ,DYLD_FRAMEWORK_PATH=f"{APP}/Frameworks",PROJ_DATA=f"{APP}/Resources/qgis/proj",PROJ_LIB=f"{APP}/Resources/qgis/proj",GDAL_DATA=f"{APP}/Resources/qgis/gdal")
GT=f"{APP}/MacOS/gdaltransform"
def trans(pts,s,t):
    if not pts: return []
    inp="\n".join(f"{a} {b}" for a,b in pts)+"\n"
    r=subprocess.run([GT,"-s_srs",s,"-t_srs",t],input=inp,capture_output=True,text=True,env=ENV)
    return [(float(l.split()[0]),float(l.split()[1])) for l in r.stdout.strip().split("\n") if l.split()]
def to_st(lo,la): return trans([(lo,la)],"EPSG:4326","EPSG:3844")[0]
def dl(nk,ek):
    p=f"{CACHE}/{nk}_{ek}.npy"
    if os.path.exists(p): return np.load(p)
    z=f"{CACHE}/{nk}_{ek}.zip"
    if not os.path.exists(z): subprocess.run(["curl","-s","--max-time","60","-o",z,f"https://geoportal.ancpi.ro/laki3_mnt/zip/{nk}_{ek}.zip"],check=False)
    try: zf=zipfile.ZipFile(z);asc=[n for n in zf.namelist() if n.lower().endswith('.asc')]
    except: return None
    if not asc: return None
    raw=zf.read(asc[0]).decode('latin-1').replace(',','.');L=raw.split('\n');i=0
    while i<len(L) and L[i].split() and L[i].split()[0].lower() in('ncols','nrows','xllcorner','yllcorner','cellsize','nodata_value'):i+=1
    d=np.fromstring(' '.join(L[i:]),sep=' ',dtype=np.float32)[:TPX*TPX].reshape(TPX,TPX);d[d==-9999]=np.nan;np.save(p,d);return d
def hs(dem,cs,azs=(315,45,135,225,270,0),alt=35):
    gy,gx=np.gradient(dem,cs);sl=np.arctan(np.hypot(gx,gy));asp=np.arctan2(-gy,gx);o=np.zeros_like(dem);ar=math.radians(alt)
    for az in azs: azr=math.radians(360-az+90);o+=np.clip(np.sin(ar)*np.cos(sl)+np.cos(ar)*np.sin(sl)*np.cos(azr-asp),0,1)
    return o/len(azs)
def downs(a,fc):
    Hh,Ww=a.shape;return a[:Hh//fc*fc,:Ww//fc*fc].reshape(Hh//fc,fc,Ww//fc,fc).mean((1,3))
def boxblur(a,r):
    ii=np.zeros((a.shape[0]+1,a.shape[1]+1));ii[1:,1:]=np.cumsum(np.cumsum(a,0),1)
    H2,W2=a.shape;ys=np.arange(H2);xs=np.arange(W2)
    y0=np.clip(ys-r,0,H2);y1=np.clip(ys+r+1,0,H2);x0=np.clip(xs-r,0,W2);x1=np.clip(xs+r+1,0,W2)
    return (ii[y1][:,x1]-ii[y0][:,x1]-ii[y1][:,x0]+ii[y0][:,x0])/((y1-y0)[:,None]*(x1-x0)[None,:])
# movile known de exclus
known=[]
if os.path.exists('/tmp/dolj_sweep_plan.json'):
    for lo,la,*_ in json.load(open('/tmp/dolj_sweep_plan.json'))['all']: known.append((lo,la))
for r in csv.DictReader(open(f'{H}/labeled/labels.csv')):
    if r.get('verdict')=='mound':
        try: known.append((float(r['lon']),float(r['lat'])))
        except: pass
est,nord=to_st(CLON,CLAT);half=KM*1000/2
e0=int((est-half)//1000);e1=int((est+half)//1000);n0=int((nord-half)//1000);n1=int((nord+half)//1000)
xll0=e0*1000;ytop0=(n1+1)*1000;W=(e1-e0+1)*TPX;Hh=(n1-n0+1)*TPX
mos=np.full((Hh,W),np.nan,np.float32);got=0
for nk in range(n0,n1+1):
    for ek in range(e0,e1+1):
        d=dl(nk,ek)
        if d is None: continue
        got+=1;ox=int((ek*1000-xll0)/CS);oy=int((ytop0-(nk+1)*1000)/CS);mos[oy:oy+TPX,ox:ox+TPX]=d[:TPX,:TPX]
print(f"mozaic {W}x{Hh} ({KM}km) {got} dale, goluri {np.isnan(mos).mean()*100:.0f}%",flush=True)
knownst=trans(known,"EPSG:4326","EPSG:3844") if known else []
def near_known(e,n,d=150): return any((e-a)**2+(n-b)**2<d*d for a,b in knownst)
f=int(round(2.0/CS));wpx=int(80/CS);HALF=wpx//2;step=int(50/CS)
def coh_lin(z):
    gy,gx=np.gradient(z,1.0)
    Jxx=(gx*gx).mean();Jyy=(gy*gy).mean();Jxy=(gx*gy).mean()
    coh=math.sqrt((Jxx-Jyy)**2+4*Jxy*Jxy)/(Jxx+Jyy+1e-9)
    energy=float(np.sqrt(gx*gx+gy*gy).mean())
    return coh,energy
def lin_ratio(w2m):
    slrm=w2m-boxblur(w2m,int(40/2))  # SLRM scară mare pe 2m
    thr=slrm.mean()+1.0*slrm.std();mask=slrm>thr
    ys,xs=np.nonzero(mask)
    if len(xs)<10: return 1.0
    cx,cy=xs.mean(),ys.mean();mxx=((xs-cx)**2).mean();myy=((ys-cy)**2).mean();mxy=((xs-cx)*(ys-cy)).mean()
    tr=mxx+myy;dd=tr*tr/4-(mxx*myy-mxy*mxy);s=math.sqrt(max(0,dd))
    return math.sqrt((tr/2+s)/max(tr/2-s,1e-6))
os.makedirs(f'{H}/dataset_neg_ditch',exist_ok=True)
mf=open(f'{H}/dataset_neg_ditch/manifest_{PREFIX}.csv','w');mw=csv.writer(mf);mw.writerow(['file','est','nord','coh','lin'])
kept=[];n=0;scanned=0
ys=list(range(HALF,Hh-HALF,step));xs=list(range(HALF,W-HALF,step))
for py in ys:
    if n>=NMAX: break
    for px in xs:
        if n>=NMAX: break
        w=mos[py-HALF:py+HALF,px-HALF:px+HALF]
        if w.shape!=(wpx,wpx) or np.isnan(w).mean()>0.05: continue
        scanned+=1
        z=downs(np.nan_to_num(w,nan=np.nanmedian(w)),f)
        coh,energy=coh_lin(z)
        if energy<ENERGY_MIN: continue
        lr=lin_ratio(z)
        if not (coh>=COH_T or lr>=LIN_T): continue
        e=xll0+px*CS;nn=ytop0-py*CS
        if near_known(e,nn): continue
        if any((e-ke)**2+(nn-kn)**2<60*60 for ke,kn in kept): continue
        h=hs(z,CS*f);lo,hi=np.percentile(h,2),np.percentile(h,98)
        if hi-lo<1e-6: continue
        st=np.asarray(Image.fromarray(np.clip((h-lo)/(hi-lo)*255,0,255).astype('uint8')).resize((128,128)),np.uint8)
        fn=f"dataset_neg_ditch/ditch_{PREFIX}_{n:05d}.png";Image.fromarray(st).save(f"{H}/{fn}")
        mw.writerow([fn,f"{e:.1f}",f"{nn:.1f}",f"{coh:.3f}",f"{lr:.1f}"]);kept.append((e,nn));n+=1
mf.close()
print(f"scanat {scanned} celule -> {n} negative-șanț -> dataset_neg_ditch/ (prefix {PREFIX})",flush=True)
