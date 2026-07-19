#!/usr/bin/env python3
# scan_zone_v4.py — scanare interactiva de zona cu SCANERUL INTEGRAT v4 (tumul_scan: filtru potrivit cu
# amprenta -> formula noisy-OR "ochi SAU forma" + garda de cusaturi NoData). Inlocuieste scan_zone.py
# in demo-ul Colab (10.07.2026, decizia din README). Acelasi contract de iesiri ca scan_zone.py:
#   /tmp/zone_dets.csv        (lon,lat,score,cnn,mahal,keep; score = scorul fuzionat v4)
#   review/zone_view.jpg      (hillshade + marcaje: VERDE = candidat pastrat, portocaliu = de verificat
#                              — scor >=0.60 dar sub prag sau taiat de filtre; pragurile vin din tumul_scan)
#   review/zone_board.jpg     (crop-uri hillshade+SLRM ale candidatilor pastrati, numerotate)
#   detected_mounds.csv       (lista pastratilor, pt. descarcare)
# Usage: scan_zone_v4.py LON LAT [KM]
import os,sys,csv,math,subprocess,importlib.util
import numpy as np
from PIL import Image,ImageFilter,ImageDraw,ImageFont
H=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLON=float(sys.argv[1]);CLAT=float(sys.argv[2]);KM=float(sys.argv[3]) if len(sys.argv)>3 else 1.0
CACHE=os.environ.get("LAKI3_CACHE","/tmp/laki3");CS=0.5;TPX=2000
os.makedirs(CACHE,exist_ok=True);os.makedirs(f'{H}/review',exist_ok=True)
os.environ["LAKI3_CACHE"]=CACHE
[os.remove(_p) for _p in ('/tmp/zone_dets.csv',f'{H}/review/zone_view.jpg',f'{H}/review/zone_board.jpg',f'{H}/detected_mounds.csv') if os.path.exists(_p)]
import pyproj
_t4326=pyproj.Transformer.from_crs("EPSG:4326","EPSG:3844",always_xy=True)
_t3844=pyproj.Transformer.from_crs("EPSG:3844","EPSG:4326",always_xy=True)
# dale: ANCPI intai (acoperire completa); daca pica (geoportal offline dupa atacul din iulie 2026),
# mirror-ul GitHub Releases cu dalele zonei demo (8x8 km, date (c) ANCPI, redistribuite nemodificat).
MIRROR=os.environ.get("TILE_MIRROR","https://github.com/ObuObuHub/tumulus-lidar-detector/releases/download/demo-tiles")
def load_one(nk,ek):
    p=f"{CACHE}/{nk}_{ek}.npy"
    if os.path.exists(p):
        try:return np.load(p)
        except:pass
    z=f"{CACHE}/{nk}_{ek}.zip";zf=None;import zipfile
    for base in ("https://geoportal.ancpi.ro/laki3_mnt/zip",MIRROR):
        if not os.path.exists(z):subprocess.run(["curl","-sL","--connect-timeout","8","--max-time","120","-o",z,f"{base}/{nk}_{ek}.zip"],check=False)
        try:zf=zipfile.ZipFile(z);break
        except:
            if os.path.exists(z):os.remove(z)
    if zf is None:return None
    asc=[n for n in zf.namelist() if n.lower().endswith('.asc')]
    if not asc:return None
    raw=zf.read(asc[0]).decode('latin-1').replace(',','.');lines=raw.split('\n');hdr={};i=0
    while i<len(lines):
        pp=lines[i].split()
        if len(pp)>=2 and pp[0].lower() in ('ncols','nrows','xllcorner','yllcorner','cellsize','nodata_value'):hdr[pp[0].lower()]=float(pp[1]);i+=1
        else:break
    nc=int(hdr['ncols']);nr=int(hdr['nrows']);nd=hdr.get('nodata_value',-9999)
    d=np.fromstring(' '.join(lines[i:]),sep=' ',dtype=np.float32)[:nc*nr].reshape(nr,nc);d[d==nd]=np.nan;np.save(p,d);return d
# dalele necesare (centru +/- KM/2, +1 km margine pt contextul scanerului)
est,nord=_t4326.transform(CLON,CLAT);half=KM*1000/2
e0=int((est-half)//1000);e1=int((est+half)//1000);n0=int((nord-half)//1000);n1=int((nord+half)//1000)
nt=0
for nk in range(n0-1,n1+2):
    for ek in range(e0-1,e1+2):
        if load_one(nk,ek) is not None and n0<=nk<=n1 and e0<=ek<=e1:nt+=1
if nt==0:print("ERROR: no LiDAR tiles here - outside the 0.5 m coverage, or ANCPI is offline and this area is not in the demo mirror. The green demo area on the map always works.");sys.exit(2)
print(f"{nt} tiles in zone ({KM}km); running v4 scanner (fingerprint detect -> fused decision)...",flush=True)
# scanerul integrat
ts=importlib.util.spec_from_file_location("ts",f"{H}/tools/tumul_scan.py");TS=importlib.util.module_from_spec(ts);ts.loader.exec_module(TS)
c1=_t3844.transform(est-half,nord-half);c2=_t3844.transform(est+half,nord+half)
mos,xll,ytop,area=TS.scan_laki3(min(c1[0],c2[0]),max(c1[0],c2[0]),min(c1[1],c2[1]),max(c1[1],c2[1]))
cands,S=TS.scan(mos,0.5)
rows=[]
for c in cands:
    if c['fuse']<0.60 and not c['keep']:continue
    E=xll+c['x2']*2.0;N=ytop-c['y2']*2.0
    if not (est-half<=E<=est+half and nord-half<=N<=nord+half):continue
    lo,la=_t3844.transform(E,N)
    rows.append(dict(lon=lo,lat=la,E=E,N=N,fuse=c['fuse'],cnn=c['cnn'],mahal=c['mahal'],keep=int(c['keep'])))
rows.sort(key=lambda r:-r['fuse'])
kept=[r for r in rows if r['keep']]
print(f"proposals scored; kept (fused >={TS.FUSE_THR} + FP filter + guards): {len(kept)} | to review (>=0.60, below threshold or filtered): {len(rows)-len(kept)}",flush=True)
with open('/tmp/zone_dets.csv','w',newline='') as fo:
    w=csv.writer(fo);w.writerow(['lon','lat','score','cnn','mahal','keep'])
    for r in rows:
        w.writerow([f"{r['lon']:.6f}",f"{r['lat']:.6f}",f"{r['fuse']:.3f}",
                    f"{r['cnn']:.3f}" if not math.isnan(r['cnn']) else '',f"{r['mahal']:.2f}",r['keep']])
with open(f'{H}/detected_mounds.csv','w',newline='') as fo:
    w=csv.writer(fo);w.writerow(['lon','lat','score','google_maps'])
    for r in kept:w.writerow([f"{r['lon']:.6f}",f"{r['lat']:.6f}",f"{r['fuse']:.3f}",f"https://maps.google.com/?q={r['lat']:.6f},{r['lon']:.6f}"])
# ── zone_view.jpg: hillshade-ul zonei + marcaje
def hshade(dem,cs=0.5):
    gy,gx=np.gradient(dem.astype(np.float64),cs)
    slope=np.arctan(np.hypot(gx,gy));aspect=np.arctan2(-gx,gy);alt=math.radians(45)
    o=np.zeros(dem.shape)
    for az in range(0,360,60):
        azr=math.radians(az)
        o+=np.clip(math.sin(alt)*np.cos(slope)+math.cos(alt)*np.sin(slope)*np.cos(azr-aspect),0,1)
    v=o.flatten();idx=np.argsort(v);r=np.empty(len(v));r[idx]=np.arange(len(v))/(len(v)-1)
    return (r.reshape(o.shape)*255).astype(np.uint8)
med=float(np.nanmedian(mos));fill=np.where(np.isfinite(mos),mos,med)
# vederea = DOAR zona ceruta (mozaicul are margine de context in plus), la 2 m/px
zy0=max(0,int((ytop-(nord+half))/CS));zy1=int((ytop-(nord-half))/CS)
zx0=max(0,int((est-half-xll)/CS));zx1=int((est+half-xll)/CS)
zview=fill[zy0:zy1,zx0:zx1]
xllv=xll+zx0*CS;ytopv=ytop-zy0*CS
fc=2
sub=zview[:zview.shape[0]//fc*fc,:zview.shape[1]//fc*fc].reshape(zview.shape[0]//fc,fc,zview.shape[1]//fc,fc).mean((1,3))
hv=hshade(sub,CS*fc)
img=Image.fromarray(hv).convert('RGB');d=ImageDraw.Draw(img)
try:FT=ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc",16)
except:FT=ImageFont.load_default()
def px(E,N):return (E-xllv)/(CS*fc),(ytopv-N)/(CS*fc)
for r in rows:
    x,y=px(r['E'],r['N'])
    col=(40,220,60) if r['keep'] else (255,150,30)
    rr=14 if r['keep'] else 9
    d.ellipse([x-rr,y-rr,x+rr,y+rr],outline=col,width=4 if r['keep'] else 3)
for i,r in enumerate(kept,1):
    x,y=px(r['E'],r['N']);d.text((x+16,y-10),str(i),fill=(40,220,60),font=FT)
img.save(f'{H}/review/zone_view.jpg',quality=85)
print(f"-> review/zone_view.jpg {img.size}",flush=True)
# ── zone_board.jpg: perechi hillshade+SLRM pt pastrati
if kept:
    P=302;cols=3;n=min(len(kept),12)
    rowsn=(n+cols-1)//cols
    board=Image.new('RGB',(cols*2*P,rowsn*(P+40)),(12,12,12));db=ImageDraw.Draw(board)
    for i,r in enumerate(kept[:n]):
        py=int((ytop-r['N'])/CS);pxx=int((r['E']-xll)/CS);HW=300
        w=fill[max(0,py-HW):py+HW,max(0,pxx-HW):pxx+HW]
        cx=(i%cols)*2*P;cy=(i//cols)*(P+40)
        db.text((cx+6,cy+4),f"#{i+1} scor {r['fuse']:.2f} · {r['lat']:.5f}, {r['lon']:.5f}",fill=(255,200,80),font=FT)
        if w.shape==(2*HW,2*HW):
            hh=Image.fromarray(hshade(w)).resize((P-2,P-2),Image.LANCZOS).convert('RGB')
            c0y=int(r['N']-ytop)  # SLRM pe grila 2m a scanerului
            sy=int((ytop-r['N'])/2.0);sx=int((r['E']-xll)/2.0)
            C=S[max(0,sy-75):sy+76,max(0,sx-75):sx+76]
            lo_,hi_=np.percentile(C,2),np.percentile(C,99)
            g=np.clip((C-lo_)/max(hi_-lo_,1e-6),0,1)
            ss=Image.fromarray((g*255).astype(np.uint8)).resize((P-2,P-2),Image.LANCZOS).convert('RGB')
            for j,pim in enumerate((hh,ss)):
                dd=ImageDraw.Draw(pim);c=pim.size[0]//2
                dd.ellipse([c-12,c-12,c+12,c+12],outline=(80,200,255),width=3)
                board.paste(pim,(cx+j*P,cy+28))
    board.save(f'{H}/review/zone_board.jpg',quality=85)
    print(f"-> review/zone_board.jpg ({n} candidates)",flush=True)
else:
    print("(0 candidates kept - clean area / no obvious mounds)",flush=True)
