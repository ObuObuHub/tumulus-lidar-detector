#!/usr/bin/env python3
# add_positives.py DET_CSV IDX1,IDX2,... — adaugă detecții confirmate de Andrei ca pozitivi RO.
# Taie stampa 80m din DTM LAKI III pe MOZAIC (gestionează marginile de dală — fix la cut_snap single-tile
# care pica la edge și pierdea movile confirmate). Snap pe vârful domului (elev local max <=30m).
# Rețetă IDENTICĂ dataset_pos (80m, downsample 2m, hillshade multidir, 128px). Dedup vs labels + între ele.
# Aliniat dataset_pos (pos_NN.png sortat) <-> randuri LAKI3_DTM din labels.csv. Backup labels înainte.
import os,sys,math,subprocess,zipfile,csv,glob,shutil,time
import numpy as np
from PIL import Image
H=os.path.dirname(os.path.dirname(os.path.abspath(__file__)));CACHE="/tmp/laki3";CS=0.5;TPX=2000
DET=sys.argv[1]; IDX=[int(x) for x in sys.argv[2].split(',') if x.strip()]
APP=os.environ.get("QGIS_APP","/Applications/QGIS-final-4_0_3.app/Contents")
ENV=dict(os.environ,DYLD_FRAMEWORK_PATH=f"{APP}/Frameworks",PROJ_DATA=f"{APP}/Resources/qgis/proj",PROJ_LIB=f"{APP}/Resources/qgis/proj",GDAL_DATA=f"{APP}/Resources/qgis/gdal")
GT=f"{APP}/MacOS/gdaltransform"
def trans(pts,s,t):
    if not pts: return []
    inp="\n".join(f"{a} {b}" for a,b in pts)+"\n"
    r=subprocess.run([GT,"-s_srs",s,"-t_srs",t],input=inp,capture_output=True,text=True,env=ENV)
    return [(float(l.split()[0]),float(l.split()[1])) for l in r.stdout.strip().split("\n") if l.split()]
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
rows={int(r['idx']):r for r in csv.DictReader(open(DET))}
want=[(i,float(rows[i]['lon']),float(rows[i]['lat']),float(rows[i].get('score',0))) for i in IDX if i in rows]
sts=trans([(lo,la) for i,lo,la,sc in want],"EPSG:4326","EPSG:3844")
ens=[(e,n) for e,n in sts]
# MOZAIC ce acoperă toate punctele + margine
e0=int(min(e for e,n in ens)//1000)-1;e1=int(max(e for e,n in ens)//1000)+1
n0=int(min(n for e,n in ens)//1000)-1;n1=int(max(n for e,n in ens)//1000)+1
xll0=e0*1000;ytop0=(n1+1)*1000;W=(e1-e0+1)*TPX;Hh=(n1-n0+1)*TPX
mos=np.full((Hh,W),np.nan,np.float32)
for nk in range(n0,n1+1):
    for ek in range(e0,e1+1):
        d=dl(nk,ek)
        if d is None: continue
        ox=int((ek*1000-xll0)/CS);oy=int((ytop0-(nk+1)*1000)/CS);mos[oy:oy+TPX,ox:ox+TPX]=d[:TPX,:TPX]
print(f"mozaic {W}x{Hh} ({e0}-{e1} x {n0}-{n1}), goluri {np.isnan(mos).mean()*100:.0f}%")
f=int(round(2.0/CS))
def cut(e,n):
    px=int((e-xll0)/CS);py=int((ytop0-n)/CS);rr=int(30/CS)
    sub=mos[py-rr:py+rr,px-rr:px+rr]
    if sub.shape==(2*rr,2*rr) and not np.isnan(sub).all():
        sm=boxblur(np.nan_to_num(sub,nan=np.nanmin(sub)),3);off=np.unravel_index(np.nanargmax(sm),sm.shape);py=py-rr+off[0];px=px-rr+off[1]
    e2=xll0+px*CS;n2=ytop0-py*CS;hw=int(40/CS);w=mos[py-hw:py+hw,px-hw:px+hw]
    if w.shape!=(2*hw,2*hw) or np.isnan(w).mean()>0.05: return None,None,None
    d2=downs(w,f);h=hs(d2,CS*f);lo,hi=np.percentile(h,2),np.percentile(h,98)
    if hi-lo<1e-6: return None,None,None
    img=Image.fromarray(np.clip((h-lo)/(hi-lo)*255,0,255).astype('uint8')).resize((128,128))
    lo2,la2=trans([(e2,n2)],"EPSG:3844","EPSG:4326")[0];return img,lo2,la2
train=[(float(r['lon']),float(r['lat'])) for r in csv.DictReader(open(f'{H}/labeled/labels.csv')) if r['verdict']=='mound']
def near_tr(lo,la,d=120): return any(((lo-a)*111000*math.cos(math.radians(la)))**2+((la-b)*111000)**2<d*d for a,b in train)
added=[];newc=[];skk=0;skf=0
for (i,lo,la,sc),(e,n) in zip(want,ens):
    img,lo2,la2=cut(e,n)
    if img is None: skf+=1;print(f"  #{i}: FAIL tăiere (gol date)");continue
    if near_tr(lo2,la2): skk+=1;print(f"  #{i}: deja în training");continue
    if any(((lo2-a)*111000*math.cos(math.radians(la2)))**2+((la2-b)*111000)**2<50*50 for a,b in newc): continue
    newc.append((lo2,la2));added.append((i,lo2,la2,sc,img))
if added:
    shutil.copy(f'{H}/labeled/labels.csv',f"{H}/labeled/labels.csv.bak-add-{int(time.time())}")
    nexist=len(glob.glob(f'{H}/dataset_pos/*.png'))
    for k,(i,lo2,la2,sc,img) in enumerate(added): img.save(f"{H}/dataset_pos/pos_{nexist+k:02d}.png")
    with open(f'{H}/labeled/labels.csv','a',newline='') as fl:
        w=csv.writer(fl)
        for i,lo2,la2,sc,img in added: w.writerow(["LAKI3_DTM","",f"{lo2:.5f}",f"{la2:.5f}","mound","tumul","model_found"])
    with open(f'{H}/labeled/new_tumuli_model.csv','a',newline='') as fd:
        w=csv.writer(fd)
        for i,lo2,la2,sc,img in added: w.writerow([f"{lo2:.5f}",f"{la2:.5f}",f"{sc:.3f}","Dolj",time.strftime("%Y-%m-%d"),f"det#{i} {os.path.basename(DET)}, confirmat Andrei"])
p=len(glob.glob(f'{H}/dataset_pos/*.png'));dd=sum(1 for r in csv.DictReader(open(f'{H}/labeled/labels.csv')) if r['tile']=='LAKI3_DTM')
print(f"\nconfirmate {len(want)} -> ADĂUGATE {len(added)} | deja train {skk} | FAIL {skf}")
print(f"pozitivi RO DTM: {p} | LAKI3_DTM rows {dd} | {'ALIGN OK' if p==dd else 'MISMATCH!'}")
