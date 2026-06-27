#!/usr/bin/env python3
# mine_terrace_neg.py [N=2000] [THRESH=0.7]
# HARD-NEGATIVE MINING ȚINTIT pe TERASE/RÂPE: baleiez zone DELUROASE Gorj fără movile (confirmat Andrei)
# cu modelul curent (homogenizat) -> culeg celulele scor-mare = fix muchiile de terasă/râpă pe care GREȘEȘTE
# (random hill le-a ratat). Salvez stampa RAW (percentil 2-98, ca celelalte negative; omogenizarea se
# aplică la train). Exclud <150m de movile cunoscute. -> dataset_neg_terrace/.
import os,sys,math,subprocess,zipfile,csv,glob
import numpy as np
from PIL import Image,ImageFilter
import torch,torch.nn as nn
H=os.path.expanduser('~/lidar-match');dev=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
N=int(sys.argv[1]) if len(sys.argv)>1 else 2000
THRESH=float(sys.argv[2]) if len(sys.argv)>2 else 0.7
CACHE="/tmp/laki3";CS=0.5;TPX=2000;os.makedirs(CACHE,exist_ok=True)
OUT=f"{H}/dataset_neg_terrace";os.makedirs(OUT,exist_ok=True)
APP="/Applications/QGIS-final-4_0_3.app/Contents"
ENV=dict(os.environ,DYLD_FRAMEWORK_PATH=f"{APP}/Frameworks",PROJ_DATA=f"{APP}/Resources/qgis/proj",PROJ_LIB=f"{APP}/Resources/qgis/proj",GDAL_DATA=f"{APP}/Resources/qgis/gdal")
GT=f"{APP}/MacOS/gdaltransform"
# centre DELUROASE Gorj (mound-free), latura ~5km
ZONES=[(23.50,44.63),(23.00,45.00),(23.30,45.10),(22.90,44.85),(23.60,45.05),(22.75,45.20),(23.20,44.90),(23.70,44.80)]
KM=5
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
def homog(a):
    a2=np.asarray(Image.fromarray(a).filter(ImageFilter.GaussianBlur(0.8)),np.uint8)
    cdf=np.bincount(a2.ravel(),minlength=256).astype(np.float64).cumsum()
    return (cdf[a2]/cdf[-1]*255).astype(np.uint8) if cdf[-1]>0 else a2
mounds=[(float(r['lon']),float(r['lat'])) for r in csv.DictReader(open(f'{H}/labeled/labels.csv')) if r['verdict']=='mound']
known=np.array(trans(mounds,"EPSG:4326","EPSG:3844")) if mounds else np.empty((0,2))
def near(e,n,d=150): return known.shape[0]>0 and bool(np.any((known[:,0]-e)**2+(known[:,1]-n)**2<d*d))
class Net(nn.Module):
    def __init__(s):
        super().__init__();s.c=nn.Sequential(nn.Conv2d(1,16,3,2,1),nn.ReLU(),nn.Conv2d(16,32,3,2,1),nn.ReLU(),nn.Conv2d(32,64,3,2,1),nn.ReLU(),nn.AdaptiveAvgPool2d(1));s.f=nn.Linear(64,1)
    def forward(s,x): return s.f(s.c(x).flatten(1)).squeeze(1)
net=Net().to(dev);net.load_state_dict(torch.load(f'{H}/combined_cnn.pt',weights_only=True));net.eval()
f=int(round(2.0/CS));wpx=int(80/CS);stride=int(40/CS)
kept=[]  # (score, raw_stamp_uint8, e, n)
for zi,(CLON,CLAT) in enumerate(ZONES):
    if len(kept)>=N: break
    est,nord=trans([(CLON,CLAT)],"EPSG:4326","EPSG:3844")[0];half=KM*1000/2
    e0=int((est-half)//1000);e1=int((est+half)//1000);n0=int((nord-half)//1000);n1=int((nord+half)//1000)
    xll0=e0*1000;ytop0=(n1+1)*1000;W=(e1-e0+1)*TPX;Hh=(n1-n0+1)*TPX
    mos=np.full((Hh,W),np.nan,np.float32);got=0
    for nk in range(n0,n1+1):
        for ek in range(e0,e1+1):
            d=dl(nk,ek)
            if d is None: continue
            got+=1;ox=int((ek*1000-xll0)/CS);oy=int((ytop0-(nk+1)*1000)/CS);mos[oy:oy+TPX,ox:ox+TPX]=d[:TPX,:TPX]
    if got==0: print(f"zona {zi+1} {CLON},{CLAT}: 0 dale, skip");continue
    ys=list(range(0,Hh-wpx,stride));xs=list(range(0,W-wpx,stride));batch=[];meta=[]
    def flush():
        global batch,meta
        if not batch: return
        xb=torch.tensor(np.array([homog(b[0]) for b in batch])).unsqueeze(1).float().to(dev)/255.
        with torch.no_grad(): sc=torch.sigmoid(net(xb)).cpu().numpy()
        for (raw,e2,n2),s in zip([(b[0],b[1],b[2]) for b in batch],sc):
            if s>=THRESH and not near(e2,n2): kept.append((float(s),raw,e2,n2))
        batch=[];meta=[]
    for yy in ys:
        for xx in xs:
            if len(kept)>=N*3: break
            w=mos[yy:yy+wpx,xx:xx+wpx]
            if np.isnan(w).mean()>0.05: continue
            d2=downs(w,f);h=hs(d2,CS*f);lo,hi=np.percentile(h,2),np.percentile(h,98)
            if hi-lo<1e-6: continue
            raw=np.asarray(Image.fromarray(np.clip((h-lo)/(hi-lo)*255,0,255).astype('uint8')).resize((128,128)),np.uint8)
            ec=xll0+(xx+wpx//2)*CS;nc=ytop0-(yy+wpx//2)*CS
            batch.append((raw,ec,nc))
            if len(batch)>=512: flush()
    flush()
    print(f"zona {zi+1} {CLON},{CLAT}: {got} dale -> total kept {len(kept)}",flush=True)
# dedup spatial 70m, prioritar scor mare
kept.sort(key=lambda c:-c[0]);ded=[]
for s,raw,e,n in kept:
    if any((e-k[2])**2+(n-k[3])**2<70*70 for k in ded): continue
    ded.append((s,raw,e,n))
    if len(ded)>=N: break
man=open(f"{OUT}/manifest.csv","w");mw=csv.writer(man);mw.writerow(["file","est","nord","score"])
for i,(s,raw,e,n) in enumerate(ded):
    fn=f"{OUT}/terr_{i:04d}.png";Image.fromarray(raw).save(fn);mw.writerow([os.path.basename(fn),f"{e:.1f}",f"{n:.1f}",f"{s:.3f}"])
man.close()
# montaj control
cols=8;rows=min(6,(len(ded)+7)//8);M=Image.new('L',(cols*132,rows*132),40)
for i,(s,raw,e,n) in enumerate(ded[:cols*rows]): M.paste(Image.fromarray(raw),((i%cols)*132+2,(i//cols)*132+2))
M.save(f"{H}/review/terrace_neg_sample.png")
print(f"GATA: {len(ded)} terase hard-neg -> {OUT}/ (scor {ded[-1][0]:.2f}..{ded[0][0]:.2f}) + review/terrace_neg_sample.png")
