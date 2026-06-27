#!/usr/bin/env python3
# anthro_negatives.py [N=1200] [THRESH=0.40]
# HARD-NEGATIVE MINING pe TEREN REAL: re-baleiaza zone ANTROPICE din Dolj (sate, canale, limite de
# parcela) cu modelul curent (combined_cnn.pt) si culege celulele scorate MARE = fals-pozitivele
# reale ale modelului pe structuri antropice (cauza precarei precizii la scanare reala).
# Reteta IDENTICA cu dataset_neg (LAKI III DTM 0.5m -> 2m efectiv -> hillshade multidir -> 128px,
# polaritate LUMINOASA). Exclude <120 m de movile cunoscute. Dedup spatial. -> dataset_neg_anthro/.
import os,sys,math,subprocess,zipfile,csv,glob
import numpy as np
from PIL import Image
import torch,torch.nn as nn
H=os.path.expanduser('~/lidar-match'); dev=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
N=int(sys.argv[1]) if len(sys.argv)>1 else 1200
THRESH=float(sys.argv[2]) if len(sys.argv)>2 else 0.40
CACHE="/tmp/laki3"; CS=0.5; TPX=2000; os.makedirs(CACHE,exist_ok=True)
OUT=f"{H}/dataset_neg_anthro"; os.makedirs(OUT,exist_ok=True)
APP="/Applications/QGIS-final-4_0_3.app/Contents"
ENV=dict(os.environ,DYLD_FRAMEWORK_PATH=f"{APP}/Frameworks",PROJ_DATA=f"{APP}/Resources/qgis/proj",PROJ_LIB=f"{APP}/Resources/qgis/proj",GDAL_DATA=f"{APP}/Resources/qgis/gdal")
GT=f"{APP}/MacOS/gdaltransform"
# Zone ANTROPICE Dolj (centru lon/lat, latura km): sate/canale/parcele unde detectorul a dat FP.
ZONES=[(23.078,44.085,3.5),  # Moreni (FP documentat in ro_hotspots.json) + sate vecine
       (23.420,43.916,3.0),  # Catane/Negoi - sate dense + canale langa Dunare
       (23.560,43.902,3.0),  # Goicea/Bistret - sate + canale
       (23.360,44.115,3.0)]  # Galicea Mare/Giubega - sate + parcele
def trans(pts,s,t):
    if not pts: return []
    inp="\n".join(f"{a} {b}" for a,b in pts)+"\n"
    r=subprocess.run([GT,"-s_srs",s,"-t_srs",t],input=inp,capture_output=True,text=True,env=ENV)
    return [(float(l.split()[0]),float(l.split()[1])) for l in r.stdout.strip().split("\n") if l.split()]
def st2ll(e,n):
    a,b=trans([(e,n)],"EPSG:3844","EPSG:4326")[0]; return a,b
# movile cunoscute -> Stereo70 (exclude 120m)
mounds=[(float(r['lon']),float(r['lat'])) for r in csv.DictReader(open(f'{H}/labeled/labels.csv')) if r['verdict']=='mound']
ran=[]
for r in list(csv.reader(open(f'{H}/vest_tumuli.csv')))[1:]:
    try: ran.append((float(r[4]),float(r[5])))
    except: pass
known=np.array(trans(mounds+ran,"EPSG:4326","EPSG:3844")) if (mounds+ran) else np.empty((0,2))
def near_mound(e,n,d=120):
    if known.shape[0]==0: return False
    return bool(np.any((known[:,0]-e)**2+(known[:,1]-n)**2<d*d))
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
    for az in azs:
        azr=math.radians(360-az+90);o+=np.clip(np.sin(ar)*np.cos(sl)+np.cos(ar)*np.sin(sl)*np.cos(azr-asp),0,1)
    return o/len(azs)
def downs(a,f):
    Hh,Ww=a.shape;return a[:Hh//f*f,:Ww//f*f].reshape(Hh//f,f,Ww//f,f).mean((1,3))
class Net(nn.Module):
    def __init__(s):
        super().__init__();s.c=nn.Sequential(nn.Conv2d(1,16,3,2,1),nn.ReLU(),nn.Conv2d(16,32,3,2,1),nn.ReLU(),nn.Conv2d(32,64,3,2,1),nn.ReLU(),nn.AdaptiveAvgPool2d(1));s.f=nn.Linear(64,1)
    def forward(s,x): return s.f(s.c(x).flatten(1)).squeeze(1)
net=Net().to(dev);net.load_state_dict(torch.load(f'{H}/combined_cnn.pt',weights_only=True));net.eval()
wpx=int(80/CS);stride=int(40/CS);f=int(round(2.0/CS))
cands=[]  # (score, est, nord, img128)
for ci,(CLON,CLAT,KM) in enumerate(ZONES):
    e,n=trans([(CLON,CLAT)],"EPSG:4326","EPSG:3844")[0];half=KM*1000/2
    e0=int((e-half)//1000);e1=int((e+half)//1000);n0=int((n-half)//1000);n1=int((n+half)//1000)
    xll0=e0*1000;ytop0=(n1+1)*1000;W=(e1-e0+1)*TPX;Hh=(n1-n0+1)*TPX
    mos=np.full((Hh,W),np.nan,np.float32);got=0
    for nk in range(n0,n1+1):
        for ek in range(e0,e1+1):
            d=dl(nk,ek)
            if d is None: continue
            got+=1;ox=int((ek*1000-xll0)/CS);oy=int((ytop0-(nk+1)*1000)/CS);mos[oy:oy+TPX,ox:ox+TPX]=d[:TPX,:TPX]
    ys=list(range(0,Hh-wpx,stride));xs=list(range(0,W-wpx,stride))
    print(f"zona {ci+1} {CLON},{CLAT} {KM}km: mozaic {W}x{Hh} {got} dale goluri {np.isnan(mos).mean()*100:.0f}% | {len(ys)}x{len(xs)} poz",flush=True)
    batch=[];meta=[]
    def flush(cands=cands):
        global batch,meta
        if not batch: return
        xb=torch.tensor(np.array(batch)).unsqueeze(1).float().to(dev)
        with torch.no_grad(): sc=torch.sigmoid(net(xb)).cpu().numpy()
        for (img,e2,n2),s in zip(meta,sc):
            if s>=THRESH and not near_mound(e2,n2): cands.append((float(s),e2,n2,img))
        batch=[];meta=[]
    for yy in ys:
        for xx in xs:
            w=mos[yy:yy+wpx,xx:xx+wpx]
            if np.isnan(w).mean()>0.05: continue
            d2=downs(w,f);h=hs(d2,CS*f);lo,hi=np.percentile(h,2),np.percentile(h,98)
            if hi-lo<1e-6: continue
            img=np.asarray(Image.fromarray(np.clip((h-lo)/(hi-lo)*255,0,255).astype('uint8')).resize((128,128)),np.float32)/255.
            ec=xll0+(xx+wpx//2)*CS;nc=ytop0-(yy+wpx//2)*CS
            batch.append(img);meta.append((img,ec,nc))
            if len(batch)>=512: flush()
    flush()
print(f"candidati >= {THRESH}: {len(cands)}",flush=True)
# dedup spatial >=70m, prioritar dupa scor (cei mai inselatori intai)
cands.sort(reverse=True,key=lambda c:c[0])
kept=[]
for s,e,n,img in cands:
    if any((e-k[1])**2+(n-k[2])**2<70*70 for k in kept): continue
    kept.append((s,e,n,img))
    if len(kept)>=N: break
print(f"pastrati dupa dedup 70m, cap {N}: {len(kept)}  (scor {kept[-1][0]:.2f}..{kept[0][0]:.2f})",flush=True)
mani=open(f"{OUT}/manifest.csv","w");mw=csv.writer(mani);mw.writerow(["file","est","nord","score"])
for i,(s,e,n,img) in enumerate(kept):
    fn=f"{OUT}/anthro_{i:04d}.png";Image.fromarray((img*255).astype('uint8')).save(fn)
    mw.writerow([os.path.basename(fn),f"{e:.1f}",f"{n:.1f}",f"{s:.3f}"])
mani.close()
# montaj de control (primii 48)
cols=8;rows=min(6,(len(kept)+cols-1)//cols);M=Image.new('L',(cols*132,rows*132),40)
for i,(s,e,n,img) in enumerate(kept[:cols*rows]):
    M.paste(Image.fromarray((img*255).astype('uint8')),((i%cols)*132+2,(i//cols)*132+2))
M.save(f"{H}/review/anthro_neg_sample.png")
print(f"-> {OUT}/ ({len(kept)} png) + manifest.csv + review/anthro_neg_sample.png")
