#!/usr/bin/env python3
# sweep_eval.py MODEL.pt CLON CLAT KM [TAG] — baleiaza o zona cu un model dat, raporteaza densitatea
# de aprinderi (%>0.5/0.6/0.8) si salveaza un overlay cu celulele scor-mare. Pt A/B intre modele pe
# aceeasi zona ANTROPICA proaspata (test de generalizare: scade FP pe sate/canale neantrenate?).
import os,sys,math,subprocess,zipfile
import numpy as np
from PIL import Image
import torch,torch.nn as nn
H=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); dev=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
MODEL=sys.argv[1]; CLON=float(sys.argv[2]); CLAT=float(sys.argv[3]); KM=float(sys.argv[4]) if len(sys.argv)>4 else 4.0
TAG=sys.argv[5] if len(sys.argv)>5 else os.path.splitext(os.path.basename(MODEL))[0]
CACHE="/tmp/laki3"; CS=0.5; TPX=2000; os.makedirs(CACHE,exist_ok=True)
APP=os.environ.get("QGIS_APP","/Applications/QGIS-final-4_0_3.app/Contents")
ENV=dict(os.environ,DYLD_FRAMEWORK_PATH=f"{APP}/Frameworks",PROJ_DATA=f"{APP}/Resources/qgis/proj",PROJ_LIB=f"{APP}/Resources/qgis/proj",GDAL_DATA=f"{APP}/Resources/qgis/gdal")
GT=f"{APP}/MacOS/gdaltransform"
def to_st(lo,la):
    r=subprocess.run([GT,"-s_srs","EPSG:4326","-t_srs","EPSG:3844"],input=f"{lo} {la}\n",capture_output=True,text=True,env=ENV);e,n,_=r.stdout.split();return float(e),float(n)
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
net=Net().to(dev);net.load_state_dict(torch.load(MODEL,weights_only=True));net.eval()
est,nord=to_st(CLON,CLAT);half=KM*1000/2
e0=int((est-half)//1000);e1=int((est+half)//1000);n0=int((nord-half)//1000);n1=int((nord+half)//1000)
xll0=e0*1000;ytop0=(n1+1)*1000;W=(e1-e0+1)*TPX;Hh=(n1-n0+1)*TPX
mos=np.full((Hh,W),np.nan,np.float32);got=0
for nk in range(n0,n1+1):
    for ek in range(e0,e1+1):
        d=dl(nk,ek)
        if d is None: continue
        got+=1;ox=int((ek*1000-xll0)/CS);oy=int((ytop0-(nk+1)*1000)/CS);mos[oy:oy+TPX,ox:ox+TPX]=d[:TPX,:TPX]
wpx=int(80/CS);stride=int(40/CS);f=int(round(2.0/CS))
ys=list(range(0,Hh-wpx,stride));xs=list(range(0,W-wpx,stride))
heat=np.full((len(ys),len(xs)),-1.,np.float32);batch=[];idx=[]
def flush():
    global batch,idx
    if not batch: return
    xb=torch.tensor(np.array(batch)).unsqueeze(1).float().to(dev)
    with torch.no_grad(): sc=torch.sigmoid(net(xb)).cpu().numpy()
    for (iy,ix),s in zip(idx,sc): heat[iy,ix]=s
    batch=[];idx=[]
for iy,yy in enumerate(ys):
    for ix,xx in enumerate(xs):
        w=mos[yy:yy+wpx,xx:xx+wpx]
        if np.isnan(w).mean()>0.05: continue
        d2=downs(w,f);h=hs(d2,CS*f);lo,hi=np.percentile(h,2),np.percentile(h,98)
        if hi-lo<1e-6: continue
        batch.append(np.asarray(Image.fromarray(np.clip((h-lo)/(hi-lo)*255,0,255).astype('uint8')).resize((128,128)),np.float32)/255.);idx.append((iy,ix))
        if len(batch)>=512: flush()
flush()
valid=heat[heat>=0];tot=valid.size
p5=int((valid>0.5).sum());p6=int((valid>0.6).sum());p8=int((valid>0.8).sum())
print(f"[{TAG}] zona {CLON},{CLAT} {KM}km, {got} dale, {tot} celule valide")
print(f"[{TAG}] >0.5: {p5} ({100*p5/tot:.2f}%) | >0.6: {p6} ({100*p6/tot:.2f}%) | >0.8: {p8} ({100*p8/tot:.2f}%)")
# overlay: hillshade zona (downsampled) + celule rosii unde scor>0.6
base=downs(np.nan_to_num(mos,nan=np.nanmedian(mos)),20);bh=hs(base,CS*20);lo,hi=np.percentile(bh,2),np.percentile(bh,98)
img=np.clip((bh-lo)/(hi-lo)*255,0,255).astype('uint8');rgb=np.stack([img]*3,-1)
sc=W/img.shape[1]
for iy,yy in enumerate(ys):
    for ix,xx in enumerate(xs):
        if heat[iy,ix]>0.6:
            cy=int((yy+wpx//2)/sc);cx=int((xx+wpx//2)/sc)
            rgb[max(0,cy-1):cy+2,max(0,cx-1):cx+2]=[255,40,40]
Image.fromarray(rgb).save(f"{H}/review/sweepeval_{TAG}.png")
print(f"[{TAG}] overlay -> review/sweepeval_{TAG}.png")
