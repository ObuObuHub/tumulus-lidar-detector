#!/usr/bin/env python3
# verify_oltenia.py — VERIFICARE riguroasă pe 0.5m (Oltenia/Mehedinți), ACELAȘI protocol ca verify_moldova:
# peak-search ±60m + controale HARD peak-searched. Set A = tumuli RAN PROASPEȚI (oltenia_tumuli.json, doar cei
# cu LAKI3); Set B = Catane 24-GT held-out. Rețeta PRODUCȚIE (LAKI3 0.5m -> downs 2m -> hillshade 6-dir -> homog).
import os,sys,math,subprocess,csv,json,random,zipfile
import numpy as np
from PIL import Image,ImageFilter,ImageDraw,ImageFont
import torch,torch.nn as nn
random.seed(13)
H=os.path.dirname(os.path.dirname(os.path.abspath(__file__)));dev=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
CACHE="/tmp/laki3";CS=0.5;TPX=2000;os.makedirs(CACHE,exist_ok=True)
APP=os.environ.get("QGIS_APP","/Applications/QGIS-final-4_0_3.app/Contents")
ENV=dict(os.environ,DYLD_FRAMEWORK_PATH=f"{APP}/Frameworks",PROJ_DATA=f"{APP}/Resources/qgis/proj",PROJ_LIB=f"{APP}/Resources/qgis/proj",GDAL_DATA=f"{APP}/Resources/qgis/gdal")
GTb=f"{APP}/MacOS/gdaltransform"
def trans(pts,s,t):
    inp="\n".join(f"{a} {b}" for a,b in pts)+"\n";r=subprocess.run([GTb,"-s_srs",s,"-t_srs",t],input=inp,capture_output=True,text=True,env=ENV)
    return [tuple(map(float,l.split()[:2])) for l in r.stdout.strip().split("\n") if l.split()]
def load_one(nk,ek):
    p=f"{CACHE}/{nk}_{ek}.npy"
    if os.path.exists(p):
        try:return np.load(p)
        except:pass
    z=f"{CACHE}/{nk}_{ek}.zip"
    if not os.path.exists(z):subprocess.run(["curl","-s","--max-time","120","-o",z,f"https://geoportal.ancpi.ro/laki3_mnt/zip/{nk}_{ek}.zip"],check=False)
    try:zf=zipfile.ZipFile(z)
    except:
        if os.path.exists(z):os.remove(z)
        return None
    asc=[n for n in zf.namelist() if n.lower().endswith('.asc')]
    if not asc:return None
    raw=zf.read(asc[0]).decode('latin-1').replace(',','.');lines=raw.split('\n');hdr={};i=0
    while i<len(lines):
        pp=lines[i].split()
        if len(pp)>=2 and pp[0].lower() in ('ncols','nrows','xllcorner','yllcorner','cellsize','nodata_value'):hdr[pp[0].lower()]=float(pp[1]);i+=1
        else:break
    nc=int(hdr['ncols']);nr=int(hdr['nrows']);nd=hdr.get('nodata_value',-9999)
    d=np.fromstring(' '.join(lines[i:]),sep=' ',dtype=np.float32)[:nc*nr].reshape(nr,nc);d[d==nd]=np.nan;np.save(p,d);return d
def hs(dem,cs,azs=(315,45,135,225,270,0),alt=35):
    gy,gx=np.gradient(dem,cs);sl=np.arctan(np.hypot(gx,gy));asp=np.arctan2(-gy,gx);o=np.zeros_like(dem);ar=math.radians(alt)
    for az in azs: azr=math.radians(360-az+90);o+=np.clip(np.sin(ar)*np.cos(sl)+np.cos(ar)*np.sin(sl)*np.cos(azr-asp),0,1)
    return o/len(azs)
def downs(a,fc):
    Hh,Ww=a.shape;return a[:Hh//fc*fc,:Ww//fc*fc].reshape(Hh//fc,fc,Ww//fc,fc).mean((1,3))
def homog(a):
    a2=np.asarray(Image.fromarray(a).filter(ImageFilter.GaussianBlur(0.8)),np.uint8);cdf=np.bincount(a2.ravel(),minlength=256).astype(np.float64).cumsum()
    return (cdf[a2]/cdf[-1]*255).astype(np.uint8) if cdf[-1]>0 else a2
class Net(nn.Module):
    def __init__(s):
        super().__init__();s.c=nn.Sequential(nn.Conv2d(1,16,3,2,1),nn.ReLU(),nn.Conv2d(16,32,3,2,1),nn.ReLU(),nn.Conv2d(32,64,3,2,1),nn.ReLU(),nn.AdaptiveAvgPool2d(1));s.f=nn.Linear(64,1)
    def forward(s,x): return s.f(s.c(x).flatten(1)).squeeze(1)
net=Net().to(dev);net.load_state_dict(torch.load(f'{H}/combined_cnn.pt',map_location=dev,weights_only=True));net.eval()
f=int(round(2.0/CS))
def mosaic(est,nord,half):
    e0=int((est-half)//1000);e1=int((est+half)//1000);n0=int((nord-half)//1000);n1=int((nord+half)//1000)
    xll0=e0*1000;ytop0=(n1+1)*1000;W=(e1-e0+1)*TPX;Hh=(n1-n0+1)*TPX;mos=np.full((Hh,W),np.nan,np.float32);nt=0
    for nk in range(n0,n1+1):
        for ek in range(e0,e1+1):
            d=load_one(nk,ek)
            if d is None:continue
            nt+=1;ox=int((ek*1000-xll0)/CS);oy=int((ytop0-(nk+1)*1000)/CS);mos[oy:oy+TPX,ox:ox+TPX]=d[:TPX,:TPX]
    return mos,xll0,ytop0,nt
def stampbatch(mos,px,py):
    out=[]
    for dpx in range(-120,121,40):
        for dpy in range(-120,121,40):
            for wm in (64,80):
                hw=int(wm/2/CS);w=mos[py+dpy-hw:py+dpy+hw,px+dpx-hw:px+dpx+hw]
                if w.shape!=(2*hw,2*hw) or np.isnan(w).mean()>0.05: continue
                d2=downs(np.nan_to_num(w,nan=np.nanmedian(w)),f);h=hs(d2,CS*f)
                lo,hi=np.percentile(h,2),np.percentile(h,98)
                if hi-lo<1e-6: continue
                out.append(homog(np.asarray(Image.fromarray(np.clip((h-lo)/(hi-lo)*255,0,255).astype('uint8')).resize((128,128)),np.uint8)))
    return out
def peak(mos,px,py):
    S=stampbatch(mos,px,py)
    if not S: return None
    X=torch.tensor(np.array(S,dtype=np.uint8))
    with torch.no_grad(): v=torch.sigmoid(net(X.unsqueeze(1).float().to(dev)/255.)).cpu().numpy()
    return float(v.max())
def auroc(p,n):
    if not p or not n: return float('nan')
    return sum((a>b)+0.5*(a==b) for a in p for b in n)/(len(p)*len(n))
def run(name,coords):
    pos=[];neg=[]
    for lo,la in coords:
        est,nord=trans([(lo,la)],"EPSG:4326","EPSG:3844")[0]
        mos,xll0,ytop0,nt=mosaic(est,nord,820)
        if nt==0: continue
        px=int((est-xll0)/CS);py=int((ytop0-nord)/CS);ps=peak(mos,px,py)
        if ps is None: continue
        pos.append(ps)
        W=mos.shape[1];Hh=mos.shape[0];cnt=0;tr=0
        while cnt<3 and tr<60:
            tr+=1;qx=random.randint(170,W-170);qy=random.randint(170,Hh-170)
            if (qx-px)**2+(qy-py)**2<(250/CS)**2: continue
            ns=peak(mos,qx,qy)
            if ns is not None: neg.append(ns);cnt+=1
    pos=np.array(pos);neg=np.array(neg)
    print(f"\n=== {name}: {len(pos)} tumuli, {len(neg)} controale hard ===")
    if len(pos):
        print(f"  TUMULI : median {np.median(pos):.3f} | recall@0.5 {100*np.mean(pos>=0.5):.0f}% @0.7 {100*np.mean(pos>=0.7):.0f}% @0.9 {100*np.mean(pos>=0.9):.0f}%")
        print(f"  CONTROL: median {np.median(neg):.3f} | %@0.5 {100*np.mean(neg>=0.5):.0f}% @0.7 {100*np.mean(neg>=0.7):.0f}% @0.9 {100*np.mean(neg>=0.9):.0f}%")
        print(f"  AUROC = {auroc(list(pos),list(neg)):.3f}")
    return pos,neg
# Set A: fresh RAN Oltenia (doar cei cu 0.5m)
T=json.load(open('/tmp/oltenia_tumuli.json'))
fresh=[(t['lon'],t['lat']) for t in T if t['loc']!='Costești']  # Costesti=404
posA,_=run("FRESH RAN Oltenia/Mehedinti (0.5m)",fresh)
# Set B: Catane 24-GT held-out
cat=[(float(r['lon']),float(r['lat'])) for r in csv.DictReader(open('/tmp/catane_gt_full.csv'))]
posB,_=run("Catane 24-GT held-out (0.5m)",cat)
# salveaza scoruri fresh pt board
json.dump({'fresh':[ {'lon':lo,'lat':la} for lo,la in fresh]}, open('/tmp/oltenia_fresh.json','w'))
