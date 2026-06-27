#!/usr/bin/env python3
# sweep_dtm.py CLON CLAT KM — faza DETECTOR pe DTM brut RO (LAKI III, polaritate luminoasă = ca modelul).
# Mozaic DTM al zonei -> fereastra glisanta a modelului (80m@2m hillshade) in memorie -> heatmap -> hot-spots.
import os,sys,math,subprocess,zipfile,json
import numpy as np
from PIL import Image,ImageFilter
import torch,torch.nn as nn
H=os.path.expanduser('~/lidar-match');dev=torch.device('mps')
CLON=float(sys.argv[1]);CLAT=float(sys.argv[2]);KM=float(sys.argv[3]) if len(sys.argv)>3 else 4.0
CACHE="/tmp/laki3";CS=0.5;TPX=2000;os.makedirs(CACHE,exist_ok=True)
APP="/Applications/QGIS-final-4_0_3.app/Contents"
ENV=dict(os.environ,DYLD_FRAMEWORK_PATH=f"{APP}/Frameworks",PROJ_DATA=f"{APP}/Resources/qgis/proj",PROJ_LIB=f"{APP}/Resources/qgis/proj",GDAL_DATA=f"{APP}/Resources/qgis/gdal")
GT=f"{APP}/MacOS/gdaltransform"
def to_st(lo,la):
    r=subprocess.run([GT,"-s_srs","EPSG:4326","-t_srs","EPSG:3844"],input=f"{lo} {la}\n",capture_output=True,text=True,env=ENV);e,n,_=r.stdout.split();return float(e),float(n)
def st2ll(e,n):
    r=subprocess.run([GT,"-s_srs","EPSG:3844","-t_srs","EPSG:4326"],input=f"{e} {n}\n",capture_output=True,text=True,env=ENV);a,b,_=r.stdout.split();return float(a),float(b)
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
est,nord=to_st(CLON,CLAT);half=KM*1000/2
e0=int((est-half)//1000);e1=int((est+half)//1000);n0=int((nord-half)//1000);n1=int((nord+half)//1000)
xll0=e0*1000;ytop0=(n1+1)*1000;W=(e1-e0+1)*TPX;Hh=(n1-n0+1)*TPX
mos=np.full((Hh,W),np.nan,np.float32);got=0
for nk in range(n0,n1+1):
    for ek in range(e0,e1+1):
        d=dl(nk,ek)
        if d is None: continue
        got+=1;ox=int((ek*1000-xll0)/CS);oy=int((ytop0-(nk+1)*1000)/CS);mos[oy:oy+TPX,ox:ox+TPX]=d[:TPX,:TPX]
print(f"mozaic DTM {W}x{Hh}px ({KM}km), {got} dale, goluri {np.isnan(mos).mean()*100:.0f}%",flush=True)
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
ys=list(range(0,Hh-wpx,stride));xs=list(range(0,W-wpx,stride))
print(f"{len(ys)}x{len(xs)} pozitii",flush=True)
heat=np.full((len(ys),len(xs)),0.,np.float32);batch=[];idx=[]
def flush():
    global batch,idx
    if not batch: return
    xb=torch.tensor(np.array(batch)).unsqueeze(1).float().to(dev)
    with torch.no_grad(): s=torch.sigmoid(net(xb)).cpu().numpy()
    for (iy,ix),sc in zip(idx,s): heat[iy,ix]=sc
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
hot=(heat>0.6).astype(float);print(f"celule >0.6: {int(hot.sum())}/{int((heat>0).sum())}",flush=True)
dens=np.asarray(Image.fromarray((hot*255).astype('uint8')).filter(ImageFilter.GaussianBlur(3)),float)
flat=sorted(((dens[iy,ix],iy,ix) for iy in range(dens.shape[0]) for ix in range(dens.shape[1])),reverse=True)
seen=[]
for d,iy,ix in flat:
    if d<25: break
    yy=ys[iy]+wpx//2;xx=xs[ix]+wpx//2;e=xll0+xx*CS;n=ytop0-yy*CS;lo,la=st2ll(e,n)
    if any((lo-s[0])**2+(la-s[1])**2<(0.005)**2 for s in seen): continue
    seen.append((lo,la,float(d)));print(f"  hot dens={d:.0f} @ {lo:.4f},{la:.4f}")
    if len(seen)>=6: break
json.dump(seen,open('/tmp/ro_hotspots.json','w'));print("hotspots -> /tmp/ro_hotspots.json")
