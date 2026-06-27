#!/usr/bin/env python3
# eval_montage.py CLON CLAT KM — set de TEST etichetabil: amestec de detecții scor-mare (tumuli+FP) +
# celule random scor-mic (negative) dintr-o zonă cu tumuli. Numerotat, FĂRĂ scor afișat (etichetare
# nepărtinitoare). Andrei marchează cu roșu tumulii reali -> /tmp/eval_map.csv pt evaluare precizie/recall
# + testarea filtrului de formă. Folosește modelul homogenizat (load identic ca train).
import os,sys,math,subprocess,zipfile,csv,random
import numpy as np
from PIL import Image,ImageFilter,ImageDraw,ImageFont
import torch,torch.nn as nn
random.seed(7)
H=os.path.dirname(os.path.dirname(os.path.abspath(__file__)));dev=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
CLON=float(sys.argv[1]);CLAT=float(sys.argv[2]);KM=float(sys.argv[3]) if len(sys.argv)>3 else 5.0
CACHE="/tmp/laki3";CS=0.5;TPX=2000
APP=os.environ.get("QGIS_APP","/Applications/QGIS-final-4_0_3.app/Contents")
ENV=dict(os.environ,DYLD_FRAMEWORK_PATH=f"{APP}/Frameworks",PROJ_DATA=f"{APP}/Resources/qgis/proj",PROJ_LIB=f"{APP}/Resources/qgis/proj",GDAL_DATA=f"{APP}/Resources/qgis/gdal")
GT=f"{APP}/MacOS/gdaltransform"
def trans(p,s,t):
    r=subprocess.run([GT,"-s_srs",s,"-t_srs",t],input=f"{p[0]} {p[1]}\n",capture_output=True,text=True,env=ENV);a,b,_=r.stdout.split();return float(a),float(b)
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
    a2=np.asarray(Image.fromarray(a).filter(ImageFilter.GaussianBlur(0.8)),np.uint8);cdf=np.bincount(a2.ravel(),minlength=256).astype(np.float64).cumsum()
    return (cdf[a2]/cdf[-1]*255).astype(np.uint8) if cdf[-1]>0 else a2
est,nord=trans((CLON,CLAT),"EPSG:4326","EPSG:3844");half=KM*1000/2
e0=int((est-half)//1000);e1=int((est+half)//1000);n0=int((nord-half)//1000);n1=int((nord+half)//1000)
xll0=e0*1000;ytop0=(n1+1)*1000;W=(e1-e0+1)*TPX;Hh=(n1-n0+1)*TPX
mos=np.full((Hh,W),np.nan,np.float32)
for nk in range(n0,n1+1):
    for ek in range(e0,e1+1):
        d=dl(nk,ek)
        if d is None: continue
        ox=int((ek*1000-xll0)/CS);oy=int((ytop0-(nk+1)*1000)/CS);mos[oy:oy+TPX,ox:ox+TPX]=d[:TPX,:TPX]
print(f"mozaic {W}x{Hh} ({KM}km)",flush=True)
class Net(nn.Module):
    def __init__(s):
        super().__init__();s.c=nn.Sequential(nn.Conv2d(1,16,3,2,1),nn.ReLU(),nn.Conv2d(16,32,3,2,1),nn.ReLU(),nn.Conv2d(32,64,3,2,1),nn.ReLU(),nn.AdaptiveAvgPool2d(1));s.f=nn.Linear(64,1)
    def forward(s,x): return s.f(s.c(x).flatten(1)).squeeze(1)
net=Net().to(dev);net.load_state_dict(torch.load(f'{H}/combined_cnn.pt',weights_only=True));net.eval()
f=int(round(2.0/CS));wpx=int(80/CS);stride=int(40/CS)
def stamp_raw(px,py):
    w=mos[py-wpx//2:py+wpx//2,px-wpx//2:px+wpx//2]
    if w.shape!=(wpx,wpx) or np.isnan(w).mean()>0.05: return None
    d2=downs(w,f);h=hs(d2,CS*f);lo,hi=np.percentile(h,2),np.percentile(h,98)
    if hi-lo<1e-6: return None
    return np.asarray(Image.fromarray(np.clip((h-lo)/(hi-lo)*255,0,255).astype('uint8')).resize((128,128)),np.uint8)
ys=list(range(wpx//2,Hh-wpx//2,stride));xs=list(range(wpx//2,W-wpx//2,stride))
cells=[];batch=[];pos=[]
def flush():
    global batch,pos
    if not batch: return
    xb=torch.tensor(np.array([homog(b) for b in batch])).unsqueeze(1).float().to(dev)/255.
    with torch.no_grad(): sc=torch.sigmoid(net(xb)).cpu().numpy()
    for raw,(e,n),s in zip(batch,pos,sc): cells.append((float(s),e,n,raw))
    batch=[];pos=[]
for yy in ys:
    for xx in xs:
        raw=stamp_raw(xx,yy)
        if raw is None: continue
        batch.append(raw);pos.append((xll0+xx*CS,ytop0-yy*CS))
        if len(batch)>=512: flush()
flush()
print(f"celule scorate: {len(cells)}",flush=True)
# eșantionare: high (>0.7) dedup 80m + random low (<0.25) = negative
cells.sort(key=lambda c:-c[0])
hi=[];
for s,e,n,r in cells:
    if s<0.7: break
    if any((e-k[1])**2+(n-k[2])**2<80*80 for k in hi): continue
    hi.append((s,e,n,r))
random.shuffle(hi); hi=hi[:40]
lowpool=[c for c in cells if c[0]<0.25]; random.shuffle(lowpool)
low=[];
for s,e,n,r in lowpool:
    if any((e-k[1])**2+(n-k[2])**2<120*120 for k in low): continue
    low.append((s,e,n,r))
    if len(low)>=24: break
mix=hi+low; random.shuffle(mix)
# mapping + montaj numerotat FĂRĂ scor
mp=open('/tmp/eval_map.csv','w');mw=csv.writer(mp);mw.writerow(['idx','lon','lat','score'])
cols=8;rows=(len(mix)+cols-1)//cols;cell=132;M=Image.new('RGB',(cols*cell,rows*cell),(15,15,15));dr=ImageDraw.Draw(M)
try: ft=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial Bold.ttf',14)
except: ft=ImageFont.load_default()
for i,(s,e,n,raw) in enumerate(mix,1):
    lon,lat=trans((e,n),"EPSG:3844","EPSG:4326");mw.writerow([i,f"{lon:.5f}",f"{lat:.5f}",f"{s:.3f}"])
    x=((i-1)%cols)*cell;y=((i-1)//cols)*cell
    M.paste(Image.fromarray(raw).convert('RGB').resize((cell-4,cell-20)),(x+2,y+18))
    dr.text((x+3,y+2),f"#{i}",fill=(255,255,0),font=ft)
mp.close();M.save(f'{H}/review/eval_set.png')
print(f"-> review/eval_set.png ({len(mix)} mostre: {len(hi)} scor-mare[tumuli+FP] + {len(low)} random[negative]) + /tmp/eval_map.csv")
