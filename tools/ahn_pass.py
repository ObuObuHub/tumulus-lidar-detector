#!/usr/bin/env python3
# ahn_pass.py CLON CLAT KM [MODEL] — rulează modelul pe LiDAR OLANDEZ AHN4 DTM 0.5m (test independent).
# Fetch AHN WCS (coverageId=dtm_05m, EPSG:28992) -> GeoTIFF -> gdal AAIGrid -> array -> pipeline IDENTIC
# (downsample 2m, hillshade 6-dir, fereastră 80m, homog, scor). Heatmap + marcaje (env BARROWS="lat,lon;..").
import sys,os,math,subprocess
import numpy as np
from PIL import Image,ImageFilter,ImageDraw,ImageFont
import torch,torch.nn as nn
H=os.path.dirname(os.path.dirname(os.path.abspath(__file__)));dev=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
CLON=float(sys.argv[1]);CLAT=float(sys.argv[2]);KM=float(sys.argv[3]) if len(sys.argv)>3 else 2.0
MODEL=sys.argv[4] if len(sys.argv)>4 else f'{H}/combined_cnn.pt'
CS=0.5
APP=os.environ.get("QGIS_APP","/Applications/QGIS-final-4_0_3.app/Contents")
ENV=dict(os.environ,DYLD_FRAMEWORK_PATH=f"{APP}/Frameworks",PROJ_DATA=f"{APP}/Resources/qgis/proj",PROJ_LIB=f"{APP}/Resources/qgis/proj",GDAL_DATA=f"{APP}/Resources/qgis/gdal")
GT=f"{APP}/MacOS/gdaltransform";GTR=f"{APP}/MacOS/gdal_translate"
def trans(lon,lat,s="EPSG:4326",t="EPSG:28992"):
    r=subprocess.run([GT,"-s_srs",s,"-t_srs",t],input=f"{lon} {lat}\n",capture_output=True,text=True,env=ENV);a,b,_=r.stdout.split();return float(a),float(b)
est,nord=trans(CLON,CLAT);half=KM*1000/2
x0,x1=est-half,est+half;y0,y1=nord-half,nord+half
url=(f"https://service.pdok.nl/rws/ahn/wcs/v1_0?service=WCS&version=2.0.1&request=GetCoverage"
     f"&coverageId=dtm_05m&subset=x({x0:.1f},{x1:.1f})&subset=y({y0:.1f},{y1:.1f})&format=image/tiff")
tif="/tmp/ahn_pass.tif";asc="/tmp/ahn_pass.asc"
print(f"fetch AHN dtm_05m {KM}km @ {CLON},{CLAT} (EPSG28992 {est:.0f},{nord:.0f})...",flush=True)
subprocess.run(["curl","-s","--max-time","120","-o",tif,url],check=False)
sz=os.path.getsize(tif) if os.path.exists(tif) else 0
if sz<10000: sys.exit(f"EROARE fetch AHN ({sz}B): {open(tif).read()[:200] if sz else 'gol'}")
if os.path.exists(asc): os.remove(asc)
subprocess.run([GTR,"-of","AAIGrid",tif,asc],capture_output=True,env=ENV)
L=open(asc).read().split('\n');hdr={};i=0
while i<len(L) and L[i].split() and L[i].split()[0].lower() in('ncols','nrows','xllcorner','yllcorner','cellsize','nodata_value'):
    k,v=L[i].split()[:2];hdr[k.lower()]=float(v);i+=1
nc,nr=int(hdr['ncols']),int(hdr['nrows']);xll=hdr['xllcorner'];yll=hdr['yllcorner'];ce=hdr['cellsize']
dem=np.fromstring(' '.join(L[i:]),sep=' ',dtype=np.float32)[:nc*nr].reshape(nr,nc)
dem[dem>1e30]=np.nan;dem[dem==hdr.get('nodata_value',-9999)]=np.nan
ytop=yll+nr*ce
print(f"DTM {nc}x{nr} @ {ce}m | elev {np.nanmin(dem):.1f}-{np.nanmax(dem):.1f}m | NaN {np.isnan(dem).mean()*100:.0f}%",flush=True)
def hs(d,cs,azs=(315,45,135,225,270,0),alt=35):
    gy,gx=np.gradient(d,cs);sl=np.arctan(np.hypot(gx,gy));asp=np.arctan2(-gy,gx);o=np.zeros_like(d);ar=math.radians(alt)
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
net=Net().to(dev);net.load_state_dict(torch.load(MODEL,map_location=dev,weights_only=True));net.eval()
f=int(round(2.0/CS));hw=int(40/CS);W=nc;Hh=nr
def stamp(px,py,m=80):
    h=int(m/2/CS);w=dem[py-h:py+h,px-h:px+h]
    if w.shape!=(2*h,2*h) or np.isnan(w).mean()>0.05: return None
    d2=downs(w,f);sh=hs(d2,CS*f);lo,hi=np.percentile(sh,2),np.percentile(sh,98)
    if hi-lo<1e-6: return None
    return homog(np.asarray(Image.fromarray(np.clip((sh-lo)/(hi-lo)*255,0,255).astype('uint8')).resize((128,128)),np.uint8))
STEP=int(10/CS);gxs=list(range(hw,W-hw,STEP));gys=list(range(hw,Hh-hw,STEP))
grid=np.zeros((len(gys),len(gxs)),np.float32);batch=[];pos=[]
for iy,py in enumerate(gys):
    for ix,px in enumerate(gxs):
        s=stamp(px,py)
        if s is not None: batch.append(s);pos.append((iy,ix))
X=torch.tensor(np.array(batch,dtype=np.uint8));sc=[]
with torch.no_grad():
    for k in range(0,len(X),512): sc.extend(torch.sigmoid(net(X[k:k+512].unsqueeze(1).float().to(dev)/255.)).cpu().numpy().tolist())
for (iy,ix),v in zip(pos,sc): grid[iy,ix]=v
v=grid[grid>0]
print(f"{v.size} celule | mediană {np.median(v):.3f} | %>=0.7 {(v>=0.7).mean()*100:.1f}% | %>=0.9 {(v>=0.9).mean()*100:.1f}%",flush=True)
# NMS detecții
flat=[(grid[iy,ix],gxs[ix],gys[iy]) for iy in range(len(gys)) for ix in range(len(gxs)) if grid[iy,ix]>=0.85]
flat.sort(reverse=True);kept=[]
for s,px,py in flat:
    if all((px-q[1])**2+(py-q[2])**2>(80/CS)**2 for q in kept): kept.append((s,px,py))
print(f"detecții >=0.85 (NMS 80m): {len(kept)}",flush=True)
# render
def jet(s):
    r=np.clip(1.5-np.abs(4*s-3),0,1);g=np.clip(1.5-np.abs(4*s-2),0,1);b=np.clip(1.5-np.abs(4*s-1),0,1);return np.stack([r,g,b],-1)
field=np.asarray(Image.fromarray((grid*255).astype('uint8')).resize((W,Hh),Image.BICUBIC),np.float32)/255.
bgs=hs(np.nan_to_num(dem,nan=float(np.nanmin(dem))),CS);bg=np.clip((bgs-np.percentile(bgs,2))/(np.percentile(bgs,98)-np.percentile(bgs,2)+1e-6),0,1)
rgb=(np.stack([bg,bg,bg],-1)*255).astype(np.float32);col=jet(field)*255;alpha=(np.clip((field-0.5)/0.5,0,1)*0.75)[...,None]
out=(rgb*(1-alpha)+col*alpha).astype(np.uint8);img=Image.fromarray(out);dr=ImageDraw.Draw(img)
try: fnt=ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf",22)
except: fnt=ImageFont.load_default()
print("--- scor model la movilele cunoscute (BARROWS) ---")
for item in os.environ.get('BARROWS','').split(';'):
    if not item.strip(): continue
    la,lo=item.split(',');la=float(la);lo=float(lo);e,n=trans(lo,la);px=(e-xll)/CS;py=(ytop-n)/CS
    if 0<=px<W and 0<=py<Hh:
        s=stamp(int(px),int(py));sv=float(torch.sigmoid(net(torch.tensor(s[None,None],dtype=torch.float32).to(dev)/255.)).item()) if s is not None else float('nan')
        print(f"  {la},{lo} -> {sv:.3f}")
        dr.ellipse([px-24,py-24,px+24,py+24],outline=(255,255,255),width=3)
os.makedirs(f"{H}/review",exist_ok=True)
# downscale pt afișare dacă mare
disp=img
if W>1600: disp=img.resize((1600,int(1600*Hh/W)))
disp.save(f"{H}/review/ahn_pass.png");print(f"-> review/ahn_pass.png")
# clean hillshade pt comparație
ci=(bg*255).astype(np.uint8);Image.fromarray(ci).resize(disp.size).save(f"{H}/review/ahn_clean.png");print("-> review/ahn_clean.png")
