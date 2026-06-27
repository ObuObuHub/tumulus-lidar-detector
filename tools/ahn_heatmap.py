#!/usr/bin/env python3
# ahn_heatmap.py CLON CLAT KM [MODEL] [SCALES] — heatmap NL: fetch AHN DTM 0.5m + hillshade NATIV + scan dens model
# (jet overlay) + MARCHEAZĂ movilele OSM (/tmp/nl_barrows.json) ca cercuri. -> review/ahn_heatmap.png + _clean.png
import sys,os,math,subprocess,json
import numpy as np
from PIL import Image,ImageFilter,ImageDraw,ImageFont
import torch,torch.nn as nn
H=os.path.dirname(os.path.dirname(os.path.abspath(__file__)));dev=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
CLON=float(sys.argv[1]);CLAT=float(sys.argv[2]);KM=float(sys.argv[3]) if len(sys.argv)>3 else 1.5
MODEL=sys.argv[4] if len(sys.argv)>4 else f'{H}/combined_cnn.pt'
SCALES=[float(x) for x in (sys.argv[5].split(',') if len(sys.argv)>5 else ['28','32','40'])]
MPP=0.5;STEP_M=5
APP=os.environ.get("QGIS_APP","/Applications/QGIS-final-4_0_3.app/Contents")
ENV=dict(os.environ,DYLD_FRAMEWORK_PATH=f"{APP}/Frameworks",PROJ_DATA=f"{APP}/Resources/qgis/proj",PROJ_LIB=f"{APP}/Resources/qgis/proj",GDAL_DATA=f"{APP}/Resources/qgis/gdal")
GT=f"{APP}/MacOS/gdaltransform";GTR=f"{APP}/MacOS/gdal_translate"
def trans(lon,lat,s="EPSG:4326",t="EPSG:28992"):
    r=subprocess.run([GT,"-s_srs",s,"-t_srs",t],input=f"{lon} {lat}\n",capture_output=True,text=True,env=ENV);a,b,_=r.stdout.split();return float(a),float(b)
est,nord=trans(CLON,CLAT);half=KM*1000/2
x0,x1=est-half,est+half;y0,y1=nord-half,nord+half
url=(f"https://service.pdok.nl/rws/ahn/wcs/v1_0?service=WCS&version=2.0.1&request=GetCoverage"
     f"&coverageId=dtm_05m&subset=x({x0:.1f},{x1:.1f})&subset=y({y0:.1f},{y1:.1f})&format=image/tiff")
tif="/tmp/ahn_hm.tif";asc="/tmp/ahn_hm.asc"
print(f"fetch AHN {KM}km @ {CLON},{CLAT}...",flush=True)
subprocess.run(["curl","-s","--max-time","180","-o",tif,url],check=False)
if not os.path.exists(tif) or os.path.getsize(tif)<10000: sys.exit("EROARE fetch AHN")
if os.path.exists(asc): os.remove(asc)
subprocess.run([GTR,"-of","AAIGrid",tif,asc],capture_output=True,env=ENV)
L=open(asc).read().split('\n');hdr={};i=0
while i<len(L) and L[i].split() and L[i].split()[0].lower() in('ncols','nrows','xllcorner','yllcorner','cellsize','nodata_value'):
    k,v=L[i].split()[:2];hdr[k.lower()]=float(v);i+=1
nc,nr=int(hdr['ncols']),int(hdr['nrows']);xll=hdr['xllcorner'];yll=hdr['yllcorner'];ce=hdr['cellsize']
dem=np.fromstring(' '.join(L[i:]),sep=' ',dtype=np.float32)[:nc*nr].reshape(nr,nc)
dem[dem>1e30]=np.nan;dem[dem==hdr.get('nodata_value',-9999)]=np.nan
ytop=yll+nr*ce;dem=np.nan_to_num(dem,nan=float(np.nanmedian(dem)))
def hs(d,cs,azs=(315,45,135,225,270,0),alt=35):
    gy,gx=np.gradient(d,cs);sl=np.arctan(np.hypot(gx,gy));asp=np.arctan2(-gy,gx);o=np.zeros_like(d);ar=math.radians(alt)
    for az in azs: azr=math.radians(360-az+90);o+=np.clip(np.sin(ar)*np.cos(sl)+np.cos(ar)*np.sin(sl)*np.cos(azr-asp),0,1)
    return o/len(azs)
sh=hs(dem,ce);lo,hi=np.percentile(sh,2),np.percentile(sh,98)
A=np.clip((sh-lo)/(hi-lo+1e-9)*255,0,255).astype(np.uint8);Hh,Ww=A.shape
def homog(a):
    a2=np.asarray(Image.fromarray(a).filter(ImageFilter.GaussianBlur(0.8)),np.uint8);cdf=np.bincount(a2.ravel(),minlength=256).astype(np.float64).cumsum()
    return (cdf[a2]/cdf[-1]*255).astype(np.uint8) if cdf[-1]>0 else a2
def stamp(cx,cy,win,content):
    h=win//2;w=A[cy-h:cy+h,cx-h:cx+h]
    if w.shape!=(2*h,2*h) or w.std()<3: return None
    plo,phi=np.percentile(w,2),np.percentile(w,98);a=np.clip((w-plo)/(phi-plo+1e-6),0,1)
    a2=np.asarray(Image.fromarray((a*255).astype('uint8')).resize((content,content)),np.uint8)
    return homog(np.asarray(Image.fromarray(a2).resize((128,128)),np.uint8))
class Net(nn.Module):
    def __init__(s):
        super().__init__();s.c=nn.Sequential(nn.Conv2d(1,16,3,2,1),nn.ReLU(),nn.Conv2d(16,32,3,2,1),nn.ReLU(),nn.Conv2d(32,64,3,2,1),nn.ReLU(),nn.AdaptiveAvgPool2d(1));s.f=nn.Linear(64,1)
    def forward(s,x): return s.f(s.c(x).flatten(1)).squeeze(1)
net=Net().to(dev);net.load_state_dict(torch.load(MODEL,map_location=dev,weights_only=True));net.eval()
step=max(2,int(STEP_M/MPP));gxs=list(range(0,Ww,step));gys=list(range(0,Hh,step))
best=np.zeros((len(gys),len(gxs)),np.float32)
for M in SCALES:
    win=int(round(M/MPP));content=max(8,int(round(M/2.0)));h=win//2;batch=[];posg=[]
    for iy,cy in enumerate(gys):
        for ix,cx in enumerate(gxs):
            if cx-h<0 or cy-h<0 or cx+h>Ww or cy+h>Hh: continue
            s=stamp(cx,cy,win,content)
            if s is not None: batch.append(s);posg.append((iy,ix))
    if not batch: continue
    X=torch.tensor(np.array(batch,dtype=np.uint8));sc=[]
    with torch.no_grad():
        for k in range(0,len(X),512): sc.extend(torch.sigmoid(net(X[k:k+512].unsqueeze(1).float().to(dev)/255.)).cpu().numpy().tolist())
    g=np.zeros_like(best)
    for (iy,ix),v in zip(posg,sc): g[iy,ix]=v
    best=np.maximum(best,g)
v=best[best>0];print(f"heatmap {nc}px scări {SCALES} | %>=0.7 {(v>=0.7).mean()*100:.1f}%",flush=True)
def jet(s):
    r=np.clip(1.5-np.abs(4*s-3),0,1);g=np.clip(1.5-np.abs(4*s-2),0,1);b=np.clip(1.5-np.abs(4*s-1),0,1);return np.stack([r,g,b],-1)
field=np.asarray(Image.fromarray((best*255).astype('uint8')).resize((Ww,Hh),Image.BICUBIC),np.float32)/255.
bg=np.clip((A-np.percentile(A,2))/(np.percentile(A,98)-np.percentile(A,2)+1e-6),0,1)
rgb=(np.stack([bg,bg,bg],-1)*255).astype(np.float32);col=jet(field)*255;alpha=(np.clip((field-0.5)/0.5,0,1)*0.7)[...,None]
out=(rgb*(1-alpha)+col*alpha).astype(np.uint8);img=Image.fromarray(out);dr=ImageDraw.Draw(img)
clean=Image.fromarray((bg*255).astype(np.uint8)).convert('RGB');drc=ImageDraw.Draw(clean)
# marchează movilele OSM din extent
barrows=json.load(open('/tmp/nl_barrows.json'));nm=0
inb=[(la,lo) for la,lo,_ in barrows if abs(la-CLAT)<KM/111.0 and abs(lo-CLON)<KM/(111.0*math.cos(math.radians(CLAT)))]
for la,lo in inb:
    e,n=trans(lo,la);px=(e-xll)/ce;py=(ytop-n)/ce
    if 8<=px<Ww-8 and 8<=py<Hh-8:
        dr.ellipse([px-13,py-13,px+13,py+13],outline=(0,255,0),width=2)
        drc.ellipse([px-13,py-13,px+13,py+13],outline=(0,255,0),width=2);nm+=1
print(f"movile OSM marcate: {nm}",flush=True)
os.makedirs(f"{H}/review",exist_ok=True)
def save(im,nm2):
    d=im
    if Ww>1500: d=im.resize((1500,int(1500*Hh/Ww)))
    d.save(f"{H}/review/{nm2}")
save(img,"ahn_heatmap.png");save(clean,"ahn_heatmap_clean.png")
print("-> review/ahn_heatmap.png + _clean.png",flush=True)
