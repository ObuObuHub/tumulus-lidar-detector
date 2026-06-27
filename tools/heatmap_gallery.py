#!/usr/bin/env python3
# heatmap_gallery.py — galerie de close-up-uri HEATMAP (scor jet peste hillshade) pt o listă de candidați,
# ca să arăt MULTE exemple de „ce vede modelul". Crop ~CROP m, scor dens single-scară 80m.
import os,math,subprocess,zipfile
import numpy as np
from PIL import Image,ImageFilter,ImageDraw,ImageFont
import torch,torch.nn as nn
H=os.path.dirname(os.path.dirname(os.path.abspath(__file__)));dev=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
CACHE="/tmp/laki3";CS=0.5;TPX=2000;CROP=520;DISP=300;STEP=8
APP=os.environ.get("QGIS_APP","/Applications/QGIS-final-4_0_3.app/Contents")
ENV=dict(os.environ,DYLD_FRAMEWORK_PATH=f"{APP}/Frameworks",PROJ_DATA=f"{APP}/Resources/qgis/proj",PROJ_LIB=f"{APP}/Resources/qgis/proj",GDAL_DATA=f"{APP}/Resources/qgis/gdal")
GTb=f"{APP}/MacOS/gdaltransform"
def trans(lo,la):
    r=subprocess.run([GTb,"-s_srs","EPSG:4326","-t_srs","EPSG:3844"],input=f"{lo} {la}\n",capture_output=True,text=True,env=ENV);a,b,_=r.stdout.split();return float(a),float(b)
def load_one(nk,ek):
    p=f"{CACHE}/{nk}_{ek}.npy"
    if os.path.exists(p):
        try:return np.load(p)
        except:pass
    z=f"{CACHE}/{nk}_{ek}.zip"
    if not os.path.exists(z):subprocess.run(["curl","-s","--max-time","120","-o",z,f"https://geoportal.ancpi.ro/laki3_mnt/zip/{nk}_{ek}.zip"],check=False)
    try:zf=zipfile.ZipFile(z)
    except:return None
    asc=[n for n in zf.namelist() if n.lower().endswith('.asc')]
    if not asc:return None
    raw=zf.read(asc[0]).decode('latin-1').replace(',','.');lines=raw.split('\n');hdr={};i=0
    while i<len(lines):
        pp=lines[i].split()
        if len(pp)>=2 and pp[0].lower() in('ncols','nrows','xllcorner','yllcorner','cellsize','nodata_value'):hdr[pp[0].lower()]=float(pp[1]);i+=1
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
f=int(round(2.0/CS));hw=int(40/CS)
def jet(s):
    r=np.clip(1.5-np.abs(4*s-3),0,1);g=np.clip(1.5-np.abs(4*s-2),0,1);b=np.clip(1.5-np.abs(4*s-1),0,1)
    return np.stack([r,g,b],-1)
def mosaic(est,nord,half):
    e0=int((est-half)//1000);e1=int((est+half)//1000);n0=int((nord-half)//1000);n1=int((nord+half)//1000)
    xll0=e0*1000;ytop0=(n1+1)*1000;W=(e1-e0+1)*TPX;Hh=(n1-n0+1)*TPX;mos=np.full((Hh,W),np.nan,np.float32)
    for nk in range(n0,n1+1):
        for ek in range(e0,e1+1):
            d=load_one(nk,ek)
            if d is None:continue
            ox=int((ek*1000-xll0)/CS);oy=int((ytop0-(nk+1)*1000)/CS);mos[oy:oy+TPX,ox:ox+TPX]=d[:TPX,:TPX]
    return mos,xll0,ytop0
def cell(lo,la,label):
    est,nord=trans(lo,la);mos,xll0,ytop0=mosaic(est,nord,700)
    cpx=int((est-xll0)/CS);cpy=int((ytop0-nord)/CS);ch=int(CROP/2/CS)
    # dense score over crop
    gxs=list(range(cpx-ch,cpx+ch,STEP));gys=list(range(cpy-ch,cpy+ch,STEP))
    batch=[];pos=[]
    for py in gys:
        for px in gxs:
            w=mos[py-hw:py+hw,px-hw:px+hw]
            if w.shape!=(2*hw,2*hw) or np.isnan(w).mean()>0.05: continue
            d2=downs(np.nan_to_num(w,nan=np.nanmedian(w)),f);h=hs(d2,CS*f);lo2,hi2=np.percentile(h,2),np.percentile(h,98)
            if hi2-lo2<1e-6: continue
            batch.append(homog(np.asarray(Image.fromarray(np.clip((h-lo2)/(hi2-lo2)*255,0,255).astype('uint8')).resize((128,128)),np.uint8)));pos.append((py,px))
    grid=np.zeros((len(gys),len(gxs)),np.float32)
    if batch:
        X=torch.tensor(np.array(batch,dtype=np.uint8));sc=[]
        with torch.no_grad():
            for k in range(0,len(X),512): sc.extend(torch.sigmoid(net(X[k:k+512].unsqueeze(1).float().to(dev)/255.)).cpu().numpy().tolist())
        gi={(py,px):v for (py,px),v in zip(pos,sc)}
        for iy,py in enumerate(gys):
            for ix,px in enumerate(gxs): grid[iy,ix]=gi.get((py,px),0.0)
    # hillshade crop (clean, native)
    sub=mos[cpy-ch:cpy+ch,cpx-ch:cpx+ch]
    h=hs(np.nan_to_num(sub,nan=np.nanmedian(sub)),CS);lo2,hi2=np.percentile(h,2),np.percentile(h,98);bg=np.clip((h-lo2)/(hi2-lo2),0,1)
    field=np.asarray(Image.fromarray((grid*255).astype('uint8')).resize((bg.shape[1],bg.shape[0]),Image.BICUBIC),np.float32)/255.
    rgb=np.stack([bg,bg,bg],-1)*255;col=jet(field)*255;alpha=(field**1.3*0.75)[...,None]
    out=(rgb*(1-alpha)+col*alpha).astype('uint8')
    im=Image.fromarray(out).resize((DISP,DISP),Image.BICUBIC)
    return im,float(grid.max())
import json,sys
TITLE=os.environ.get('GTITLE',"Exemple close-up (heatmap ~520m): ce vede modelul pe movile, 0.5m Oltenia")
OUT=os.environ.get('GOUT',f"{H}/review/heatmap_examples.jpg")
LBLCOL=tuple(int(x) for x in os.environ.get('GCOL','0,255,120').split(','))
if os.environ.get('GJSON'):
    CANDS=[(c[0],c[1],c[2]) for c in json.load(open(os.environ['GJSON']))]
else:
    # Pass candidates via GJSON env (list of [label, lon, lat]). No coordinates are hardcoded here
    # so the published tool does not ship any site locations.
    raise SystemExit("Set GJSON=<file.json> with [[label,lon,lat],...] — no default coordinates are shipped.")
cols=4;rows=(len(CANDS)+cols-1)//cols
G=Image.new('RGB',(cols*DISP,rows*DISP+24),(15,15,15));dr=ImageDraw.Draw(G)
try: fnt=ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf",15)
except: fnt=ImageFont.load_default()
dr.text((6,5),TITLE,fill=(255,255,0),font=fnt)
for k,(lab,lo,la) in enumerate(CANDS):
    im,mx=cell(lo,la,lab);x=(k%cols)*DISP;y=(k//cols)*DISP+24;G.paste(im,(x,y))
    dr.text((x+5,y+4),f"{lab}  s{mx:.2f}",fill=LBLCOL,font=fnt)
    print(f"{lab}: max {mx:.2f}",flush=True)
G.save(OUT,quality=90)
print(f"-> {OUT}")
