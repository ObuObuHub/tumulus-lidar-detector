#!/usr/bin/env python3
# heatmap_dtm_ms.py CLON CLAT KM [MODEL] [SCALES] — heatmap OARBĂ MULTI-SCARĂ pe LAKI3/Oltenia 0.5m.
# Scorează fiecare celulă la mai multe ferestre fizice (default 48,64,80,110 m) și ia MAX -> prinde movile
# de mărimi diferite (cele mici/scarificate pe care fereastra fixă 80m le rata). SCALES = listă m separată prin virgulă.
# -> review/heatmap_dtm_ms.png + stat per scară + combinat.
import os,sys,math,subprocess
import numpy as np
from PIL import Image,ImageFilter,ImageDraw,ImageFont
import torch,torch.nn as nn
H=os.path.dirname(os.path.dirname(os.path.abspath(__file__)));dev=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
CLON=float(sys.argv[1]);CLAT=float(sys.argv[2]);KM=float(sys.argv[3]) if len(sys.argv)>3 else 1.5
MODEL=sys.argv[4] if len(sys.argv)>4 else f'{H}/combined_cnn.pt'
SCALES=[float(x) for x in (sys.argv[5].split(',') if len(sys.argv)>5 else "48,64,80,110".split(','))]
STEP_M=12.0;CACHE="/tmp/laki3";CS=0.5;TPX=2000
APP=os.environ.get("QGIS_APP","/Applications/QGIS-final-4_0_3.app/Contents")
ENV=dict(os.environ,DYLD_FRAMEWORK_PATH=f"{APP}/Frameworks",PROJ_DATA=f"{APP}/Resources/qgis/proj",PROJ_LIB=f"{APP}/Resources/qgis/proj",GDAL_DATA=f"{APP}/Resources/qgis/gdal")
GT=f"{APP}/MacOS/gdaltransform"
def trans(p,s,t):
    r=subprocess.run([GT,"-s_srs",s,"-t_srs",t],input=f"{p[0]} {p[1]}\n",capture_output=True,text=True,env=ENV);a,b,_=r.stdout.split();return float(a),float(b)
def dl(nk,ek):
    p=f"{CACHE}/{nk}_{ek}.npy";return np.load(p) if os.path.exists(p) else None
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
mos=np.full((Hh,W),np.nan,np.float32);ntiles=0
for nk in range(n0,n1+1):
    for ek in range(e0,e1+1):
        d=dl(nk,ek)
        if d is None: continue
        ntiles+=1;ox=int((ek*1000-xll0)/CS);oy=int((ytop0-(nk+1)*1000)/CS);mos[oy:oy+TPX,ox:ox+TPX]=d[:TPX,:TPX]
if ntiles==0: print("EROARE: niciun tile laki3");sys.exit(1)
print(f"mozaic {W}x{Hh}px ({KM}km, 0.5m), scări {SCALES}m, model {os.path.basename(MODEL)}",flush=True)
f=int(round(2.0/CS))
class Net(nn.Module):
    def __init__(s):
        super().__init__();s.c=nn.Sequential(nn.Conv2d(1,16,3,2,1),nn.ReLU(),nn.Conv2d(16,32,3,2,1),nn.ReLU(),nn.Conv2d(32,64,3,2,1),nn.ReLU(),nn.AdaptiveAvgPool2d(1));s.f=nn.Linear(64,1)
    def forward(s,x): return s.f(s.c(x).flatten(1)).squeeze(1)
net=Net().to(dev);net.load_state_dict(torch.load(MODEL,map_location=dev,weights_only=True));net.eval()
def stamp(px,py,M):
    hw=int(M/2/CS);w=mos[py-hw:py+hw,px-hw:px+hw]
    if w.shape!=(2*hw,2*hw) or np.isnan(w).mean()>0.05: return None
    d2=downs(w,f);h=hs(d2,CS*f);lo,hi=np.percentile(h,2),np.percentile(h,98)
    if hi-lo<1e-6: return None
    return homog(np.asarray(Image.fromarray(np.clip((h-lo)/(hi-lo)*255,0,255).astype('uint8')).resize((128,128)),np.uint8))
step=max(1,int(STEP_M/CS));hwmax=int(max(SCALES)/2/CS)
gxs=list(range(hwmax,W-hwmax,step));gys=list(range(hwmax,Hh-hwmax,step))
gmax=np.zeros((len(gys),len(gxs)),np.float32)
for M in SCALES:
    batch=[];pos=[]
    for iy,py in enumerate(gys):
        for ix,px in enumerate(gxs):
            s=stamp(px,py,M)
            if s is not None: batch.append(s);pos.append((iy,ix))
    sc=[]
    X=torch.tensor(np.array(batch,dtype=np.uint8))
    with torch.no_grad():
        for k in range(0,len(X),512): sc.extend(torch.sigmoid(net(X[k:k+512].unsqueeze(1).float().to(dev)/255.)).cpu().numpy().tolist())
    grid=np.zeros_like(gmax)
    for (iy,ix),v in zip(pos,sc): grid[iy,ix]=v
    gmax=np.maximum(gmax,grid)
    print(f"  scara {M:.0f}m: %>=0.7 {(grid>=0.7).mean()*100:.1f}% | medie {grid.mean():.3f}",flush=True)
print(f"  COMBINAT (max): medie {gmax.mean():.3f} | mediană {np.median(gmax):.3f} | %>=0.7 {(gmax>=0.7).mean()*100:.1f}% | %>=0.9 {(gmax>=0.9).mean()*100:.1f}%",flush=True)
bg=hs(np.nan_to_num(mos,nan=float(np.nanmin(mos))),CS);bg=(np.clip((bg-np.percentile(bg,2))/(np.percentile(bg,98)-np.percentile(bg,2)+1e-6),0,1)*255).astype('uint8')
field=np.asarray(Image.fromarray((gmax*255).astype('uint8')).resize((W,Hh),Image.BICUBIC),np.float32)/255.
s4=field*4;Rr=np.clip(s4-2,0,1);Gg=np.clip(np.minimum(s4,4-s4),0,1);Bb=np.clip(2-s4,0,1)
cmap=np.stack([Rr,Gg,Bb],-1)*255;alpha=np.clip((field-0.25)/0.5,0,1)[...,None]*0.78
base=np.stack([bg,bg,bg],-1).astype(np.float32);out=(base*(1-alpha)+cmap*alpha).astype('uint8');ov=Image.fromarray(out)
dr=ImageDraw.Draw(ov)
try: ft=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial Bold.ttf',max(13,W//90))
except: ft=ImageFont.load_default()
dr.text((6,4),f"HEATMAP MULTI-SCARA {SCALES}m | %>=.7 {(gmax>=0.7).mean()*100:.1f}% | rosu=aprinde la ORICE scara",fill=(255,255,255),font=ft)
mx=max(W,Hh);disp=ov.resize((int(W*min(1,1400/mx)),int(Hh*min(1,1400/mx))))
disp.save(f"{H}/review/heatmap_dtm_ms.png");print(f"  -> review/heatmap_dtm_ms.png")
