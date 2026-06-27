#!/usr/bin/env python3
# grad_cam.py [MODEL] — Grad-CAM pe stampe Catane (tumuli reali + FP) ca să vedem UNDE se uită modelul.
# Centru = semnal real (movilă); margini/fundal = shortcut/artefact (feedback cercetător ML). Catane cache.
import os,sys,math,subprocess,csv
import numpy as np
from PIL import Image,ImageFilter,ImageDraw,ImageFont
import torch,torch.nn as nn,torch.nn.functional as Fn
H=os.path.expanduser('~/lidar-match');dev=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
MODEL=sys.argv[1] if len(sys.argv)>1 else f'{H}/combined_mdhbal.pt'
CACHE="/tmp/laki3";CS=0.5;TPX=2000
APP="/Applications/QGIS-final-4_0_3.app/Contents"
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
# mozaic Catane
est,nord=trans((23.4181,43.9141),"EPSG:4326","EPSG:3844");half=2500
e0=int((est-half)//1000);e1=int((est+half)//1000);n0=int((nord-half)//1000);n1=int((nord+half)//1000)
xll0=e0*1000;ytop0=(n1+1)*1000;W=(e1-e0+1)*TPX;Hh=(n1-n0+1)*TPX
mos=np.full((Hh,W),np.nan,np.float32)
for nk in range(n0,n1+1):
    for ek in range(e0,e1+1):
        d=dl(nk,ek)
        if d is None: continue
        ox=int((ek*1000-xll0)/CS);oy=int((ytop0-(nk+1)*1000)/CS);mos[oy:oy+TPX,ox:ox+TPX]=d[:TPX,:TPX]
f=int(round(2.0/CS));hw=int(40/CS)
def stamp(lo,la):
    e,n=trans((lo,la),"EPSG:4326","EPSG:3844");px=int((e-xll0)/CS);py=int((ytop0-n)/CS);w=mos[py-hw:py+hw,px-hw:px+hw]
    if w.shape!=(2*hw,2*hw) or np.isnan(w).mean()>0.05: return None
    d2=downs(w,f);h=hs(d2,CS*f);lo2,hi2=np.percentile(h,2),np.percentile(h,98)
    raw=np.asarray(Image.fromarray(np.clip((h-lo2)/(hi2-lo2)*255,0,255).astype('uint8')).resize((128,128)),np.uint8)
    return homog(raw)
class Net(nn.Module):
    def __init__(s):
        super().__init__();s.c=nn.Sequential(nn.Conv2d(1,16,3,2,1),nn.ReLU(),nn.Conv2d(16,32,3,2,1),nn.ReLU(),nn.Conv2d(32,64,3,2,1),nn.ReLU(),nn.AdaptiveAvgPool2d(1));s.f=nn.Linear(64,1)
    def forward(s,x): return s.f(s.c(x).flatten(1)).squeeze(1)
net=Net().to(dev);net.load_state_dict(torch.load(MODEL,map_location=dev,weights_only=True));net.eval()
# Grad-CAM pe ultimul conv (s.c[4]=Conv32->64); hook activare + gradient
acts={};grads={}
def fh(m,i,o): acts['v']=o.detach()
def bh(m,gi,go): grads['v']=go[0].detach()
net.c[4].register_forward_hook(fh);net.c[4].register_full_backward_hook(bh)
def gradcam(img128):
    x=torch.tensor(img128).unsqueeze(0).unsqueeze(0).float().to(dev)/255.;x.requires_grad_(True)
    net.zero_grad();out=net(x);out.backward()
    A=acts['v'][0];G=grads['v'][0]            # [64,h,w]
    wts=G.mean(dim=(1,2));cam=Fn.relu((wts[:,None,None]*A).sum(0))
    cam=cam.cpu().numpy();cam=cam/(cam.max()+1e-9)
    cam=np.asarray(Image.fromarray((cam*255).astype('uint8')).resize((128,128)))
    return float(torch.sigmoid(out).item()),cam
# truth + FP Catane
emap={int(r['idx']):(float(r['lon']),float(r['lat'])) for r in csv.DictReader(open('/tmp/eval_map.csv'))}
truth=[11,30,64,18,57];fp=[34,56,12,23]
items=[('TUMUL',i) for i in truth]+[('FP',i) for i in fp]
cols=len(items);cell=140;img=Image.new('RGB',(cols*cell,2*cell+24),(15,15,15));dr=ImageDraw.Draw(img)
try: ft=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial Bold.ttf',13)
except: ft=ImageFont.load_default()
dr.text((4,4),f"Grad-CAM ({os.path.basename(MODEL)}): SUS=stampa, JOS=unde se uită (roșu=atenție). Centru=movilă bun; margini=shortcut.",fill=(255,230,90),font=ft)
for k,(lbl,idx) in enumerate(items):
    lo,la=emap[idx];s=stamp(lo,la)
    if s is None: continue
    sc,cam=gradcam(s)
    base=Image.fromarray(s).convert('RGB')
    heat=Image.fromarray(np.stack([cam,np.zeros_like(cam),255-cam],-1).astype('uint8'))  # roșu=mare
    over=Image.blend(base,heat,0.55)
    x=k*cell;img.paste(base.resize((136,136)),(x+2,24));img.paste(over.resize((136,136)),(x+2,24+cell))
    col=(120,255,120) if lbl=='TUMUL' else (255,120,120)
    dr.text((x+3,24),f"{lbl}#{idx} {sc:.2f}",fill=col,font=ft)
img.save(f'{H}/review/gradcam.png');print(f"-> review/gradcam.png ({len(items)} stampe, model {os.path.basename(MODEL)})")
