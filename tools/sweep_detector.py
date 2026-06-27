#!/usr/bin/env python3
# sweep_detector.py CLON CLAT KM — faza DETECTOR: mozaic MDH al unei zone -> fereastra glisanta
# a modelului (80m@2m) in memorie -> harta de caldura a scorurilor -> hot-spots (campuri de movile).
import os,sys,math,subprocess
import numpy as np
from PIL import Image,ImageFilter
import torch,torch.nn as nn
H=os.path.dirname(os.path.dirname(os.path.abspath(__file__)));dev=torch.device('mps')
CLON=float(sys.argv[1]);CLAT=float(sys.argv[2]);KM=float(sys.argv[3]) if len(sys.argv)>3 else 5.0
R=6378137.0;C=2*math.pi*R;ORIG=-20037508.342787;ORIGY=20037508.342787;Z=17;MPP=C/(256*2**Z)
ORG="Q2Kmg0bQDn3rySgn";TM="/tmp/mdh_tiles"
MDHs=[("AR_MDH_tif",(20.67,45.86,22.77,46.70)),("BH_MDH_tif",(21.37,46.36,22.83,47.61)),("HD_MDH_tif",(22.32,45.23,23.60,46.37)),("AB_MDH_tif",(22.66,45.44,23.82,46.59))]
def pickm(lo,la):
    for s,(a,b,c,d) in MDHs:
        if a<=lo<=c and b<=la<=d: return s
    return None
svc=pickm(CLON,CLAT); print("serviciu:",svc)
def tile(col,row):
    fn=f"{TM}/{svc}_{col}_{row}.png"
    if not os.path.exists(fn): subprocess.run(["curl","-s","--max-time","20","-o",fn,f"https://tiles.arcgis.com/tiles/{ORG}/arcgis/rest/services/{svc}/MapServer/tile/{Z}/{row}/{col}"],check=False)
    try: return Image.open(fn).convert('L')
    except: return None
# mozaic box
halfm=KM*1000/2;half=halfm/MPP
x=R*math.radians(CLON);y=R*math.log(math.tan(math.pi/4+math.radians(CLAT)/2));px=(x-ORIG)/MPP;py=(ORIGY-y)/MPP
x0=px-half;y0=py-half;W=int(2*half)
cv=Image.new('L',(W,W),0);nt=0
for col in range(int(x0//256),int((x0+W)//256)+1):
    for row in range(int(y0//256),int((y0+W)//256)+1):
        t=tile(col,row)
        if t: cv.paste(t,(col*256-int(x0),row*256-int(y0)));nt+=1
print(f"mozaic {W}x{W}px ({KM}km), {nt} tile-uri")
mos=np.asarray(cv,np.float32)
# model
class Net(nn.Module):
    def __init__(s):
        super().__init__();s.c=nn.Sequential(nn.Conv2d(1,16,3,2,1),nn.ReLU(),nn.Conv2d(16,32,3,2,1),nn.ReLU(),nn.Conv2d(32,64,3,2,1),nn.ReLU(),nn.AdaptiveAvgPool2d(1));s.f=nn.Linear(64,1)
    def forward(s,x): return s.f(s.c(x).flatten(1)).squeeze(1)
net=Net().to(dev);net.load_state_dict(torch.load(f'{H}/combined_cnn.pt',weights_only=True));net.eval()
# fereastra glisanta: 80m = 80/MPP px nativ; stride ~40m
wpx=int(80/MPP);stride=int(40/MPP)
ys=list(range(0,W-wpx,stride));xs=list(range(0,W-wpx,stride))
print(f"fereastra {wpx}px, {len(ys)}x{len(xs)} pozitii")
heat=np.zeros((len(ys),len(xs)),np.float32)
batch=[];idxs=[]
def flush():
    global batch,idxs
    if not batch: return
    xb=torch.tensor(np.array(batch)).unsqueeze(1).float().to(dev)
    with torch.no_grad(): s=torch.sigmoid(net(xb)).cpu().numpy()
    for (iy,ix),sc in zip(idxs,s): heat[iy,ix]=sc
    batch=[];idxs=[]
for iy,yy in enumerate(ys):
    for ix,xx in enumerate(xs):
        w=mos[yy:yy+wpx,xx:xx+wpx]
        if w.std()<0.3: continue  # gol
        # homogenizare 2m + autocontrast + resize 128 (ca la stampe)
        f=max(1,int(round(2.0/MPP)));d=w[:w.shape[0]//f*f,:w.shape[1]//f*f].reshape(w.shape[0]//f,f,w.shape[1]//f,f).mean((1,3))
        lo,hi=np.percentile(d,2),np.percentile(d,98);d=np.clip((d-lo)/(hi-lo+1e-6),0,1)
        im=np.asarray(Image.fromarray((d*255).astype('uint8')).resize((128,128)),np.float32)/255.
        batch.append(im);idxs.append((iy,ix))
        if len(batch)>=512: flush()
flush()
# hot-spots: zone cu densitate mare de scor mare
hot=(heat>0.6).astype(float)
dens=np.asarray(Image.fromarray((hot*255).astype('uint8')).filter(ImageFilter.GaussianBlur(3)),float)
print(f"celule scor>0.6: {int(hot.sum())}/{hot.size} ({100*hot.mean():.1f}%)")
# top 5 hot-spots (maxime locale ale densitatii)
flat=[(dens[iy,ix],iy,ix) for iy in range(dens.shape[0]) for ix in range(dens.shape[1])]
flat.sort(reverse=True)
seen=[];print("TOP hot-spots (densitate movile):")
for d,iy,ix in flat:
    if d<30: break
    yy=ys[iy]+wpx//2;xx=xs[ix]+wpx//2
    # px mozaic -> lon/lat
    gx=x0+xx;gy=y0+yy
    lon=(ORIG+gx*MPP)/R*180/math.pi;lat=(2*math.atan(math.exp((ORIGY-gy*MPP)/R))-math.pi/2)*180/math.pi
    if any((lon-s[0])**2+(lat-s[1])**2<(0.006)**2 for s in seen): continue
    seen.append((lon,lat,float(d)));print(f"  dens={d:.0f} @ {lon:.4f},{lat:.4f}")
    if len(seen)>=5: break
# salveaza heatmap vizual
hm=Image.fromarray((np.clip(heat,0,1)*255).astype('uint8')).resize((600,600))
mosv=Image.fromarray(np.clip((mos-np.percentile(mos,2))/(np.percentile(mos,98)-np.percentile(mos,2)+1e-6)*255,0,255).astype('uint8')).resize((600,600))
import json;json.dump(seen,open('/tmp/hotspots.json','w'))
Image.blend(mosv.convert('RGB'),hm.convert('RGB').resize((600,600)),0.45).save(f'{H}/review/sweep_heat.png')
print("heatmap -> review/sweep_heat.png ; hotspots -> /tmp/hotspots.json")
