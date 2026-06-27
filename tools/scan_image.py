#!/usr/bin/env python3
# scan_image.py IMG MODEL WIN_PX [TH] — scanează un hillshade dat ca IMAGINE (nu tile geo) cu modelul.
# WIN_PX = mărimea ferestrei în px care corespunde la ~80m pe teren (= 80m / (m/px al imaginii)).
# Replică pipeline-ul de stampă: fereastră -> downsample la ~40px (2m efectiv) -> 128 -> homog -> scor. NMS -> overlay.
import sys,os,math
import numpy as np
from PIL import Image,ImageFilter,ImageDraw,ImageFont
import torch,torch.nn as nn
H=os.path.expanduser('~/lidar-match');dev=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
IMG=sys.argv[1];MODEL=sys.argv[2];WIN=int(sys.argv[3]);TH=float(sys.argv[4]) if len(sys.argv)>4 else 0.6
im=Image.open(IMG).convert('L');A=np.asarray(im,np.float32);Hh,Ww=A.shape
print(f"imagine {Ww}x{Hh}px | fereastra {WIN}px (~80m) | prag {TH}",flush=True)
def homog(a):
    a2=np.asarray(Image.fromarray(a).filter(ImageFilter.GaussianBlur(0.8)),np.uint8);cdf=np.bincount(a2.ravel(),minlength=256).astype(np.float64).cumsum()
    return (cdf[a2]/cdf[-1]*255).astype(np.uint8) if cdf[-1]>0 else a2
def stamp(cx,cy):
    h=WIN//2;w=A[cy-h:cy+h,cx-h:cx+h]
    if w.shape!=(2*h,2*h) or w.std()<3: return None
    lo,hi=np.percentile(w,2),np.percentile(w,98);a=np.clip((w-lo)/(hi-lo+1e-6),0,1)
    a40=np.asarray(Image.fromarray((a*255).astype('uint8')).resize((40,40)),np.float32)/255.  # 2m efectiv
    return homog(np.asarray(Image.fromarray((a40*255).astype('uint8')).resize((128,128)),np.uint8))
class Net(nn.Module):
    def __init__(s):
        super().__init__();s.c=nn.Sequential(nn.Conv2d(1,16,3,2,1),nn.ReLU(),nn.Conv2d(16,32,3,2,1),nn.ReLU(),nn.Conv2d(32,64,3,2,1),nn.ReLU(),nn.AdaptiveAvgPool2d(1));s.f=nn.Linear(64,1)
    def forward(s,x): return s.f(s.c(x).flatten(1)).squeeze(1)
net=Net().to(dev);net.load_state_dict(torch.load(MODEL,map_location=dev,weights_only=True));net.eval()
step=max(6,WIN//3);h=WIN//2;batch=[];pos=[]
for cy in range(h,Hh-h,step):
    for cx in range(h,Ww-h,step):
        s=stamp(cx,cy)
        if s is not None: batch.append(s);pos.append((cx,cy))
sc=[]
X=torch.tensor(np.array(batch,dtype=np.uint8))
with torch.no_grad():
    for k in range(0,len(X),512): sc.extend(torch.sigmoid(net(X[k:k+512].unsqueeze(1).float().to(dev)/255.)).cpu().numpy().tolist())
sc=np.array(sc);idx=np.argsort(-sc);kept=[]
for k in idx:
    if sc[k]<TH: break
    cx,cy=pos[k]
    if any((cx-q[0])**2+(cy-q[1])**2 < (WIN*0.8)**2 for q in kept): continue
    kept.append((cx,cy,float(sc[k])))
print(f"  {len(kept)} detecții (NMS) >= {TH}",flush=True)
ov=im.convert('RGB');dr=ImageDraw.Draw(ov)
try: ft=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial Bold.ttf',15)
except: ft=ImageFont.load_default()
for cx,cy,s in kept:
    r=WIN//2;col=(255,40,40) if s>=0.85 else (40,220,40) if s>=0.7 else (255,200,40)
    dr.ellipse([cx-r,cy-r,cx+r,cy+r],outline=col,width=3);dr.text((cx-r,cy-r-15),f"{s:.2f}",fill=col,font=ft)
out=f"{H}/review/scan_image_out.png";ov.save(out);print(f"  -> {out} (roșu>=.85 verde>=.7 galben>={TH})")
