#!/usr/bin/env python3
# scan_image_gradcam.py IMG MODEL WIN [TH] — scanează imagine + Grad-CAM la FIECARE detecție.
# Output: review/scan_gc_overlay.png (detecții pe imagine) + review/scan_gc_montaj.png (per detecție: stampă|Grad-CAM).
# Grad-CAM = unde se uită modelul: centru pe dom = real; difuz/margine = scurtătură. Ca să vezi că nu minte.
import sys,os,math
import numpy as np
from PIL import Image,ImageFilter,ImageDraw,ImageFont
import torch,torch.nn as nn,torch.nn.functional as Fn
H=os.path.expanduser('~/lidar-match');dev=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
IMG=sys.argv[1];MODEL=sys.argv[2];WIN=int(sys.argv[3]);TH=float(sys.argv[4]) if len(sys.argv)>4 else 0.85
im=Image.open(IMG).convert('L');A=np.asarray(im,np.float32);Hh,Ww=A.shape
print(f"imagine {Ww}x{Hh} | WIN {WIN} | TH {TH}",flush=True)
def homog(a):
    a2=np.asarray(Image.fromarray(a).filter(ImageFilter.GaussianBlur(0.8)),np.uint8);cdf=np.bincount(a2.ravel(),minlength=256).astype(np.float64).cumsum()
    return (cdf[a2]/cdf[-1]*255).astype(np.uint8) if cdf[-1]>0 else a2
def stamp(cx,cy):
    h=WIN//2;w=A[cy-h:cy+h,cx-h:cx+h]
    if w.shape!=(2*h,2*h) or w.std()<3: return None
    lo,hi=np.percentile(w,2),np.percentile(w,98);a=np.clip((w-lo)/(hi-lo+1e-6),0,1)
    a40=np.asarray(Image.fromarray((a*255).astype('uint8')).resize((40,40)),np.uint8)
    return homog(np.asarray(Image.fromarray(a40).resize((128,128)),np.uint8))
class Net(nn.Module):
    def __init__(s):
        super().__init__();s.c=nn.Sequential(nn.Conv2d(1,16,3,2,1),nn.ReLU(),nn.Conv2d(16,32,3,2,1),nn.ReLU(),nn.Conv2d(32,64,3,2,1),nn.ReLU(),nn.AdaptiveAvgPool2d(1));s.f=nn.Linear(64,1)
    def forward(s,x): return s.f(s.c(x).flatten(1)).squeeze(1)
net=Net().to(dev);net.load_state_dict(torch.load(MODEL,map_location=dev,weights_only=True));net.eval()
acts={};grads={}
net.c[4].register_forward_hook(lambda m,i,o: acts.__setitem__('v',o.detach()))
net.c[4].register_full_backward_hook(lambda m,gi,go: grads.__setitem__('v',go[0].detach()))
def gradcam(s128):
    x=torch.tensor(s128).unsqueeze(0).unsqueeze(0).float().to(dev)/255.;x.requires_grad_(True)
    net.zero_grad();out=net(x);out.backward()
    Aa=acts['v'][0];G=grads['v'][0];wts=G.mean(dim=(1,2));cam=Fn.relu((wts[:,None,None]*Aa).sum(0))
    cam=cam.cpu().numpy();cam=cam/(cam.max()+1e-9)
    return float(torch.sigmoid(out).item()),np.asarray(Image.fromarray((cam*255).astype('uint8')).resize((128,128)))
# scan
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
# overlay numerotat
ov=im.convert('RGB');dr=ImageDraw.Draw(ov)
try: ft=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial Bold.ttf',22)
except: ft=ImageFont.load_default()
for i,(cx,cy,s) in enumerate(kept,1):
    r=WIN//2;col=(255,40,40) if s>=0.95 else (40,220,40)
    dr.ellipse([cx-r,cy-r,cx+r,cy+r],outline=col,width=4);dr.text((cx-r,cy-r-24),f"{i}:{s:.2f}",fill=col,font=ft)
ov.save(f"{H}/review/scan_gc_overlay.png");print(f"  -> review/scan_gc_overlay.png")
# montaj gradcam top-24
top=kept[:24];C=6;rw=math.ceil(len(top)/C);cell=120;hh=26
mont=Image.new('RGB',(C*cell,hh+rw*(2*cell+18)),(15,15,15));d2=ImageDraw.Draw(mont)
try: ft2=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial Bold.ttf',13)
except: ft2=ImageFont.load_default()
d2.text((4,5),f"Grad-CAM la detectie ({os.path.basename(MODEL)}): SUS=hillshade, JOS=atentie(rosu). Centru pe dom=real.",fill=(255,230,90),font=ft2)
for i,(cx,cy,s) in enumerate(top):
    st=stamp(cx,cy);_,cam=gradcam(st)
    base=Image.fromarray(st).convert('RGB');heat=Image.fromarray(np.stack([cam,np.zeros_like(cam),255-cam],-1).astype('uint8'))
    over=Image.blend(base,heat,0.55)
    gx=(i%C)*cell;gy=hh+(i//C)*(2*cell+18)
    mont.paste(base.resize((cell-4,cell-4)),(gx+2,gy+16));mont.paste(over.resize((cell-4,cell-4)),(gx+2,gy+16+cell))
    d2.text((gx+3,gy+2),f"#{i+1} {s:.2f}",fill=(120,255,120) if s>=0.95 else (255,210,90),font=ft2)
mont.save(f"{H}/review/scan_gc_montaj.png");print(f"  -> review/scan_gc_montaj.png ({len(top)})")
