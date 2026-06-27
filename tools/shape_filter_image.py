#!/usr/bin/env python3
# shape_filter_image.py IMG MODEL WIN [TH] — scanează imagine + FILTRU DE FORMĂ pe detecții.
# Idee: movila = bump IZOLAT (relief în centru, împrejurimi PLATE); FP pe teren erodat = împrejurimi tot rugoase.
# Metrica: std central (disk) vs std inel (annulus). Izolare = centru rugos & inel plat. + raport axe (alungit=taie).
# Test ONEST: raportează câte FP taie ȘI dacă cele 7 GT supraviețuiesc. Overlay înainte/după.
import sys,os,math
import numpy as np
from PIL import Image,ImageFilter,ImageDraw,ImageFont
import torch,torch.nn as nn
H=os.path.expanduser('~/lidar-match');dev=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
IMG=sys.argv[1];MODEL=sys.argv[2];WIN=int(sys.argv[3]);TH=float(sys.argv[4]) if len(sys.argv)>4 else 0.9
GT=[(105,165),(425,605),(590,830),(672,920),(755,1010),(812,1095),(868,1165)]  # cei 7 marcați de Andrei
im=Image.open(IMG).convert('L');A=np.asarray(im,np.float32);Hh,Ww=A.shape
def homog(a):
    a2=np.asarray(Image.fromarray(a).filter(ImageFilter.GaussianBlur(0.8)),np.uint8);cdf=np.bincount(a2.ravel(),minlength=256).astype(np.float64).cumsum()
    return (cdf[a2]/cdf[-1]*255).astype(np.uint8) if cdf[-1]>0 else a2
def stamp(cx,cy):
    h=WIN//2;w=A[cy-h:cy+h,cx-h:cx+h]
    if w.shape!=(2*h,2*h) or w.std()<2: return None
    lo,hi=np.percentile(w,2),np.percentile(w,98);a=np.clip((w-lo)/(hi-lo+1e-6),0,1)
    a40=np.asarray(Image.fromarray((a*255).astype('uint8')).resize((40,40)),np.uint8)
    return homog(np.asarray(Image.fromarray(a40).resize((128,128)),np.uint8))
# === FILTRU DE FORMĂ pe imaginea brută (relief = high-pass) ===
BL=np.asarray(Image.fromarray(A.astype('uint8')).filter(ImageFilter.GaussianBlur(8)),np.float32)
REL=np.abs(A-BL)  # relief local (textură/pantă); plat=mic, rugos=mare
yy,xx=np.mgrid[-60:61,-60:61];rr=np.hypot(xx,yy)
DISK=rr<=WIN*0.55; ANN=(rr>=WIN*0.9)&(rr<=WIN*1.7)
def shape_ok(cx,cy):
    if cy-60<0 or cy+61>Hh or cx-60<0 or cx+61>Ww: return False,0,0
    patch=REL[cy-60:cy+61,cx-60:cx+61]
    c=patch[DISK].mean(); a=patch[ANN].mean()
    iso=c-a                       # >0 = centru mai rugos decât inelul = bump izolat
    # raport axe al blob-ului central (alungit=ravenă)
    cm=patch*DISK; ys,xs=np.nonzero(cm>cm.mean()+cm.std())
    ratio=1.0
    if len(xs)>=12:
        mx,my=xs.mean(),ys.mean();mxx=((xs-mx)**2).mean();myy=((ys-my)**2).mean();mxy=((xs-mx)*(ys-my)).mean()
        tr=mxx+myy;dd=tr*tr/4-(mxx*myy-mxy*mxy);s=math.sqrt(max(0,dd));l1=tr/2+s;l2=tr/2-s;ratio=math.sqrt(l1/max(l2,1e-6))
    ok = (iso > ISO_T) and (a < ANN_T) and (ratio < RATIO_T)
    return ok,iso,a
ISO_T=float(os.environ.get('ISO_T',1.0)); ANN_T=float(os.environ.get('ANN_T',8.0)); RATIO_T=float(os.environ.get('RATIO_T',2.6))
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
X=torch.tensor(np.array(batch,dtype=np.uint8));sc=[]
with torch.no_grad():
    for k in range(0,len(X),1024): sc.extend(torch.sigmoid(net(X[k:k+1024].unsqueeze(1).float().to(dev)/255.)).cpu().numpy().tolist())
sc=np.array(sc);idx=np.argsort(-sc);kept=[]
for k in idx:
    if sc[k]<TH: break
    cx,cy=pos[k]
    if any((cx-q[0])**2+(cy-q[1])**2<(WIN*0.8)**2 for q in kept): continue
    kept.append((cx,cy,float(sc[k])))
filt=[(cx,cy,s) for cx,cy,s in kept if shape_ok(cx,cy)[0]]
def gt_hit(dets):
    hit=0
    for gx,gy in GT:
        if any((gx-cx)**2+(gy-cy)**2<(WIN*1.0)**2 for cx,cy,_ in dets): hit+=1
    return hit
print(f"ÎNAINTE filtru: {len(kept)} detecții | GT prinși: {gt_hit(kept)}/7")
print(f"DUPĂ filtru formă (ISO_T={ISO_T} ANN_T={ANN_T} RATIO_T={RATIO_T}): {len(filt)} detecții | GT prinși: {gt_hit(filt)}/7")
print(f"  -> FP tăiate: {len(kept)-len(filt)} ({100*(len(kept)-len(filt))/max(1,len(kept)):.0f}%)")
# overlay dupa
ov=im.convert('RGB');dr=ImageDraw.Draw(ov)
for cx,cy,s in filt:
    r=WIN//2;dr.ellipse([cx-r,cy-r,cx+r,cy+r],outline=(40,220,40),width=3)
for gx,gy in GT:
    dr.line([(gx-12,gy-12),(gx+12,gy+12)],fill=(255,255,0),width=3);dr.line([(gx-12,gy+12),(gx+12,gy-12)],fill=(255,255,0),width=3)
ov.save(f"{H}/review/shape_filter_after.png");print(f"  -> review/shape_filter_after.png (verde=detecții după filtru, X galben=GT Andrei)")
