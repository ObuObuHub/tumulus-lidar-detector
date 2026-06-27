#!/usr/bin/env python3
# test_image.py MODEL IMG [THRESH] — rulează modelul ca slider MULTI-SCARĂ pe o imagine hillshade arbitrară
# (scară necunoscută). Testează generalizarea: recunoaște tumuli la VREO scară? Suprapune detecțiile.
import os,sys,math
import numpy as np
from PIL import Image,ImageFilter,ImageDraw
import torch,torch.nn as nn
H=os.path.dirname(os.path.dirname(os.path.abspath(__file__)));dev=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
MODEL=sys.argv[1];IMG=sys.argv[2];TH=float(sys.argv[3]) if len(sys.argv)>3 else 0.9
def homog(a):
    a2=np.asarray(Image.fromarray(a).filter(ImageFilter.GaussianBlur(0.8)),np.uint8);cdf=np.bincount(a2.ravel(),minlength=256).astype(np.float64).cumsum()
    return (cdf[a2]/cdf[-1]*255).astype(np.uint8) if cdf[-1]>0 else a2
class Net(nn.Module):
    def __init__(s):
        super().__init__();s.c=nn.Sequential(nn.Conv2d(1,16,3,2,1),nn.ReLU(),nn.Conv2d(16,32,3,2,1),nn.ReLU(),nn.Conv2d(32,64,3,2,1),nn.ReLU(),nn.AdaptiveAvgPool2d(1));s.f=nn.Linear(64,1)
    def forward(s,x): return s.f(s.c(x).flatten(1)).squeeze(1)
net=Net().to(dev);net.load_state_dict(torch.load(MODEL,map_location=dev,weights_only=True));net.eval()
img=Image.open(IMG).convert('L');W,Hh=img.size;arr=np.asarray(img,np.uint8)
print(f"imagine {W}x{Hh}, model {os.path.basename(MODEL)}")
# fereastra modelului = 80m. Nu știm m/px -> încerc mai multe mărimi de fereastră (în px) = scări diferite.
dets=[]  # (cx,cy,win,score)
for win in [48,64,90,128,180,256]:
    if win>min(W,Hh): continue
    step=max(8,win//3);batch=[];pos=[]
    for y in range(0,Hh-win+1,step):
        for x in range(0,W-win+1,step):
            c=arr[y:y+win,x:x+win]
            c128=np.asarray(Image.fromarray(c).resize((128,128)),np.uint8)
            batch.append(homog(c128));pos.append((x+win//2,y+win//2,win))
    if not batch: continue
    X=torch.tensor(np.array(batch,dtype=np.uint8))
    sc=[]
    with torch.no_grad():
        for i in range(0,len(X),512): sc.extend(torch.sigmoid(net(X[i:i+512].unsqueeze(1).float().to(dev)/255.)).cpu().numpy().tolist())
    sc=np.array(sc);hi=sc>=TH
    print(f"  win {win}px: {int(hi.sum())}/{len(sc)} >= {TH} (max {sc.max():.3f})")
    for (cx,cy,w),s in zip(pos,sc):
        if s>=TH: dets.append((cx,cy,w,float(s)))
# non-max suppression simplu (păstrează scor mare, suprimă vecini <win/2)
dets.sort(key=lambda d:-d[3]);keep=[]
for d in dets:
    if all((d[0]-k[0])**2+(d[1]-k[1])**2 > (max(d[2],k[2])*0.5)**2 for k in keep): keep.append(d)
print(f"  -> {len(keep)} detecții (după NMS)")
ov=img.convert('RGB');dr=ImageDraw.Draw(ov)
for cx,cy,w,s in keep:
    r=w//2;col=(255,40,40) if s>=0.95 else (255,180,40)
    dr.ellipse([cx-r,cy-r,cx+r,cy+r],outline=col,width=3)
    dr.text((cx-r,cy-r-12),f"{s:.2f}",fill=col)
out=f"{H}/review/test_image_{os.path.basename(MODEL).replace('.pt','')}.png";ov.save(out)
print(f"  -> {out}")
