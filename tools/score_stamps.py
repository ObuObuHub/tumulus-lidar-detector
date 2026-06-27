#!/usr/bin/env python3
# score_stamps.py DIR [MODEL] [GLOB] — scorează stampe dintr-un dir cu modelul (recipe load training:
# load_raw 128 + homog blur0.8+histeq) și raportează distribuția. Pt a verifica dacă negativele extrase
# sunt HARD (model aprinde pe ele = utile) sau EASY (model deja le dă scor mic = inutile pt precizie).
import os,sys,glob,math
import numpy as np
from PIL import Image,ImageFilter
import torch,torch.nn as nn
H=os.path.expanduser('~/lidar-match');dev=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
DIR=sys.argv[1];MODEL=sys.argv[2] if len(sys.argv)>2 else f'{H}/combined_cnn.pt';PAT=sys.argv[3] if len(sys.argv)>3 else '*.png'
def _histeq(a):
    h=np.bincount(a.ravel(),minlength=256).astype(np.float64);cdf=h.cumsum()
    return a if cdf[-1]==0 else (cdf[a]/cdf[-1]*255).astype(np.uint8)
def load(f):
    a=np.asarray(Image.open(f).convert('L').resize((128,128)),np.uint8)
    return _histeq(np.asarray(Image.fromarray(a).filter(ImageFilter.GaussianBlur(0.8)),np.uint8))
class Net(nn.Module):
    def __init__(s):
        super().__init__();s.c=nn.Sequential(nn.Conv2d(1,16,3,2,1),nn.ReLU(),nn.Conv2d(16,32,3,2,1),nn.ReLU(),nn.Conv2d(32,64,3,2,1),nn.ReLU(),nn.AdaptiveAvgPool2d(1));s.f=nn.Linear(64,1)
    def forward(s,x): return s.f(s.c(x).flatten(1)).squeeze(1)
net=Net().to(dev);net.load_state_dict(torch.load(MODEL,map_location=dev,weights_only=True));net.eval()
files=sorted(glob.glob(f'{DIR}/{PAT}'))
if not files: print("0 fisiere");sys.exit()
sc=[]
for i in range(0,len(files),512):
    b=files[i:i+512];xb=torch.tensor(np.array([load(f) for f in b],dtype=np.uint8)).unsqueeze(1).float().to(dev)/255.
    with torch.no_grad(): sc+=torch.sigmoid(net(xb)).cpu().numpy().tolist()
sc=np.array(sc)
print(f"{DIR}  model {os.path.basename(MODEL)}  n={len(sc)}")
print(f"  scor neg: median {np.median(sc):.3f} mean {sc.mean():.3f} | %>=0.3 {100*(sc>=0.3).mean():.1f}% %>=0.5 {100*(sc>=0.5).mean():.1f}% %>=0.7 {100*(sc>=0.7).mean():.1f}% %>=0.9 {100*(sc>=0.9).mean():.1f}%")
# per-clasa daca numele are prefix
import collections
byc=collections.defaultdict(list)
for f,s in zip(files,sc): byc[os.path.basename(f).split('_')[0]].append(s)
for c in sorted(byc):
    v=np.array(byc[c]);print(f"  [{c}] n={len(v)} median {np.median(v):.3f} %>=0.5 {100*(v>=0.5).mean():.1f}% %>=0.7 {100*(v>=0.7).mean():.1f}%")
