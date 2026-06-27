#!/usr/bin/env python3
# score_ditches.py [T] — scorează negativele-șanț cu modelul PRE-șanț (combined_bigbatch.pt) ca să separe
# HARD (model păcălit, scor mare = valoros, ca samples albastre Andrei) de EASY (linii pe care modelul deja le respinge).
# Păstrează HARD în dataset_neg_ditch/, mută EASY în dataset_neg_ditch_easy/. Replicare exactă load()+Net din train.
import os,sys,glob,shutil
import numpy as np
from PIL import Image,ImageFilter
import torch,torch.nn as nn
H=os.path.dirname(os.path.dirname(os.path.abspath(__file__)));dev=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
T=float(sys.argv[1]) if len(sys.argv)>1 else None
def _histeq(a):
    h=np.bincount(a.ravel(),minlength=256).astype(np.float64);cdf=h.cumsum()
    if cdf[-1]==0: return a
    return (cdf[a]/cdf[-1]*255).astype(np.uint8)
def load(f):
    a=np.asarray(Image.open(f).convert('L').resize((128,128)),np.uint8)
    return _histeq(np.asarray(Image.fromarray(a).filter(ImageFilter.GaussianBlur(0.8)),np.uint8))
class Net(nn.Module):
    def __init__(s):
        super().__init__();s.c=nn.Sequential(nn.Conv2d(1,16,3,2,1),nn.ReLU(),nn.Conv2d(16,32,3,2,1),nn.ReLU(),nn.Conv2d(32,64,3,2,1),nn.ReLU(),nn.AdaptiveAvgPool2d(1));s.f=nn.Linear(64,1)
    def forward(s,x): return s.f(s.c(x).flatten(1)).squeeze(1)
net=Net().to(dev);net.load_state_dict(torch.load(f'{H}/combined_bigbatch.pt',map_location=dev,weights_only=True));net.eval()
fs=sorted(glob.glob(f'{H}/dataset_neg_ditch/*.png'))
scores=[]
with torch.no_grad():
    for i in range(0,len(fs),256):
        b=fs[i:i+256];xb=torch.tensor(np.array([load(f) for f in b],dtype=np.uint8)).unsqueeze(1).float().to(dev)/255.
        scores.extend(torch.sigmoid(net(xb)).cpu().numpy().tolist())
scores=np.array(scores)
print(f"{len(fs)} șanțuri scorate cu bigbatch. Distribuție scor (cât de mult păcălesc modelul):")
for q in (0.5,0.4,0.3,0.25,0.2,0.15,0.1,0.05):
    print(f"  scor>={q}: {int((scores>=q).sum())} ({100*(scores>=q).mean():.0f}%)")
print(f"  median {np.median(scores):.3f}  mean {scores.mean():.3f}  max {scores.max():.3f}")
if T is None:
    print("\n(dry-run; dă un prag T ca argument ca să muți EASY<T în dataset_neg_ditch_easy/)");raise SystemExit
easy=f'{H}/dataset_neg_ditch_easy';os.makedirs(easy,exist_ok=True)
mv=0
for f,sc in zip(fs,scores):
    if sc<T: shutil.move(f,os.path.join(easy,os.path.basename(f)));mv+=1
print(f"\nT={T}: mutat {mv} EASY -> dataset_neg_ditch_easy/ ; rămas HARD în dataset_neg_ditch/: {len(fs)-mv}")
