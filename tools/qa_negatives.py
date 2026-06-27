#!/usr/bin/env python3
# qa_negatives.py — QA pe negative: pt fiecare folder, scoreaza toate cu modelul si scoate 2 montaje:
# (1) RANDOM (reprezentativ), (2) TOP scor = cele mai "movila-like" (cele suspecte de movila scapata).
# Daca nici cele mai sus-scorate nu-s movile -> setul e curat.
import os,glob,random
import numpy as np
from PIL import Image,ImageDraw
import torch,torch.nn as nn
H=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); dev=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
class Net(nn.Module):
    def __init__(s):
        super().__init__();s.c=nn.Sequential(nn.Conv2d(1,16,3,2,1),nn.ReLU(),nn.Conv2d(16,32,3,2,1),nn.ReLU(),nn.Conv2d(32,64,3,2,1),nn.ReLU(),nn.AdaptiveAvgPool2d(1));s.f=nn.Linear(64,1)
    def forward(s,x): return s.f(s.c(x).flatten(1)).squeeze(1)
net=Net().to(dev);net.load_state_dict(torch.load(f'{H}/combined_cnn.pt',weights_only=True));net.eval()
def score_all(files):
    out=[]
    for i in range(0,len(files),256):
        b=files[i:i+256]
        xb=torch.tensor(np.array([np.asarray(Image.open(f).convert('L').resize((128,128)),np.float32)/255. for f in b])).unsqueeze(1).float().to(dev)
        with torch.no_grad(): s=torch.sigmoid(net(xb)).cpu().numpy()
        out+=list(zip(s.tolist(),b))
    return out
def montage(files,title,out,n=48):
    ff=files[:n];cols=8;rows=(len(ff)+cols-1)//cols
    M=Image.new('RGB',(cols*132+4,rows*132+24),(25,25,25));d=ImageDraw.Draw(M);d.text((6,6),title,fill=(200,200,120))
    for i,f in enumerate(ff): M.paste(Image.open(f).convert('RGB').resize((128,128)),((i%cols)*132+2,(i//cols)*132+20))
    M.save(out);return out
for label,folder in [("DEAL","dataset_neg_ro_hill"),("CAMPIE","dataset_neg_ro_plain"),("ANTROPIC","dataset_neg_anthro")]:
    fs=glob.glob(f'{H}/{folder}/*.png')
    if not fs: print(f"{label}: GOL"); continue
    sc=score_all(fs); sc.sort(reverse=True)
    arr=np.array([s for s,_ in sc])
    print(f"{label} ({len(fs)}): scor model mediu {arr.mean():.3f} | >0.5: {int((arr>0.5).sum())} ({100*(arr>0.5).mean():.1f}%) | >0.7: {int((arr>0.7).sum())} | max {arr.max():.3f}")
    random.seed(5); rnd=[f for _,f in sc]; random.shuffle(rnd)
    montage(rnd,f"{label} RANDOM (reprezentativ, {len(fs)})",f'{H}/review/qa_{folder}_random.png')
    montage([f for _,f in sc],f"{label} TOP-SCOR (cele mai movila-like = suspecte)",f'{H}/review/qa_{folder}_top.png')
    print(f"  -> review/qa_{folder}_random.png + qa_{folder}_top.png")
