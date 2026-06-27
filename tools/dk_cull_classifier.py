#!/usr/bin/env python3
# dk_cull_classifier.py [--apply SCORE_OUT] — antrenează CNN mic pe marcajele Andrei (dk_cull_labels.csv)
# „contaminat (șanț mare lângă movilă)" vs „curat". Cross-valid 20% held-out. Cu --apply: scorează TOȚI 21k.
import os,sys,csv,glob,random
import numpy as np
from PIL import Image,ImageFilter
import torch,torch.nn as nn
random.seed(0);torch.manual_seed(0);np.random.seed(0)
H=os.path.expanduser('~/lidar-match');dev=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
def _histeq(a):
    h=np.bincount(a.ravel(),minlength=256).astype(np.float64);cdf=h.cumsum()
    return (cdf[a]/cdf[-1]*255).astype(np.uint8) if cdf[-1]>0 else a
def load(p):
    a=np.asarray(Image.open(p).convert('L').resize((128,128)).filter(ImageFilter.GaussianBlur(0.8)),np.uint8)
    return _histeq(a)
def find(png):
    for d in ('dataset_pos_dk','dataset_pos_dk_culled'):
        p=f'{H}/{d}/{png}'
        if os.path.exists(p): return p
    return None
# etichete dedup (ultima valoare)
lab={}
for r in csv.reader(open(f'{H}/labeled/dk_cull_labels.csv')):
    if len(r)>=2: lab[r[0]]=int(r[1])
items=[(find(p),y) for p,y in lab.items() if find(p)]
random.shuffle(items)
n=len(items);nv=n//5
val=items[:nv];tr=items[nv:]
print(f"{n} etichete ({sum(y for _,y in items)} contaminate / {n-sum(y for _,y in items)} curate); train {len(tr)} / val {len(val)}")
def aug(a): return [np.rot90(a,k) for k in range(4)]+[np.fliplr(np.rot90(a,k)) for k in range(4)]
Xtr=[];Ytr=[]
for p,y in tr:
    for v in aug(load(p)): Xtr.append(v);Ytr.append(y)
Xtr=torch.tensor(np.array(Xtr,dtype=np.uint8));Ytr=torch.tensor(Ytr,dtype=torch.float32).to(dev)
Xva=torch.tensor(np.array([load(p) for p,_ in val],dtype=np.uint8)).to(dev);Yva=np.array([y for _,y in val])
class Net(nn.Module):
    def __init__(s):
        super().__init__();s.c=nn.Sequential(nn.Conv2d(1,16,3,2,1),nn.ReLU(),nn.Conv2d(16,32,3,2,1),nn.ReLU(),nn.Conv2d(32,64,3,2,1),nn.ReLU(),nn.AdaptiveAvgPool2d(1));s.f=nn.Linear(64,1);s.d=nn.Dropout(0.3)
    def forward(s,x): return s.f(s.d(s.c(x).flatten(1))).squeeze(1)
net=Net().to(dev);opt=torch.optim.Adam(net.parameters(),1e-3,weight_decay=1e-3)
posw=torch.tensor([(Ytr==0).sum()/max(1,(Ytr==1).sum().item())]).to(dev)
lf=nn.BCEWithLogitsLoss(pos_weight=posw);bs=64;N=len(Xtr)
for e in range(60):
    net.train();pm=torch.randperm(N)
    for i in range(0,N,bs):
        idx=pm[i:i+bs];xb=Xtr[idx].unsqueeze(1).float().to(dev)/255.
        opt.zero_grad();lf(net(xb),Ytr[idx]).backward();opt.step()
net.eval()
with torch.no_grad(): pv=torch.sigmoid(net(Xva.unsqueeze(1).float()/255.)).cpu().numpy()
for T in (0.5,0.6,0.7):
    pred=(pv>=T).astype(int)
    tp=int(((pred==1)&(Yva==1)).sum());fp=int(((pred==1)&(Yva==0)).sum());fn=int(((pred==0)&(Yva==1)).sum())
    prec=tp/(tp+fp+1e-9);rec=tp/(tp+fn+1e-9)
    print(f"  VAL @{T}: precizie {100*prec:.0f}% recall {100*rec:.0f}% (tp{tp} fp{fp} fn{fn})")
if '--apply' in sys.argv:
    fs=sorted(glob.glob(f'{H}/dataset_pos_dk/*.png'))
    out=sys.argv[sys.argv.index('--apply')+1]
    sc=[]
    with torch.no_grad():
        for i in range(0,len(fs),512):
            b=fs[i:i+512];xb=torch.tensor(np.array([load(f) for f in b],dtype=np.uint8)).unsqueeze(1).float().to(dev)/255.
            sc.extend(torch.sigmoid(net(xb)).cpu().numpy().tolist())
    with open(out,'w',newline='') as f:
        w=csv.writer(f);w.writerow(['file','score'])
        for fn,s in sorted(zip(fs,sc),key=lambda x:-x[1]): w.writerow([os.path.basename(fn),f"{s:.3f}"])
    import numpy as _np;sc=_np.array(sc)
    print(f"  aplicat pe {len(fs)} -> {out}; scor>=0.7: {int((sc>=0.7).sum())}, >=0.5: {int((sc>=0.5).sum())}")
