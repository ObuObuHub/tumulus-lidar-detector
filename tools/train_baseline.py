#!/usr/bin/env python3
# train_baseline.py — primul CNN baseline (underfit by design) detector movila/non-movila.
# Reguli: split pe GEOGRAFIE inainte de augmentare (anti-leakage), augmentare x16 DOAR train,
# CNN mic, ponderare de clasa (pozitivi rari). Eval pe pozitivi tinuti deoparte + negative val.
import os,glob,csv,math,random
import numpy as np
from PIL import Image
import torch,torch.nn as nn
random.seed(0); torch.manual_seed(0); np.random.seed(0)
H=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dev=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print("device:",dev)
# --- pozitivi: dataset_pos + coord/locality din labels.csv (LAKI3_DTM) pt split geografic ---
posfiles=sorted(glob.glob(f'{H}/dataset_pos/*.png'))
dtm_rows=[r for r in csv.DictReader(open(f'{H}/labeled/labels.csv')) if r['tile']=='LAKI3_DTM']
# grup geografic = rotunjire coord la ~2km
def grp(lon,lat): return (round(float(lon)*50)/50, round(float(lat)*50)/50)
pos=[]
for i,f in enumerate(posfiles):
    g=grp(dtm_rows[i]['lon'],dtm_rows[i]['lat']) if i<len(dtm_rows) else ('?',i)
    pos.append((f,g))
groups=sorted(set(g for _,g in pos))
random.shuffle(groups)
nval=max(1,len(groups)//4); val_g=set(groups[:nval])
postr=[f for f,g in pos if g not in val_g]; posva=[f for f,g in pos if g in val_g]
print(f"pozitivi: {len(pos)} in {len(groups)} grupuri geo -> train {len(postr)}, val {len(posva)} (split pe locatie, fara leakage)")
# --- negative ---
negfiles=sorted(glob.glob(f'{H}/dataset_neg/*.png')); random.shuffle(negfiles)
nneg_val=300; negva=negfiles[:nneg_val]; negtr=negfiles[nneg_val:]
print(f"negative: {len(negfiles)} -> train {len(negtr)}, val {len(negva)}")
def load(f): return np.asarray(Image.open(f).convert('L').resize((128,128)),dtype=np.float32)/255.0
def aug(a):
    out=[]
    for k in range(4):
        r=np.rot90(a,k)
        out.append(r); out.append(np.fliplr(r))
    return out  # x8 (rot4 x flip2); restul variatie din crop ar veni separat
# train tensors: augmentam pozitivii x8, negativii ii subesantionam ~ raport 1:3
Xtr=[];Ytr=[]
for f in postr:
    for a in aug(load(f)): Xtr.append(a);Ytr.append(1)
npos_aug=len(Xtr)
random.shuffle(negtr); negtr_use=negtr[:npos_aug*3]
for f in negtr_use: Xtr.append(load(f));Ytr.append(0)
print(f"train: {npos_aug} pozitivi augmentati + {len(negtr_use)} negativi = {len(Xtr)}")
Xva=[];Yva=[]
for f in posva: Xva.append(load(f));Yva.append(1)
for f in negva: Xva.append(load(f));Yva.append(0)
def tens(X,Y):
    x=torch.tensor(np.array(X)).unsqueeze(1).float(); y=torch.tensor(Y).float()
    return x,y
Xtr_t,Ytr_t=tens(Xtr,Ytr); Xva_t,Yva_t=tens(Xva,Yva)
class Net(nn.Module):
    def __init__(s):
        super().__init__()
        s.c=nn.Sequential(nn.Conv2d(1,16,3,2,1),nn.ReLU(),nn.Conv2d(16,32,3,2,1),nn.ReLU(),
                          nn.Conv2d(32,64,3,2,1),nn.ReLU(),nn.AdaptiveAvgPool2d(1))
        s.f=nn.Linear(64,1)
    def forward(s,x): return s.f(s.c(x).flatten(1)).squeeze(1)
net=Net().to(dev)
opt=torch.optim.Adam(net.parameters(),1e-3,weight_decay=1e-4)
posw=torch.tensor([3.0]).to(dev)  # pozitivii rari
lossf=nn.BCEWithLogitsLoss(pos_weight=posw)
Xtr_t,Ytr_t=Xtr_t.to(dev),Ytr_t.to(dev); Xva_t,Yva_t=Xva_t.to(dev),Yva_t.to(dev)
n=len(Xtr_t); bs=64
for ep in range(30):
    net.train(); perm=torch.randperm(n)
    for i in range(0,n,bs):
        idx=perm[i:i+bs]; opt.zero_grad()
        out=net(Xtr_t[idx]); l=lossf(out,Ytr_t[idx]); l.backward(); opt.step()
    if (ep+1)%5==0:
        net.eval()
        with torch.no_grad():
            pv=torch.sigmoid(net(Xva_t))
            pred=(pv>0.5).float()
            tp=((pred==1)&(Yva_t==1)).sum().item(); fp=((pred==1)&(Yva_t==0)).sum().item()
            fn=((pred==0)&(Yva_t==1)).sum().item(); tn=((pred==0)&(Yva_t==0)).sum().item()
            rec=tp/(tp+fn+1e-9); prec=tp/(tp+fp+1e-9)
            print(f"ep{ep+1}: val recall {rec:.2f} ({tp}/{tp+fn} movile), precizie {prec:.2f}, FP {fp}/{fp+tn} negative")
torch.save(net.state_dict(),f'{H}/baseline_cnn.pt')
print("model salvat: baseline_cnn.pt")
print("NOTA: putini pozitivi -> baseline underfit, valideaza pipeline-ul cap-coada. Volum real via broker Arad.")
