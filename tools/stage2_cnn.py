#!/usr/bin/env python3
# stage2_cnn.py train  NPZ MODEL_OUT [CHANNELS]   — antreneaza re-scorer stage-2 (CNN mic 2-canal) pe stampe taiate cu cut_2ch.py
# stage2_cnn.py apply  MODEL NPZ OUT_CSV          — scoreaza fiecare stampa -> lon,lat,p2 (prob tumul), 1=keep daca ok
# CHANNELS: 'both'(hillshade+SLRM, default) | 'slrm'(doar canal 1) | 'hs'(doar canal 0).
# Geo-split CV (0.1°) ca sa nu scurga vecini. Raporteaza CV AUC + curba recall_pos vs %neg taiate.
import os,sys,json,math
import numpy as np
import torch,torch.nn as nn
dev=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
H=os.path.expanduser('~/lidar-match')
def auc(score,y):
    pos=score[y==1];neg=score[y==0]
    if len(pos)==0 or len(neg)==0:return 0.5
    allv=np.concatenate([pos,neg]);_,inv,cnt=np.unique(allv,return_inverse=True,return_counts=True)
    csum=np.cumsum(cnt);start=csum-cnt;avg=(start+csum+1)/2.0;ranks=avg[inv]
    return (ranks[:len(pos)].sum()-len(pos)*(len(pos)+1)/2)/(len(pos)*len(neg))
def sel(X,ch):
    if ch=='slrm':return X[:,1:2]
    if ch=='hs':return X[:,0:1]
    return X
class Net(nn.Module):
    def __init__(s,nin):
        super().__init__();s.c=nn.Sequential(nn.Conv2d(nin,16,3,2,1),nn.ReLU(),nn.Conv2d(16,32,3,2,1),nn.ReLU(),nn.Conv2d(32,64,3,2,1),nn.ReLU(),nn.AdaptiveAvgPool2d(1));s.f=nn.Linear(64,1)
    def forward(s,x):return s.f(s.c(x).flatten(1)).squeeze(1)
def augment(X,y):
    # X (N,C,128,128) uint8 -> x8 (4 rot x 2 flip)
    outX=[];outY=[]
    for k in range(4):
        r=np.rot90(X,k,axes=(2,3))
        outX.append(r);outY.append(y);outX.append(r[:,:,:,::-1].copy());outY.append(y)
    return np.concatenate(outX),np.concatenate(outY)
def train_net(Xtr,ytr,nin,epochs=40,lr=1e-3):
    net=Net(nin).to(dev);opt=torch.optim.Adam(net.parameters(),lr=lr,weight_decay=1e-4)
    npos=max(1,(ytr==1).sum());nneg=max(1,(ytr==0).sum());pw=torch.tensor([nneg/npos],dtype=torch.float32,device=dev)
    lossf=nn.BCEWithLogitsLoss(pos_weight=pw)
    Xt=torch.tensor(Xtr,dtype=torch.float32).to(dev)/255.;yt=torch.tensor(ytr,dtype=torch.float32).to(dev)
    n=len(yt);torch.manual_seed(0)
    for ep in range(epochs):
        perm=torch.randperm(n)
        for i in range(0,n,64):
            idx=perm[i:i+64];opt.zero_grad();out=net(Xt[idx]);loss=lossf(out,yt[idx]);loss.backward();opt.step()
    return net
def score_net(net,X):
    net.eval();Xt=torch.tensor(X,dtype=torch.float32).to(dev)/255.;out=[]
    with torch.no_grad():
        for i in range(0,len(Xt),256):out.extend(torch.sigmoid(net(Xt[i:i+256])).cpu().numpy().tolist())
    return np.array(out)
def geo_folds(lon,lat,k=5):
    fold=[(round(a*10),round(b*10)) for a,b in zip(lon,lat)];uniq=sorted(set(fold));cmap={c:(hash(c)%k) for c in uniq}
    return np.array([cmap[c] for c in fold])
def train(npz,model_out,ch='both'):
    d=np.load(npz);X=d['X'];y=d['y'];lon=d['lon'];lat=d['lat'];ok=d['ok']
    m=ok&(y>=0);X=sel(X[m],ch);y=y[m];lon=lon[m];lat=lat[m];nin=X.shape[1]
    print(f"train: {len(y)} stampe ({nin}ch '{ch}') | pos {int((y==1).sum())} neg {int((y==0).sum())}")
    folds=geo_folds(lon,lat,5);oof=np.full(len(y),np.nan)
    for kf in range(5):
        tr=folds!=kf;te=folds==kf
        if te.sum()==0 or (y[tr]==1).sum()<3:continue
        Xa,ya=augment(X[tr],y[tr]);net=train_net(Xa,ya,nin);oof[te]=score_net(net,X[te])
    mm=~np.isnan(oof);cv=auc(oof[mm],y[mm]);print(f"geo-split CV AUC: {cv:.3f} ({mm.sum()} stampe)")
    pos=oof[mm&(y==1)];neg=oof[mm&(y==0)]
    print("  Prag | recall_pos | %neg(FP) taiate:")
    sugg={}
    for rec in (1.0,0.97,0.95,0.90):
        thr=float(np.quantile(pos,1-rec)) if rec<1.0 else float(pos.min())
        print(f"    {thr:.3f}  recall_pos {(pos>=thr).mean()*100:.0f}%  taie {(neg<thr).mean()*100:.0f}% FP")
        sugg[f'thr_rec{int(rec*100)}']=thr
    # final fit pe tot
    Xa,ya=augment(X,y);net=train_net(Xa,ya,nin)
    torch.save(net.state_dict(),model_out)
    json.dump(dict(ch=ch,nin=nin,cvauc=float(cv),**sugg),open(model_out+'.json','w'),indent=1)
    print(f"-> {model_out} (+ .json; thr_rec100={sugg['thr_rec100']:.3f}, thr_rec95={sugg['thr_rec95']:.3f})")
def apply(model,npz,out_csv,ch=None):
    meta=json.load(open(model+'.json'));ch=ch or meta['ch'];nin=meta['nin']
    d=np.load(npz);X=sel(d['X'],ch);ok=d['ok'];lon=d['lon'];lat=d['lat']
    net=Net(nin).to(dev);net.load_state_dict(torch.load(model,map_location=dev,weights_only=True))
    p=score_net(net,X)
    with open(out_csv,'w') as f:
        f.write('lon,lat,p2,ok\n')
        for lo,la,pp,o in zip(lon,lat,p,ok):f.write(f"{lo:.6f},{la:.6f},{pp:.4f},{int(bool(o))}\n")
    print(f"-> {out_csv} | {len(p)} scorate")
if __name__=='__main__':
    if sys.argv[1]=='train':train(sys.argv[2],sys.argv[3],sys.argv[4] if len(sys.argv)>4 else 'both')
    elif sys.argv[1]=='apply':apply(sys.argv[2],sys.argv[3],sys.argv[4])
