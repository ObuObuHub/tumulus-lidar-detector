#!/usr/bin/env python3
# train_2ch.py — VARIANTĂ 2-CANALE a train_combined.py: canal0=hillshade omogenizat, canal1=CURBURĂ-proxy
# (Laplacian al hillshade-ului, prinde convex-dom vs liniar). Test ieftin al ipotezei multi-bandă FĂRĂ re-fetch
# elevație (curbura adevărată din elevație = pas 2 dacă ăsta promite). Salvează combined_2ch.pt (NU atinge cel bun).
import os,glob,csv,math,random
import numpy as np
from PIL import Image,ImageDraw,ImageFont,ImageFilter
import torch,torch.nn as nn
random.seed(0); torch.manual_seed(0); np.random.seed(0)
H=os.path.expanduser('~/lidar-match'); dev=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
def _rankavg(a):
    a=np.asarray(a,float); s=np.argsort(a,kind='mergesort'); aS=a[s]; n=len(a); rs=np.arange(1,n+1,dtype=float); i=0
    while i<n:
        j=i
        while j+1<n and aS[j+1]==aS[i]: j+=1
        rs[i:j+1]=(i+j+2)/2.0; i=j+1
    r=np.empty(n); r[s]=rs; return r
def auroc(scores,labels):
    y=np.asarray(labels,float); npos=y.sum(); nneg=len(y)-npos
    if npos==0 or nneg==0: return float('nan')
    r=_rankavg(scores); return float((r[y==1].sum()-npos*(npos+1)/2)/(npos*nneg))
def auprc(scores,labels):
    y=np.asarray(labels,float); o=np.argsort(-np.asarray(scores,float),kind='mergesort'); y=y[o]
    tp=np.cumsum(y); fp=np.cumsum(1-y); P=tp/(tp+fp); R=tp/(y.sum() if y.sum()>0 else 1.0)
    Rp=np.concatenate(([0.],R[:-1])); return float(np.sum((R-Rp)*P))
MDH=[(20.67,45.86,22.77,46.70),(21.37,46.36,22.83,47.61),(22.32,45.23,23.60,46.37),(22.66,45.44,23.82,46.59)]
def in_mdh(lo,la): return any(a<=lo<=c and b<=la<=d for a,b,c,d in MDH)
pos=[]
rows=[r for r in csv.DictReader(open(f'{H}/labeled/labels.csv')) if r['verdict']=='mound']
dtm_rows=[r for r in rows if r['tile']=='LAKI3_DTM']
mdh_rows=[r for r in rows if in_mdh(float(r['lon']),float(r['lat']))]
for f,r in zip(sorted(glob.glob(f'{H}/dataset_pos/*.png')),dtm_rows): pos.append((f,float(r['lon']),float(r['lat']),'ll','DTM'))
for f,r in zip(sorted(glob.glob(f'{H}/dataset_pos_mdh/*.png')),mdh_rows): pos.append((f,float(r['lon']),float(r['lat']),'ll','MDH'))
dkman={os.path.basename(r['file']):(float(r['est']),float(r['nord'])) for r in csv.DictReader(open(f'{H}/dataset_pos_dk/manifest.csv'))} if os.path.exists(f'{H}/dataset_pos_dk/manifest.csv') else {}
for f in sorted(glob.glob(f'{H}/dataset_pos_dk/*.png')):
    bn=os.path.basename(f)
    if bn in dkman: e,n=dkman[bn]; pos.append((f,e,n,'utm','DK'))
from collections import Counter
print("pozitivi total:",len(pos),dict(Counter(p[4] for p in pos)))
ndk_png=len(glob.glob(f'{H}/dataset_pos_dk/*.png')); ndk_used=sum(1 for p in pos if p[4]=='DK')
if ndk_png>100 and ndk_used<ndk_png*0.9:
    import sys; sys.exit(f"⛔ ABORT: {ndk_png} PNG DK dar doar {ndk_used} în manifest.")
from collections import defaultdict
def xy_m(p):
    if p[3]=='utm': return p[1],p[2]
    la=p[2]*math.pi/180; return p[1]*111320*math.cos(la), p[2]*110540
THR=50.0; iso=[]
for crs in set(p[3] for p in pos):
    pts=[p for p in pos if p[3]==crs]; XY=[xy_m(p) for p in pts]
    grid=defaultdict(list)
    for idx,(x,y) in enumerate(XY): grid[(int(x//THR),int(y//THR))].append(idx)
    for idx,(x,y) in enumerate(XY):
        gx,gy=int(x//THR),int(y//THR); ok=True
        for dx in (-1,0,1):
            for dy in (-1,0,1):
                for j in grid.get((gx+dx,gy+dy),()):
                    if j!=idx and (x-XY[j][0])**2+(y-XY[j][1])**2 < THR*THR: ok=False;break
                if not ok: break
            if not ok: break
        if ok: iso.append(pts[idx])
print("  -> izolati:",len(iso),dict(Counter(p[4] for p in iso)),flush=True)
pos=iso
def grp(p):
    if p[3]=='utm': return ('DK',round(p[1]/3000),round(p[2]/3000))
    return ('RO',round(p[1]*33),round(p[2]*33))
groups=sorted(set(grp(p) for p in pos),key=str); random.shuffle(groups)
ng=len(groups); nte=max(1,ng//7); nva=max(1,ng//7)
test_g=set(groups[:nte]); val_g=set(groups[nte:nte+nva])
poste=[p for p in pos if grp(p) in test_g]; posva=[p for p in pos if grp(p) in val_g]
postr=[p for p in pos if grp(p) not in test_g and grp(p) not in val_g]
print(f"  split geo 3-cai: train {len(postr)} / val {len(posva)} / TEST {len(poste)}")
negdirs=['dataset_neg','dataset_neg_mdh','dataset_neg_dk','dataset_neg_village','dataset_neg_ro_plain','dataset_neg_ro_hill','dataset_neg_terrace','dataset_neg_domefp','dataset_neg_ditch']
negf=[f for nd in negdirs for f in sorted(glob.glob(f'{H}/{nd}/*.png'))]
random.shuffle(negf); negte=negf[:1500]; negva=negf[1500:2100]; negtr=negf[2100:]
print("negative:",len(negf),{nd.replace('dataset_neg','').strip('_') or 'DTM':len(glob.glob(f'{H}/{nd}/*.png')) for nd in negdirs})
def _histeq(a):
    h=np.bincount(a.ravel(),minlength=256).astype(np.float64);cdf=h.cumsum()
    if cdf[-1]==0: return a
    return (cdf[a]/cdf[-1]*255).astype(np.uint8)
def load_raw(f): return np.asarray(Image.open(f).convert('L').resize((128,128)),np.uint8)
def homog(a):
    return _histeq(np.asarray(Image.fromarray(a).filter(ImageFilter.GaussianBlur(0.8)),np.uint8))
# === CANAL 2 = CURBURĂ-proxy: Laplacian al hillshade-ului (convex-dom => semnătură compactă, linie => liniară) ===
def curv(a):
    g=np.asarray(Image.fromarray(a).filter(ImageFilter.GaussianBlur(1.2)),np.float32)
    lap=(np.roll(g,1,0)+np.roll(g,-1,0)+np.roll(g,1,1)+np.roll(g,-1,1)-4*g)
    lo,hi=np.percentile(lap,2),np.percentile(lap,98)
    return np.clip((lap-lo)/(hi-lo+1e-6)*255,0,255).astype(np.uint8)
def stack2(raw): return np.stack([homog(raw),curv(raw)])   # [2,128,128] uint8
def load2(f): return stack2(load_raw(f))
def aug(a): return [t for k in range(4) for t in (np.rot90(a,k),np.fliplr(np.rot90(a,k)))]
N_BAL=3000
postr_ro=[p for p in postr if p[4] in ('DTM','MDH')]; postr_dk=[p for p in postr if p[4]=='DK']
random.shuffle(postr_dk); dk_sel=postr_dk[:N_BAL]
reps=max(1,round(N_BAL/max(1,len(postr_ro)))); ro_sel=postr_ro*reps
postr_bal=dk_sel+ro_sel; random.shuffle(postr_bal)
print(f"  rebalansare pozitivi: RO {len(postr_ro)}x{reps}={len(ro_sel)} + DK {len(dk_sel)} = {len(postr_bal)} (×8 aug)",flush=True)
posaug=[stack2(v) for p in postr_bal for v in aug(load_raw(p[0]))]
print(f"  posaug: {len(posaug)} (2 canale)",flush=True)
Xva=np.array([load2(p[0]) for p in posva]+[load2(f) for f in negva],dtype=np.uint8); Yva=np.array([1.]*len(posva)+[0.]*len(negva)); srcva=[p[4] for p in posva]
Xva_t=torch.tensor(Xva).to(dev); Yva_t=torch.tensor(Yva).float().to(dev)
Xte=np.array([load2(p[0]) for p in poste]+[load2(f) for f in negte],dtype=np.uint8); Yte=np.array([1.]*len(poste)+[0.]*len(negte)); srcte=[p[4] for p in poste]
Xte_t=torch.tensor(Xte).to(dev)
class Net(nn.Module):
    def __init__(s):
        super().__init__();s.c=nn.Sequential(nn.Conv2d(2,16,3,2,1),nn.ReLU(),nn.Conv2d(16,32,3,2,1),nn.ReLU(),nn.Conv2d(32,64,3,2,1),nn.ReLU(),nn.AdaptiveAvgPool2d(1));s.f=nn.Linear(64,1)
    def forward(s,x): return s.f(s.c(x).flatten(1)).squeeze(1)
def build(negs):
    X=np.array(list(posaug)+[load2(f) for f in negs],dtype=np.uint8); Y=[1.]*len(posaug)+[0.]*len(negs)
    return torch.tensor(X),torch.tensor(Y).float()
def train(negs,ep=40):
    Xtr,Ytr=build(negs);Ytr=Ytr.to(dev)
    net=Net().to(dev);opt=torch.optim.Adam(net.parameters(),1e-3,weight_decay=1e-4)
    lf=nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0]).to(dev));n=len(Xtr);bs=64
    for e in range(ep):
        net.train();pm=torch.randperm(n)
        for i in range(0,n,bs):
            idx=pm[i:i+bs];xb=Xtr[idx].float().to(dev)/255.
            opt.zero_grad();lf(net(xb),Ytr[idx]).backward();opt.step()
    return net
def ev(net):
    net.eval()
    with torch.no_grad():
        pv=(torch.sigmoid(net(Xva_t.float()/255.))>0.5).float()
        tp=((pv==1)&(Yva_t==1)).sum().item();fp=((pv==1)&(Yva_t==0)).sum().item();fn=((pv==0)&(Yva_t==1)).sum().item();tn=((pv==0)&(Yva_t==0)).sum().item()
        pvn=pv.cpu().numpy();rec={s:(int(sum(pvn[i] for i in range(len(srcva)) if srcva[i]==s)),sum(1 for s2 in srcva if s2==s)) for s in set(srcva)}
    return tp,fp,fn,tn,rec
def score(net,files):
    net.eval();o=[]
    with torch.no_grad():
        for i in range(0,len(files),256):
            b=files[i:i+256];xb=torch.tensor(np.array([load2(f) for f in b],dtype=np.uint8)).float().to(dev)/255.
            o+=list(zip(torch.sigmoid(net(xb)).cpu().numpy().tolist(),b))
    return o
neg1=negtr[:len(posaug)*3]; net=train(neg1); tp,fp,fn,tn,rec=ev(net)
print(f"RUNDA1: recall {tp}/{tp+fn} | FP {fp}/{fp+tn} ({100*fp/(fp+tn+1e-9):.1f}%) | pe sursa {rec}")
sc=score(net,negtr);sc.sort(reverse=True);hard=[f for s,f in sc[:800]]
neg2=hard*2+negtr[:len(posaug)*3];random.shuffle(neg2)
net2=train(neg2,50);tp,fp,fn,tn,rec=ev(net2)
print(f"RUNDA2(mining): recall {tp}/{tp+fn} | FP {fp}/{fp+tn} ({100*fp/(fp+tn+1e-9):.1f}%) | pe sursa {rec}")
torch.save(net2.state_dict(),f'{H}/combined_2ch.pt');print("model -> combined_2ch.pt")
def probs(net,Xt):
    net.eval();out=[]
    with torch.no_grad():
        for i in range(0,len(Xt),512): out.append(torch.sigmoid(net(Xt[i:i+512].float().to(dev)/255.)).cpu().numpy())
    return np.concatenate(out) if out else np.array([])
pte=probs(net2,Xte_t); yte=Yte
auc=auroc(pte,yte); ap=auprc(pte,yte); pred=(pte>0.5)
tp=int(((pred==1)&(yte==1)).sum());fp=int(((pred==1)&(yte==0)).sum());fn=int(((pred==0)&(yte==1)).sum());tn=int(((pred==0)&(yte==0)).sum())
sens=tp/(tp+fn+1e-9);spec=tn/(tn+fp+1e-9)
recte={s:(int(sum((pred[i]==1) for i in range(len(srcte)) if srcte[i]==s)),sum(1 for s2 in srcte if s2==s)) for s in set(srcte)}
print("=== TEST 2-CANALE (held-out geografic) ===")
print(f"  poz {int(yte.sum())} / neg {int(len(yte)-yte.sum())}")
print(f"  AUROC {auc:.3f} | AUPRC {ap:.3f}")
print(f"  @0.5: sens {sens:.3f} ({tp}/{tp+fn}) | spec {spec:.3f} ({tn}/{tn+fp}) | recall pe sursa {recte}")
