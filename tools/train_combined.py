#!/usr/bin/env python3
# train_combined.py — CNN pe set COMBINAT-ECHILIBRAT 3 surse: RO-DTM(Dolj) + RO-MDH(Arad) + DK(Rundhoj).
# Multi-CRS: RO=lon/lat, DK=UTM EPSG:25832. Izolare+split geo PER-CRS. Hard-negative mining x2.
# Vezi MODEL.md. Sursa decorelata de eticheta (fiecare sursa are poz+neg) -> robustete.
import os,glob,csv,math,random
import numpy as np
from PIL import Image,ImageDraw,ImageFont,ImageFilter
import torch,torch.nn as nn
random.seed(0); torch.manual_seed(0); np.random.seed(0)
H=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); dev=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
# AUROC (rang Mann-Whitney, ties mediate) + AUPRC (average precision) — fara sklearn
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
# === POZITIVI din 3 surse: (file, x, y, crs, source) ===
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
# GARD anti-eșec-tăcut: DK se adaugă DOAR cu manifest (coord pt split geo); manifestul e scris de
# dk_download_all.py abia la FINAL. Dacă PNG-urile DK sunt pe disc dar manifestul lipsește/e parțial,
# DK ar fi tăcut exclus -> antrenare pe ~87 poz în loc de ~17k. Oprește zgomotos.
ndk_png=len(glob.glob(f'{H}/dataset_pos_dk/*.png')); ndk_used=sum(1 for p in pos if p[4]=='DK')
if ndk_png>100 and ndk_used<ndk_png*0.9:
    import sys; sys.exit(f"⛔ ABORT: {ndk_png} PNG DK pe disc dar doar {ndk_used} în manifest → DK aproape absent. Rulează dk_download_all.py până la capăt (scrie manifestul la final) ÎNAINTE de antrenare.")
# izolare >50m (per-CRS) cu GRILĂ spațială O(n) — vechiul O(n²) cădea pe 21k pozitivi (~440M comparații).
from collections import defaultdict
def xy_m(p):  # coord în metri (utm direct; ll -> metri local)
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
ng=len(groups); nte=max(1,ng//7); nva=max(1,ng//7)   # ~15% TEST, ~15% val, ~70% train (pe GRUPURI geo disjuncte)
test_g=set(groups[:nte]); val_g=set(groups[nte:nte+nva])
poste=[p for p in pos if grp(p) in test_g]; posva=[p for p in pos if grp(p) in val_g]
postr=[p for p in pos if grp(p) not in test_g and grp(p) not in val_g]
print(f"  split geo 3-cai: train {len(postr)} / val {len(posva)} / TEST {len(poste)} (grupuri geo disjuncte)")
# === NEGATIVE din toate sursele ===
# anthro (mining pe scor) SCOS: selectia pe scor mare culege fix movilele reale -> negative otravite.
# Inlocuit cu dataset_neg_village (random in sate = contaminare neglijabila). Cele 1193 anthro raman
# pe disc ca CANDIDATI de descoperire (active learning), nu ca negative.
# NB: dataset_neg_mdh_linear SCOS intentionat 22.06 — negativele de șanț reduc FP DAR suprimă movile reale
# NOTĂ 22.06 ~20:50: dataset_neg_mdh_linear (3310, negative LINIARE Arad) E în producție (combined_cnn.pt =
# combined_pre_smallditch). „Colapsul recall 71%->39%" de care se temea comentariul vechi = ARTEFACT de centrare
# (corectat la 98%->87% cu fereastra ±30m); decizia FINALĂ (re-revert msg 10665) a păstrat modelul liniar ca
# producție. Îl includem ca să reproducem producția; singura schimbare la retrain = pozitivii triați după claritate.
_NEGBASE=['dataset_neg','dataset_neg_mdh','dataset_neg_dk','dataset_neg_village','dataset_neg_ro_plain','dataset_neg_ro_hill','dataset_neg_terrace','dataset_neg_domefp','dataset_neg_ditch','dataset_neg_mdh_linear','dataset_neg_heatmap_zone2','dataset_neg_ro_fp5k','dataset_neg_ro_fp5k2','dataset_neg_expert_fp','dataset_neg_ro_fp20k']
# dataset_neg_ro_fp5k (25.06, Andrei): hard-negative mining RO (arătură/șanț/pârâu/antropic), dome-veto central. harvest_fp_hard.py.
# dataset_neg_ro_fp5k2 (25.06): runda 2 hard-mining pe modelul promovat (fphard). Cumulativ ~9.3k hard-neg curate.
# FORMCLEAN (25.06, principiul de formă Andrei): env NODOMEFP=1 scoate dataset_neg_domefp (domurile nu-s negative).
negdirs=[d for d in _NEGBASE if not (os.environ.get('NODOMEFP') and d=='dataset_neg_domefp')]
negf=[f for nd in negdirs for f in sorted(glob.glob(f'{H}/{nd}/*.png'))]
random.shuffle(negf); negte=negf[:1500]; negva=negf[1500:2100]; negtr=negf[2100:]
print("negative:",len(negf),{nd.replace('dataset_neg','').strip('_') or 'DTM':len(glob.glob(f'{H}/{nd}/*.png')) for nd in negdirs})
# MEMORIE: stocam imaginile ca uint8 (4x mai putin ca float32) si normalizam per-batch pe GPU.
# Tensorii mari raman pe CPU; doar batch-ul ajunge pe MPS. -> ruleaza pe Mac Mini M4 16GB fara OOM.
# OMOGENIZARE INTENSITATE/ZGOMOT (insight Andrei): DK-poz și RO-neg trebuie să aibă ACEEAȘI
# distribuție de intensitate + claritate, altfel modelul învață „stil sursă" nu forma movilei
# (cauza reală a AUROC umflat). blur comun (egalizează claritatea DK 0.4m vs RO 0.5m) + egalizare
# histogramă (distribuție identică pe toate sursele). Aplicată IDENTIC la TOATE stampele (pos+neg).
def _histeq(a):
    h=np.bincount(a.ravel(),minlength=256).astype(np.float64);cdf=h.cumsum()
    if cdf[-1]==0: return a
    return (cdf[a]/cdf[-1]*255).astype(np.uint8)
def load_raw(f): return np.asarray(Image.open(f).convert('L').resize((128,128)),np.uint8)
def homog(a):  # blur comun + egalizare histogramă
    return _histeq(np.asarray(Image.fromarray(a).filter(ImageFilter.GaussianBlur(0.8)),np.uint8))
def load(f): return homog(load_raw(f))
# AUGMENTARE MULTI-REZOLUȚIE (insight Andrei): degradez la ~3-5m (downsample px apoi înapoi la 128)
# ca să imite datele grosiere (Iași=5m). Aplicat pe POZ ȘI NEG: modelul vede și NEGATIVE de 5m
# (teren grosier fără movilă) → nu mai crede că tot ce-i bloc la 5m e movilă. 16px≈5m, 24px≈3.3m la 80m.
def degrade(a,px): return np.asarray(Image.fromarray(a).resize((px,px),Image.BILINEAR).resize((128,128),Image.BILINEAR),np.uint8)
DEGR=[16,24]
def aug(a): return [t for k in range(4) for t in (np.rot90(a,k),np.fliplr(np.rot90(a,k)))]
# REBALANSARE SURSĂ (insight Andrei): RO (~zeci) e înecat de DK (~12k) -> detector de movile DANEZE.
# Subesantionez DK la N_BAL + oversample RO la ~N_BAL -> 1:1 RO:DK, fiecare sursă contează egal.
N_BAL=3000
postr_ro=[p for p in postr if p[4] in ('DTM','MDH')]; postr_dk=[p for p in postr if p[4]=='DK']
# HARD-POSITIVE oversampling (25.06, recuperare recall): pozitivii faint/scarificați pe care hard-negativele
# i-au suprimat (scor mic) -> replic EXTRA, ca să nu fie striviți. Env HARDPOS_FILE=listă căi + HARDPOS_REP=N.
_HP=os.environ.get('HARDPOS_FILE');_HR=int(os.environ.get('HARDPOS_REP','0'))
if _HP and _HR>0 and os.path.exists(_HP):
    hpset=set(l.strip() for l in open(_HP) if l.strip())
    extra=[p for p in postr_ro if os.path.abspath(p[0]) in hpset]*_HR
    postr_ro=postr_ro+extra
    print(f"  HARD-POS oversample: +{len(extra)} copii ({sum(1 for p in postr_ro if os.path.abspath(p[0]) in hpset)//(_HR+1) if _HR else 0} faint ×{_HR}) -> postr_ro {len(postr_ro)}",flush=True)
random.shuffle(postr_dk); dk_sel=postr_dk[:N_BAL]
reps=max(1,round(N_BAL/max(1,len(postr_ro)))); ro_sel=postr_ro*reps
postr_bal=dk_sel+ro_sel; random.shuffle(postr_bal)
print(f"  rebalansare pozitivi: RO {len(postr_ro)}x{reps}={len(ro_sel)} + DK {len(dk_sel)} (din {len(postr_dk)}) = {len(postr_bal)} (×8 aug)",flush=True)
# multi-rezoluție SCOASĂ (a crescut FP + n-a rezolvat Iași — vezi memorie). Doar homog + rebalansare.
posaug=[homog(v) for p in postr_bal for v in aug(load_raw(p[0]))]
print(f"  posaug: {len(posaug)}",flush=True)
Xva=np.array([load(p[0]) for p in posva]+[load(f) for f in negva],dtype=np.uint8); Yva=np.array([1.]*len(posva)+[0.]*len(negva)); srcva=[p[4] for p in posva]
Xva_t=torch.tensor(Xva).to(dev); Yva_t=torch.tensor(Yva).float().to(dev)   # uint8 pe dev (val mic), normalizat la folosire
# TEST = held-out geografic; NU se atinge in tuning; scorat O SINGURA data la final
Xte=np.array([load(p[0]) for p in poste]+[load(f) for f in negte],dtype=np.uint8); Yte=np.array([1.]*len(poste)+[0.]*len(negte)); srcte=[p[4] for p in poste]
Xte_t=torch.tensor(Xte).to(dev)
class Net(nn.Module):
    def __init__(s):
        super().__init__();s.c=nn.Sequential(nn.Conv2d(1,16,3,2,1),nn.ReLU(),nn.Conv2d(16,32,3,2,1),nn.ReLU(),nn.Conv2d(32,64,3,2,1),nn.ReLU(),nn.AdaptiveAvgPool2d(1));s.f=nn.Linear(64,1)
    def forward(s,x): return s.f(s.c(x).flatten(1)).squeeze(1)
def build(negs):
    X=np.array(list(posaug)+[load(f) for f in negs],dtype=np.uint8); Y=[1.]*len(posaug)+[0.]*len(negs)
    return torch.tensor(X),torch.tensor(Y).float()   # X uint8 pe CPU
import torch.nn.functional as F
# AUGMENTARE DE SCARĂ (23.06): movilele mici/scarificate + șanțurile mari erau ratate/FP fiindcă modelul nu-i
# invariant la scară (fereastră fixă 80m). Zoom random per BATCH la antrenare (0.6-1.6x) -> modelul vede movile
# ȘI negative (șanțuri) la mărimi variate -> prinde movile mici (recall) + respinge șanțuri la TOATE scările (FP).
def scale_jit(xb):
    fac=random.uniform(0.6,1.6);s=max(16,int(round(128*fac)))
    y=F.interpolate(xb,size=(s,s),mode='bilinear',align_corners=False)
    if s>=128: o=(s-128)//2;return y[:,:,o:o+128,o:o+128]
    p=(128-s)//2;return F.pad(y,(p,128-s-p,p,128-s-p),mode='replicate')
# AUGMENTARE DE TRANSLAȚIE (insight Andrei 24.06): pozitivii au movila FIX în centru -> model sensibil la
# poziție (aceeași movilă decentrată scorează 0.0 vs 0.7). Jitter random per-sample (px), reflection pad ->
# model invariant la translație. Env JITTER=px (ex 24 ≈ ±15m). Aplicat pe POZ ȘI NEG (poziția nu poartă etichetă).
JITTER=float(os.environ.get('JITTER','0'))
def trans_jit(xb):
    if JITTER<=0: return xb
    B=xb.shape[0];m=JITTER/64.0
    tx=(torch.rand(B,device=xb.device)*2-1)*m; ty=(torch.rand(B,device=xb.device)*2-1)*m
    th=torch.zeros(B,2,3,device=xb.device); th[:,0,0]=1; th[:,1,1]=1; th[:,0,2]=tx; th[:,1,2]=ty
    grid=F.affine_grid(th,xb.shape,align_corners=False)
    return F.grid_sample(xb,grid,align_corners=False,padding_mode='reflection')
def train(negs,ep=40):
    Xtr,Ytr=build(negs);Ytr=Ytr.to(dev)   # Xtr ramane uint8 pe CPU; batch -> GPU normalizat
    net=Net().to(dev);opt=torch.optim.Adam(net.parameters(),1e-3,weight_decay=1e-4)
    lf=nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0]).to(dev));n=len(Xtr);bs=64  # 3.0->1.0: cu 12.8k poz reali augmentarea+pos_weight emfazau pozitivii ~24x -> supra-declansare
    for e in range(ep):
        net.train();pm=torch.randperm(n)
        for i in range(0,n,bs):
            idx=pm[i:i+bs];xb=Xtr[idx].unsqueeze(1).float().to(dev)/255.  # SINGLE-scale (scale_jit a scăzut recall scarificați 23.06 — fixul = pozitivi scarificați reali, nu augmentare)
            xb=trans_jit(xb)  # jitter de translație (env JITTER>0) — invarianță la poziție
            opt.zero_grad();lf(net(xb),Ytr[idx]).backward();opt.step()
    return net
def ev(net):
    net.eval()
    with torch.no_grad():
        pv=(torch.sigmoid(net(Xva_t.unsqueeze(1).float()/255.))>0.5).float()
        tp=((pv==1)&(Yva_t==1)).sum().item();fp=((pv==1)&(Yva_t==0)).sum().item();fn=((pv==0)&(Yva_t==1)).sum().item();tn=((pv==0)&(Yva_t==0)).sum().item()
        pvn=pv.cpu().numpy();rec={s:(int(sum(pvn[i] for i in range(len(srcva)) if srcva[i]==s)),sum(1 for s2 in srcva if s2==s)) for s in set(srcva)}
    return tp,fp,fn,tn,rec
def score(net,files):
    net.eval();o=[]
    with torch.no_grad():
        for i in range(0,len(files),256):
            b=files[i:i+256];xb=torch.tensor(np.array([load(f) for f in b],dtype=np.uint8)).unsqueeze(1).float().to(dev)/255.
            o+=list(zip(torch.sigmoid(net(xb)).cpu().numpy().tolist(),b))
    return o
neg1=negtr[:len(posaug)*3]; net=train(neg1); tp,fp,fn,tn,rec=ev(net)
print(f"RUNDA1: recall {tp}/{tp+fn} | FP {fp}/{fp+tn} ({100*fp/(fp+tn+1e-9):.1f}%) | pe sursa {rec}")
sc=score(net,negtr);sc.sort(reverse=True);hard=[f for s,f in sc[:800]]
neg2=hard*2+negtr[:len(posaug)*3];random.shuffle(neg2)
net2=train(neg2,50);tp,fp,fn,tn,rec=ev(net2)
print(f"RUNDA2(mining): recall {tp}/{tp+fn} | FP {fp}/{fp+tn} ({100*fp/(fp+tn+1e-9):.1f}%) | pe sursa {rec}")
OUT=os.environ.get('OUT',f'{H}/combined_cnn.pt')
torch.save(net2.state_dict(),OUT);print(f"model -> {OUT}")
# === EVALUARE FINALA pe TEST (held-out geografic, O SINGURA data) ===
def probs(net,Xt):
    net.eval();out=[]
    with torch.no_grad():
        for i in range(0,len(Xt),512): out.append(torch.sigmoid(net(Xt[i:i+512].unsqueeze(1).float().to(dev)/255.)).cpu().numpy())
    return np.concatenate(out) if out else np.array([])
pte=probs(net2,Xte_t); yte=Yte
auc=auroc(pte,yte); ap=auprc(pte,yte); pred=(pte>0.5)
tp=int(((pred==1)&(yte==1)).sum());fp=int(((pred==1)&(yte==0)).sum());fn=int(((pred==0)&(yte==1)).sum());tn=int(((pred==0)&(yte==0)).sum())
sens=tp/(tp+fn+1e-9);spec=tn/(tn+fp+1e-9)
recte={s:(int(sum((pred[i]==1) for i in range(len(srcte)) if srcte[i]==s)),sum(1 for s2 in srcte if s2==s)) for s in set(srcte)}
print("=== TEST (held-out geografic, o singura data) ===")
print(f"  poz {int(yte.sum())} / neg {int(len(yte)-yte.sum())}")
print(f"  AUROC {auc:.3f} | AUPRC {ap:.3f}")
print(f"  @0.5: sensibilitate {sens:.3f} ({tp}/{tp+fn}) | specificitate {spec:.3f} ({tn}/{tn+fp}) | recall pe sursa {recte}")
