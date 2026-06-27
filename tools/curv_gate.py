#!/usr/bin/env python3
# curv_gate.py train FEAT_CSV GATE_JSON      — antreneaza filtru de curbura (logistic, numpy) pe feat cu coloana label=pos/neg
# curv_gate.py apply GATE_JSON FEAT_CSV OUT_CSV  — adauga coloana pgate (prob tumul) si keep (1/0 la pragul din gate)
# Trasaturi folosite: convex,convex_in,rugoz,asim,relief_m,slrm_pk,mono (NA -> exclus la train, pgate=NA la apply).
# Geo-split CV (bucket 0.1°) ca sa nu scurga vecini. Raporteaza AUC per-trasatura + AUC CV + cat % neg taie la recall_pos tinta.
import sys,json,csv,math
import numpy as np
NONFEAT={'label','src','lon','lat','idx','board','zona','score','istp','pgate','keep','longitude','latitude'}
def detect_feats(rows):
    feats=[]
    for c in rows[0].keys():
        if c.lower() in NONFEAT:continue
        # numeric in at least one row?
        for r in rows:
            try:
                float(r[c]);feats.append(c);break
            except:
                if r[c]=='NA':continue
                else:break
    return feats
FEATS=['convex','convex_in','rugoz','asim','relief_m','slrm_pk','mono']  # override in load/apply
def load(path):
    global FEATS
    rows=list(csv.DictReader(open(path)));FEATS=detect_feats(rows);X=[];y=[];ll=[];keep=[]
    for r in rows:
        try:
            v=[float(r[f]) for f in FEATS]
        except:
            keep.append(False);continue
        if any(math.isnan(x) or math.isinf(x) for x in v):keep.append(False);continue
        X.append(v);keep.append(True)
        y.append(1 if r.get('label','').lower().startswith('pos') else 0)
        try:ll.append((float(r.get('lon',0)),float(r.get('lat',0))))
        except:ll.append((0,0))
    return np.array(X),np.array(y),ll,rows,keep
def auc(score,y):
    # Mann-Whitney AUC of score predicting y=1
    pos=score[y==1];neg=score[y==0]
    if len(pos)==0 or len(neg)==0:return 0.5
    allv=np.concatenate([pos,neg]);order=allv.argsort();ranks=np.empty(len(allv));ranks[order]=np.arange(1,len(allv)+1)
    # tie-correct via average ranks
    _,inv,cnt=np.unique(allv,return_inverse=True,return_counts=True)
    csum=np.cumsum(cnt);start=csum-cnt;avg=(start+csum+1)/2.0;ranks=avg[inv]
    rpos=ranks[:len(pos)].sum();return (rpos-len(pos)*(len(pos)+1)/2)/(len(pos)*len(neg))
def fit_logistic(X,y,iters=4000,lr=0.2,l2=1e-3):
    mu=X.mean(0);sd=X.std(0)+1e-9;Z=(X-mu)/sd
    n,d=Z.shape;w=np.zeros(d);b=0.0
    npos=max(1,(y==1).sum());nneg=max(1,(y==0).sum())
    cw=np.where(y==1,n/(2*npos),n/(2*nneg))
    for _ in range(iters):
        p=1/(1+np.exp(-(Z@w+b)));g=(p-y)*cw
        gw=Z.T@g/n+l2*w;gb=g.mean()
        w-=lr*gw;b-=lr*gb
    return dict(w=w.tolist(),b=float(b),mu=mu.tolist(),sd=sd.tolist(),feats=list(FEATS))
def predict(gate,X):
    w=np.array(gate['w']);b=gate['b'];mu=np.array(gate['mu']);sd=np.array(gate['sd'])
    Z=(X-mu)/sd;return 1/(1+np.exp(-(Z@w+b)))
def geo_folds(ll,k=5):
    cells={};fold=[]
    for lon,lat in ll:
        c=(round(lon*10),round(lat*10));fold.append(c)
    uniq=sorted(set(fold));cmap={c:(hash(c)%k) for i,c in enumerate(uniq)}
    return np.array([cmap[c] for c in fold])
def train(feat_csv,gate_json):
    X,y,ll,rows,keep=load(feat_csv)
    print(f"train: {len(y)} randuri | pos {int((y==1).sum())} neg {int((y==0).sum())}")
    print("AUC per-trasatura (>0.5 = mai mare la tumul; <0.5 = mai mic la tumul):")
    for j,f in enumerate(FEATS):
        a=auc(X[:,j],y);print(f"  {f:10} AUC {a:.3f}")
    # geo-split CV
    folds=geo_folds(ll,5);oof=np.full(len(y),np.nan)
    for k in range(5):
        tr=folds!=k;te=folds==k
        if te.sum()==0 or (y[tr]==1).sum()<3:continue
        g=fit_logistic(X[tr],y[tr]);oof[te]=predict(g,X[te])
    m=~np.isnan(oof);cvauc=auc(oof[m],y[m])
    print(f"\nLOGISTIC geo-split CV AUC: {cvauc:.3f}  (out-of-fold, {m.sum()} randuri)")
    # operating: la ce prag tinem recall_pos, cat neg taiem (din OOF)
    pos=oof[m & (y==1)];neg=oof[m & (y==0)]
    print("  Prag gate | recall_pos | %neg(FP) taiate (din OOF, gate=keep daca pgate>=prag):")
    for rec in (1.0,0.97,0.95,0.90):
        thr=np.quantile(pos,1-rec) if rec<1.0 else pos.min()
        kept_pos=(pos>=thr).mean();cut_neg=(neg<thr).mean()
        print(f"    {thr:.3f}   recall_pos {kept_pos*100:.0f}%   taie {cut_neg*100:.0f}% din FP")
    # final fit pe tot
    g=fit_logistic(X,y);g['cvauc']=cvauc
    # praguri sugerate (din OOF): cel mai agresiv prag care tine recall_pos=100% si 95%
    g['thr_rec100']=float(np.quantile(pos,0.0));g['thr_rec95']=float(np.quantile(pos,0.05))
    json.dump(g,open(gate_json,'w'),indent=1);print(f"-> {gate_json} (thr_rec100={g['thr_rec100']:.3f}, thr_rec95={g['thr_rec95']:.3f})")
def apply(gate_json,feat_csv,out_csv,thr=None):
    global FEATS
    gate=json.load(open(gate_json));FEATS=gate.get('feats',FEATS);thr=thr if thr is not None else gate.get('thr_rec100',0.5)
    rdr=csv.DictReader(open(feat_csv));rows=list(rdr);hdr=list(rdr.fieldnames or [])+['pgate','keep']
    with open(out_csv,'w',newline='') as fo:
        wr=csv.writer(fo);wr.writerow(hdr);ok=0;cut=0
        for r in rows:
            try:
                v=np.array([[float(r[f]) for f in FEATS]])
                if np.any(~np.isfinite(v)):raise ValueError
                p=float(predict(gate,v)[0]);k=1 if p>=thr else 0;ok+=1;cut+=(1-k)
                wr.writerow([r[c] for c in rows[0].keys()]+[f"{p:.4f}",k])
            except:
                wr.writerow([r[c] for c in rows[0].keys()]+['NA',1])  # NA -> nu suprima (fail-open)
    print(f"-> {out_csv} | scored {ok} | gate-cut {cut} (prag {thr:.3f})")
if __name__=='__main__':
    if sys.argv[1]=='train':train(sys.argv[2],sys.argv[3])
    elif sys.argv[1]=='apply':apply(sys.argv[2],sys.argv[3],sys.argv[4],float(sys.argv[5]) if len(sys.argv)>5 else None)
