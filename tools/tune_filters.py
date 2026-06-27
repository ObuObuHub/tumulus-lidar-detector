#!/usr/bin/env python3
# tune_filters.py — grid-search praguri filtru (coerență direcțională @rază + liniaritate) care taie cele mai multe
# fals-pozitive păstrând movilele, SIMULTAN pe RO (Catane+batch FP) și NL (movile OSM+control).
# Constrângere DURĂ: niciuna din cele 10 movile Catane nu poate fi suprimată. -> raportează top combinații.
import csv,itertools,math
RO=list(csv.DictReader(open('/tmp/ro_feat.csv')))
NL=list(csv.DictReader(open('/tmp/nl_feat.csv')))
COHS={'15':'coh15','22':'coh22','30':'coh30','40':'coh40'}
def gf(r,key): return float(r[key])
# filtru: SUPRIMAT dacă coh[rad] > cT  SAU  lin > lT
def suppressed(r,radkey,cT,lT): return (gf(r,radkey)>cT) or (gf(r,'lin')>lT)
ro_pos=[r for r in RO if r['label']=='pos'];ro_neg=[r for r in RO if r['label']=='neg']
ro_cat=[r for r in RO if r['kind'].startswith('catane')]
# NL: filtrul contează doar la detecții (score>=0.7)
nl_pos7=[r for r in NL if r['label']=='pos' and float(r['score'])>=0.7]
nl_neg7=[r for r in NL if r['label']=='neg' and float(r['score'])>=0.7]
def auroc(pos,neg):
    a=sorted([(v,1) for v in pos]+[(v,0) for v in neg]);n=len(pos)*len(neg)
    if not n: return 0.5
    rank=0;c=0
    for i,(v,lab) in enumerate(sorted([(float(x['__g']),x['__l']) for x in pos+neg])):
        pass
    return None
# AUROC gated NL: scor->0 dacă suprimat, peste TOATE punctele NL
def auroc_nl(radkey,cT,lT):
    P=[0.0 if suppressed(r,radkey,cT,lT) else float(r['score']) for r in NL if r['label']=='pos']
    N=[0.0 if suppressed(r,radkey,cT,lT) else float(r['score']) for r in NL if r['label']=='neg']
    allv=sorted([(v,1) for v in P]+[(v,0) for v in N])
    # Mann-Whitney via rank sum
    vals=[v for v,_ in allv];labs=[l for _,l in allv]
    import numpy as np
    a=np.array([v for v in P]+[v for v in N]);rr=a.argsort().argsort()+1
    return (rr[:len(P)].sum()-len(P)*(len(P)+1)/2)/(len(P)*len(N))
import numpy as np
rows=[]
for radkey in ['coh15','coh22','coh30','coh40']:
    for cT in [0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,1.01]:
        for lT in [3,4,6,8,99]:
            cat_lost=sum(1 for r in ro_cat if suppressed(r,radkey,cT,lT))
            ro_keep=1-sum(1 for r in ro_pos if suppressed(r,radkey,cT,lT))/max(1,len(ro_pos))
            ro_sup=sum(1 for r in ro_neg if suppressed(r,radkey,cT,lT))/max(1,len(ro_neg))
            nl_keep=1-(sum(1 for r in nl_pos7 if suppressed(r,radkey,cT,lT))/max(1,len(nl_pos7)))
            nl_sup=sum(1 for r in nl_neg7 if suppressed(r,radkey,cT,lT))/max(1,len(nl_neg7))
            nlauc=auroc_nl(radkey,cT,lT)
            rows.append(dict(rad=radkey,cT=cT,lT=lT,cat_lost=cat_lost,ro_keep=ro_keep,ro_sup=ro_sup,
                             nl_keep=nl_keep,nl_sup=nl_sup,nlauc=nlauc))
# baseline (fără filtru)
base_nlauc=auroc_nl('coh15',1.01,99)
print(f"BASELINE (fără filtru): NL AUROC {base_nlauc:.3f} | RO FP suprimate 0% | NL FP@0.7 suprimate 0%")
print(f"  populații: RO {len(ro_pos)}poz/{len(ro_neg)}neg (Catane {len(ro_cat)}) | NL@0.7 {len(nl_pos7)}poz/{len(nl_neg7)}neg")
# constrângeri: cat_lost==0, ro_keep>=0.95, nl_keep>=0.92. Obiectiv: maximizează (ro_sup + nl_sup + (nlauc-base))
def obj(r): return r['ro_sup']+r['nl_sup']+ (r['nlauc']-base_nlauc)*2
valid=[r for r in rows if r['cat_lost']==0 and r['ro_keep']>=0.95 and r['nl_keep']>=0.92 and not(r['cT']>1 and r['lT']>90)]
valid.sort(key=obj,reverse=True)
print(f"\n=== TOP 12 combinații (constrângere: 0 Catane pierdute, ≥95% pozitivi RO, ≥92% movile NL@0.7) ===")
print(f"{'rad':5} {'cohT':>5} {'linT':>4} | {'RO_keep':>7} {'RO_supFP':>8} | {'NL_keep':>7} {'NL_supFP':>8} {'NL_AUROC':>8} | {'obj':>5}")
for r in valid[:12]:
    print(f"{r['rad']:5} {r['cT']:>5.2f} {r['lT']:>4} | {r['ro_keep']*100:>6.0f}% {r['ro_sup']*100:>7.0f}% | {r['nl_keep']*100:>6.0f}% {r['nl_sup']*100:>7.0f}% {r['nlauc']:>8.3f} | {obj(r):>5.2f}")
