#!/usr/bin/env python3
# tune_final.py — raport final tuning filtru coerență (rază 22m). Confirmă precizia Catane end-to-end cu filtrul,
# suprimarea FP broad RO + NL AUROC/FP, la praguri 0.65/0.70/0.75, + penalizare soft. Folosește /tmp/ro_feat.csv + /tmp/nl_feat.csv + /tmp/eval_rescore.csv
import csv,numpy as np
RO=list(csv.DictReader(open('/tmp/ro_feat.csv')));NL=list(csv.DictReader(open('/tmp/nl_feat.csv')))
sc={int(r['idx']):(float(r['score']) if r['score'] not in('NA','') else 0.0) for r in csv.DictReader(open('/tmp/eval_rescore.csv'))}
TRUTH={11,17,18,30,43,45,50,55,57,64}
# coh22 pt fiecare idx Catane (din ro_feat: catane{i} și catfp{i})
coh={}
for r in RO:
    k=r['kind']
    if k.startswith('catane'): coh[int(k[6:])]=float(r['coh22'])
    elif k.startswith('catfp'): coh[int(k[5:])]=float(r['coh22'])
def catane_prec(THR):
    s2={i:(0.0 if coh.get(i,0)>THR else v) for i,v in sc.items()}
    real={i:s2[i] for i in TRUTH};op=min(real.values());mi=min(real,key=real.get)
    fp=[(i,s2[i]) for i in s2 if i not in TRUTH and s2[i]>=op]
    return op,mi,len(fp),100*len(TRUTH)/(len(TRUTH)+len(fp)),fp
print("=== CATANE held-out (RO sacru) — precizie@recall100 ===")
op,mi,nfp,prec,fp=catane_prec(99)  # baseline fără filtru
print(f"  BASELINE: prag {op:.3f}(idx{mi}) | FP={nfp} {[(i,round(s,2)) for i,s in fp]} | precizie {prec:.0f}%")
for THR in [0.75,0.70,0.65,0.60]:
    op,mi,nfp,prec,fp=catane_prec(THR)
    lost=[i for i in TRUTH if coh.get(i,0)>THR]
    print(f"  coh22>{THR}: prag {op:.3f}(idx{mi}) | FP={nfp} {[(i,round(s,2)) for i,s in fp]} | precizie {prec:.0f}% | movile pierdute: {lost}")
# RO broad FP (batchfp) suprimare
ro_neg=[float(r['coh22']) for r in RO if r['kind']=='batchfp']
ro_pos=[float(r['coh22']) for r in RO if r['kind'].startswith(('catane','ropos'))]
print("\n=== RO broad (412 batch-FP marcate / 21 pozitivi) — suprimare la coh22>THR ===")
for THR in [0.75,0.70,0.65,0.60,0.50]:
    sup=100*np.mean([c>THR for c in ro_neg]);keep=100*np.mean([c<=THR for c in ro_pos])
    print(f"  coh22>{THR}: FP suprimate {sup:.0f}% | pozitivi păstrați {keep:.0f}%")
# NL
def auroc(P,N):
    a=np.array(P+N);rr=a.argsort().argsort()+1;return (rr[:len(P)].sum()-len(P)*(len(P)+1)/2)/(len(P)*len(N))
nlP=[(float(r['score']),float(r['coh22'])) for r in NL if r['label']=='pos']
nlN=[(float(r['score']),float(r['coh22'])) for r in NL if r['label']=='neg']
print(f"\n=== NL (245 movile / 434 control) ===")
print(f"  BASELINE AUROC {auroc([s for s,_ in nlP],[s for s,_ in nlN]):.3f}")
print("  --- gate dur coh22>THR (scor->0) ---")
for THR in [0.75,0.70,0.65,0.60,0.50]:
    P=[0.0 if c>THR else s for s,c in nlP];N=[0.0 if c>THR else s for s,c in nlN]
    # FP@0.7: control care erau ≥0.7 și sunt tăiate
    n7=[(s,c) for s,c in nlN if s>=0.7];p7=[(s,c) for s,c in nlP if s>=0.7]
    fpsup=100*np.mean([c>THR for s,c in n7]) if n7 else 0
    plost=100*np.mean([c>THR for s,c in p7]) if p7 else 0
    print(f"  coh22>{THR}: AUROC {auroc(P,N):.3f} | FP@0.7 suprimate {fpsup:.0f}% | movile@0.7 pierdute {plost:.0f}%")
print("  --- penalizare SOFT: scor*(1-coh22)^gamma ---")
for g in [0.5,1.0,1.5,2.0,3.0]:
    P=[s*(1-c)**g for s,c in nlP];N=[s*(1-c)**g for s,c in nlN]
    print(f"  gamma={g}: AUROC {auroc(P,N):.3f}")
