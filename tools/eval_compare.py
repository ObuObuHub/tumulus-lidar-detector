import csv
truth={11,17,18,30,43,45,50,55,57,64}
fp_idx={10,23,34,38,56,63}
def load(p): return {int(r['idx']):(float(r['score']) if r['score']!='NA' else 0.0) for r in csv.DictReader(open(p))}
terr=load('/tmp/eval_rescore_terrace.csv'); dome=load('/tmp/eval_rescore.csv')
shape={int(r['idx']):float(r['axis_ratio']) for r in csv.DictReader(open('labeled/eval_session_22iun/eval_shape.csv'))}

def metrics(sc, thr, sf=True, ratio=2.75):
    pred={i for i,s in sc.items() if s>=thr and (not sf or shape.get(i,99)<=ratio)}
    tp=len(pred&truth); fp=len(pred-truth)
    return tp, fp, len(pred), tp/len(truth), (tp/len(pred) if pred else 0)
def best_op(sc, sf=True):
    cand=sorted({round(s,3) for s in sc.values()}); best=None
    for thr in cand:
        tp,fp,n,rec,prec=metrics(sc,thr,sf)
        if rec>=1.0: best=(thr,prec,fp,n)
    return best

print("OPERATING POINT (prag max cu recall 100%, + filtru formă):")
for nm,sc in [("terrace (înainte)",terr),("dome-FP (acum)",dome)]:
    b=best_op(sc)
    if b: print(f"  {nm:20s}: thr≤{b[0]:.3f}  precizie={b[1]*100:.0f}%  FP={b[2]}  (pred {b[3]})")
    else: print(f"  {nm:20s}: NU mai există prag cu recall 100% (a picat un tumul real!)")

print("\nCei 6 FP rotunzi Catane — scor terrace -> dome-FP (a scăzut = bine):")
drops=0
for i in sorted(fp_idx):
    t,d=terr[i],dome[i]; ar=shape.get(i,0)
    flag="↓" if d<t-0.02 else ("↑" if d>t+0.02 else "=")
    if d<t-0.02: drops+=1
    print(f"  fp{i:2d}: {t:.3f} -> {d:.3f}  {flag}  (raport {ar:.2f})")
print(f"  -> {drops}/6 FP au scăzut")

print("\nCei 10 tumuli REALI — scor terrace -> dome-FP (NU trebuie să scadă sub prag):")
rec_lost=0
for i in sorted(truth):
    t,d=terr[i],dome[i]
    flag="↓↓" if d<t-0.05 else ""
    print(f"  idx{i:2d}: {t:.3f} -> {d:.3f}  {flag}")
print(f"  min scor tumul real (dome-FP) = {min(dome[i] for i in truth):.3f}")
