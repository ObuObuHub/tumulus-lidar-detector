#!/usr/bin/env python3
# catane_eval.py — citește /tmp/eval_rescore.csv (scoruri model curent pe 64 celule Catane) + adevărul lui Andrei
# (10 tumuli reali) și raportează: recall, marjă min-real, prag care ține toți realii, FP la acel prag, precizie.
# Operating point = pragul = scorul minim al unui real (ca să ținem recall 100%); FP = non-real >= acel prag.
import csv
TRUTH={11,17,18,30,43,45,50,55,57,64}
sc={int(r['idx']):float(r['score']) for r in csv.DictReader(open('/tmp/eval_rescore.csv'))}
real={i:sc[i] for i in TRUTH if i in sc}
fp_all={i:sc[i] for i in sc if i not in TRUTH}
min_real=min(real.values()); min_idx=min(real,key=real.get)
thr=min_real
fp=sorted([(i,s) for i,s in fp_all.items() if s>=thr], key=lambda x:-x[1])
print(f"recall: {len(real)}/{len(TRUTH)} reali scorați (toți prezenți în eval)")
print(f"marjă min-real: {min_real:.3f} (idx {min_idx})")
print(f"prag (= min real, ține recall 100%): {thr:.3f}")
print(f"FP la acest prag: {len(fp)}  -> {[(i,round(s,3)) for i,s in fp]}")
prec=len(real)/(len(real)+len(fp))
print(f"precizie @recall100: {100*prec:.0f}%  ({len(real)}/{len(real)+len(fp)})")
# gap = separare între cel mai slab real și cel mai tare FP (mare = robust)
top_fp=fp[0][1] if fp else 0.0
print(f"top FP: {top_fp:.3f} | gap (min_real - top_fp): {min_real-top_fp:+.3f}")
