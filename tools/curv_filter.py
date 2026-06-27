#!/usr/bin/env python3
# curv_filter.py DET_CSV OUT_CSV [GATE_JSON=<repo>/curv_gate.json] [THR=0.70]
# Filtru de CURBURA pt productie: ia detectii (CSV cu lon,lat[,...]), calculeaza trasaturile de forma v2 din
# elevatia LAKI3 (cache) si scorul gate-ului, si scrie OUT_CSV cu coloane pgate + keep (1 daca pgate>=THR).
# Taie fals-pozitivele compacte/aspre (mușuroaie naturale, arături) pastrand domurile netede (tumuli).
# Validat held-out Catane: 47->16 FP la recall 100% (prag 0.70 calibrat pe tumuli confirmati, min 0.713).
import sys,os,subprocess,csv
H=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DET=sys.argv[1];OUT=sys.argv[2]
GATE=sys.argv[3] if len(sys.argv)>3 else f'{H}/curv_gate.json'
THR=sys.argv[4] if len(sys.argv)>4 else '0.70'
PY=sys.executable
tmp=OUT+'.feat.tmp'
subprocess.run([PY,f'{H}/tools/curv_features3.py',DET,tmp],check=True)  # v3 scale-adaptiv (24.06): merge pe movile mari ȘI mici
subprocess.run([PY,f'{H}/tools/curv_gate.py','apply',GATE,tmp,OUT,THR],check=True)
n=sum(1 for _ in open(OUT))-1
kept=sum(1 for r in csv.DictReader(open(OUT)) if r.get('keep')=='1')
print(f"curv_filter: {n} detections -> {kept} kept / {n-kept} cut (threshold {THR})")
try:os.remove(tmp)
except:pass
