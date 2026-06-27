# Model: detector tumuli LiDAR (notă scurtă, v1 baseline)

## Sarcină
Clasificare binară pe stampe: **movilă (tumul) vs non-movilă**. Primul pas spre un detector (fereastră glisantă), baseline „underfit by design", de la simplu la complex.

## Input
- Stampă **128×128 px, 1 canal (grayscale)**.
- Acoperă **80 m teren**, adusă la **rezoluție efectivă 2 m comună** (downsample), ca toate sursele să arate la fel.
- Vizualizare: **hillshade multidirecțional** (DTM: calculat din înălțime, 6 direcții; MDH-Hegyi: gata calculat) + autocontrast (percentile 2-98).
- O movilă centrată per stampă (pozitivii = doar movile IZOLATE, fără vecin <50 m).

## Arhitectură (`tools/train_combined.py`, clasa Net)
CNN mic:
- Conv 1→16, stride 2, ReLU
- Conv 16→32, stride 2, ReLU
- Conv 32→64, stride 2, ReLU
- AdaptiveAvgPool → vector 64
- Linear 64→1 (logit)
~ câteva zeci de mii de parametri. Intenționat mic (puțini pozitivi → model mic ca să nu memoreze).

## Date & surse (combinat-echilibrat, 3 surse + hard-neg antropic)
- **Pozitivi (386, 343 izolați):** DTM 0,5 m (Dolj/Mehedinți, RAN) + MDH 1 m (Arad, harvest om-în-buclă) + **DK 0,4 m (282 Rundhøj, transfer internațional)**. Doar izolați.
- **Negative (6699):** 5000 random (3000 DTM + 2000 MDH) + 500 DK + **1200 hard-negatives ANTROPICE** (`dataset_neg_anthro/`, FP reale culese pe sate/canale Dolj, §7.bis PROCEDURA). Homogenizate identic. Fiecare sursă are ȘI pozitivi ȘI negativi → **sursa decorelată de etichetă** (modelul nu poate folosi sursa ca scurtătură; învață forma + robustețe la randare).

## Antrenare
- **Split pe GEOGRAFIE (~3 km) ÎNAINTE de augmentare** (anti-leakage: rotațiile unei movile nu trec în validare).
- **Augmentare ×8** (4 rotații × 2 oglindiri) DOAR pe train.
- Negative subeșantionate ~3:1 față de pozitivii augmentați.
- Loss: **BCEWithLogitsLoss, pos_weight=3** (pozitivii rari).
- Optimizator: Adam, lr 1e-3, weight_decay 1e-4. 40 epoci, batch 64.
- GPU: Apple MPS (M4). Mediu: `venv` (torch 2.12).

## Rezultate
- **v1 baseline:** ~47 pozitivi izolați, recall ~5/6, FP ~24%. Învăța „rotund/umbră", nu domul specific.
- **v2 (3 surse + DK):** 343 izolați, 266 train / 77 val. Recall 69/77 = **90%**, FP 10,8%. **Transfer DK→RO confirmat** (DK 90%, Dolj 100%).
- **v3 (+ hard-neg antropic, model curent `combined_cnn.pt`):** recall 88% (Dolj 100%); **aprinderi pe teren antropic proaspăt −78% la prag 0,6** (A/B Calafat, `tools/sweep_eval.py`). Backups: `combined_anthro.pt` (=curent), `combined_noanthro.pt` (v2 reprodus). Antrenarea e deterministă (seed fix).

## Roadmap (simplu → complex)
1. ✅ **Hard-negative mining** (generic + antropic pe teren real): FP scade mult, inclusiv pe sate/canale.
1b. **Filtru de mărime/formă în detector** (post-procesare): resping mari/alungite (sate/canale); praguri din atributele Diameter/Højde ale registrului DK Rundhøj.
2. **Volum pozitivi**: active learning (modelul propune, omul validează).
3. **Multi-bandă** (pantă/openness din DTM brut), precizie mai mare.
4. **Faza detector**: fereastră glisantă pe plăci; clusterele (cimitire înșirate) intră aici, nu în baseline.
5. **Validare riguroasă** (metodologia Sobotkova: puncte→grilă→TP/FN), pe zonă ținută deoparte.

---

## EVOLUȚIE MODEL → PRODUCȚIE r4 (la 26.06.2026; detaliu metodic = PROCEDURA.md §17-22)
Arhitectura a rămas identică (CNN-ul mic de mai sus); toate câștigurile = DATE (pozitivi + hard-negative) + protocol. Lanțul (arhivă `combined_*.pt`):
- **v3 `combined_anthro.pt`** → scarificați/buclă-țintită → **`combined_loop.pt`** (91% Catane held-out) → cull DK + balansare sursă.
- **25.06, hard-negative mining ITERATIV (2 runde) + DOME-VETO CENTRAL:** mine FP-ul REAL al modelului (celule scor ≥0.5), NU negative pe formă (alea-s EASY, CRESC FP). R1 `combined_fphard.pt` → R2 **`combined_fphard_r2.pt`**. Catane 24-GT: FP @recall92% 384→240→115 (−70%), AUPRC 0.41→0.52.
- **26.06, r3 EȘEC apoi r4 SUCCES:** r3 (79 FP marcate manual) = MAI PROST (volum prea mic → stocastic domină). r4 (+4393 hard-neg MINATE autonom cu `harvest_fp_hard.py`, dome-veto) = **SUCCES**.

### 🟢 PRODUCȚIE CURENTĂ = `combined_cnn.pt` = `combined_fphard_r4.pt` (md5 b35c3d17)
- **Metrici (serious_eval, r2→r4):** FP held-out 5 terenuri ≥0.85: 103→6. Recall 73 movile diverse: ținut/mai bun. Catane prag-corect: AUPRC 0.517→0.554; recall-max 23/24·169FP → 21/24·**38FP (−77%)**. Grad-CAM = atenție centrală pe movilă.
- **⚠ r4 scorează ~0.2 MAI JOS (conservator).** PRAG DE OPERARE recalibrat: candidat **0.60** (nu 0.85), env `CANDTHR` în `scan_zone.py`; movile Dolj cunoscute scorează 0.79/0.95. Filtrele coerență(coh22>0.70)+curbură(curv_gate 0.70) = MODEL-INDEPENDENTE (rămân).
- **Backup / revert:** `combined_prod_pre_r3.pt` = r2 (md5 1ec37724). Revert = `cp combined_prod_pre_r3.pt combined_cnn.pt`.
- **Limită cunoscută:** movile MICI (~15m, teren rugos/pădure) = ratate la scara 80m (nepotrivire de SCARĂ, nu formă); multi-scară (`scan_zone_ms.py`) recuperează parțial (2-4/12) dar readuce FP. Fix real = pozitivi de movile mici în antrenament.
- **Generalizare rezoluție:** 0.5m = perfect; 1-2m = modelul gata DAR cere DTM FLOAT (avem doar hillshade randat); 5m = OOD (zid fizic).

### _NEGBASE actual (train_combined.py)
13 dir-uri negative incl. hard-neg cumulativ: `dataset_neg_ro_fp5k`(R1)+`dataset_neg_ro_fp5k2`(R2)+`dataset_neg_expert_fp`(79 marcate/luncă)+`dataset_neg_ro_fp20k`(4393 mined). Env retrain: `NODOMEFP=1` (scoate `dataset_neg_domefp`, domurile NU-s negative, principiul de formă), `OUT=`, `HARDPOS_FILE`+`HARDPOS_REP`.
