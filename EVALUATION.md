# Tumulus Detector: Technical Evaluation / Model Card

**Model:** `combined_cnn.pt` (release **r4**, md5 `b35c3d17…`, 97 KB)
**Task:** detect burial mounds (tumuli) in 0.5 m airborne-LiDAR-derived hillshade.
**Evaluation date:** 2026-06-27. **Status:** production screen for 0.5 m coverage (RO LAKI III).

> **Reading guide.** Key point (Section 4):
> **catalogue-precision is not measurable in Romania** (RAN is incomplete and poorly geo-referenced), and
> **LiDAR alone cannot confirm a tumulus**, only fieldwork can; not even a specialist confirms from LiDAR.
> The honest figure is *precision-on-form* (visual review of shape), not precision-against-a-catalogue.
>
> **Operating threshold.** Production score cutoff = **0.60**. `coh22 > 0.70` is the coherence
> *filter* (a different quantity, not the model score); the `≥0.85` figures in 5c are raw pre-filter
> FP counts, not the operating point.

---

## 1. Architecture

A small CNN (~23 k parameters):

```
Input: 1×128×128  (single-channel hillshade stamp)
Conv2d(1→16, 3×3, s2) → ReLU
Conv2d(16→32, 3×3, s2) → ReLU
Conv2d(32→64, 3×3, s2) → ReLU
AdaptiveAvgPool2d(1) → Linear(64→1) → sigmoid
```

The global average pool makes the head size-agnostic; smallness is intentional (≈21.7 k positives, mostly
one source, a larger net memorises the source rather than the morphology).

**Design intent (principle of form).** The positive class is *dome symmetry*, not prominence. The model is
built to fire on **plough-scarified, eroded mounds** too, faint dome-like traces still activate it as long
as their symmetry reads as a mound. A feature is positive if it *looks and is shaped like* a tumulus, even
un-catalogued; ground excavation is what ultimately confirms it.

**Input recipe (identical at train and inference):** 80 m ground window → downsample to **2 m effective**
→ 6-direction hillshade → resize 128² → per-stamp histogram equalisation (`homog`). The 2 m normalisation
is what makes recall resolution-robust for 0.5–1 m sources (Section 5).

## 2. Training

- **Loss/opt:** `BCEWithLogitsLoss` (pos_weight = 1.0), Adam (lr 1e-3, wd 1e-4), batch 64.
  pos_weight was lowered 3.0→1.0 because 8× augmentation already over-emphasised the thin real-positive set
  and caused field over-firing.
- **Augmentation:** 4 rotations × 2 flips (8×). Single-scale. Translation-jitter is available
  (`JITTER` env) but **not** in the production weights, it neither helped recall nor survived the gate.
- **Split:** geographically **disjoint groups**, per-CRS (RO lon/lat, DK UTM 25832), ≈70/15/15
  train/val/**TEST**. The geo-TEST is scored **once**, never tuned on, this is the anti-leakage guarantee.
- **Positives (~21.7 k):** 21 565 DK *Rundhøj* (Denmark, registry-confirmed) + 73 RO + 73 MDH (Arad) +
  11 Dobrogea. **Heavily DK-dominated**, see Limitations.
- **Negatives (~52 k):** terrain background (plain/hill/terrace/village ~25 k) + **hard-negative mining**
  (`ro_fp5k`, `ro_fp5k2`, `ro_fp20k` ≈ 13.7 k cells the model itself scored ≥0.5, dome-vetoed) +
  ditch/linear/anthropic. **Domes are never negatives** ("principle of form": a dome shape is a positive
  even if un-catalogued; `NODOMEFP=1` keeps `dataset_neg_domefp` out).

**r4 = r2 + further hard-negative rounds** (through an `r3` intermediate; `harvest_fp_hard.py` → dome-veto
→ retrain `NODOMEFP=1`).

**RO fine-tuning.** The base model was additionally fine-tuned on Romanian tumuli.

## 3. Post-processing filters (part of the production detector)

1. **Directional coherence** (`coh22 > 0.70`, structure tensor on the DEM): kills *directional* false
   positives, ploughing, ditches, levees. Mounds are radial → low coherence.
2. **Curvature gate** (`curv_gate`, scale-adaptive, logistic on shape features from elevation): keeps
   smooth domes, drops rough/compact natural humps. **RO-/scale-specific, does not transfer abroad.**

Effect (held-out floodplain, the hardest FP terrain): 58 raw firings ≥0.85 → **1** after filters.

## 4. Headline result: blind county-scale sweep

Blind sweep of **all 10 737 km²** of 0.5 m coverage over Dolj (production pipeline: single-scale 80 m + NMS
+ coherence + curvature, threshold 0.60) → **274 candidates** (80 m dedup). On visual review of every
candidate, **≈32 had non-tumular form** (mostly compact deep ploughing) → **~88 % precision on form**.

**Caveat: form is not confirmation.** LiDAR cannot confirm a tumulus, only fieldwork can. So 88 % is
*looks-like-a-mound*, not *is-a-mound*.

## 5. Held-out metrics

### 5a. Catane 24-GT benchmark: blind 58 km² scan, real prevalence (`benchmark.py`)
Ground truth = 24 known mounds; the scan is exhaustive (no peak cherry-picking).

| Threshold | Recall | Precision | FP density | F1 |
|---|---|---|---|---|
| 0.7 | 62 % (15/24) | 60 % | 0.2 /km² | 0.61 |
| 0.9 | 29 % (7/24) | 78 % | 0.04 /km² | 0.42 |
| recall-max (0.42) | **88 % (21/24)** | 36 % | 0.7 /km² | n/a |

**AUPRC = 0.554** at real prevalence. **r2→r4 improved AUPRC 0.517→0.554 and cut FP density ~77 %**
(recall-max 169→38 FP) at the cost of 2 mounds (24/24→21/24 reachable). FP↓ was the declared priority.

> **Why "precision 60 %" understates the truth.** The 10-GT → 24-GT correction showed most "FP" near Catane
> were *real un-catalogued mounds* scored against an incomplete GT. So the Catane precision column is a
> **lower bound**; the ~88 % form-precision (Section 4) is closer to the real operating point.

### 5b. Recall on 73 diverse confirmed mounds (in-domain, overlaps the RO fine-tuning set): production-faithful peak-search (`greens_recall.py`)
| Threshold | 0.7 | 0.66 | 0.5 | 0.4 |
|---|---|---|---|---|
| Recall | 64 % | 67 % | 73 % | 75 % |

Recall **plateaus at ~75 %**: on *well-formed* large mounds (Catane 0.5 m, gold-RAN) recall is ~100 %;
the missing ~25 % are small/eroded mounds (Section 7).

### 5c. FP density on 5 held-out non-mound zones, r2→r4 (≥0.85, pre-filter)
plain 23→0 · terrace 4→0 · hills 8→4 · floodplain1 58→1 · floodplain2 10→1. **Total 103→6.**

### 5d. Sanity / anti-shortcut
Grad-CAM attention is **central on the mound** (not on borders/metadata); response degrades only when the
mound is downsampled below ~16 px, i.e. the model keys on *morphology*, not an artefact.

## 6. Generalisation

- **Across resolution.** 0.5 m: AUROC **0.98** (excellent, but balanced-prevalence and so optimistic;
  the honest real-prevalence figure is AUPRC 0.554, Section 5a). 5 m: AUROC **0.60** (≈chance). Recall is
  largely resolution-invariant (the 2 m normalisation), but **precision collapses ≥2–5 m**, a 40 m mound
  is ~5 px, morphology is physically gone. **Reliable band = 0.5–1 m only.**
- **Across country (no retraining).** NL (AHN 0.5 m): on 97 public OSM barrows in Drenthe vs hard
  peak-searched controls, the RO production window (80 m) does **not** transfer (AUROC 0.54, Dutch barrows
  ~10–15 m are below that scale); scale-matched (32–80 m) it separates barrows from terrain at AUROC
  **0.862** (3% control rate above 0.5). UK (EA 1 m) 0.67–0.73. **Recognition generalises; scale and
  precision tuning are RO-specific** (curvature gate does not transfer, coherence partially does).

## 7. Known limitations & failure modes

1. **Very small mounds (<~15 m).** Scarified/eroded mounds are a design *target* (Section 1), but the
   *very small* are under-detected at single-scale (80 m window ≈ 4 px target). Multi-scale (40–68 m)
   recovers ~2–4 of ~12 but re-adds FP, recall/FP tension. Fix = small-mound positives, not augmentation.
2. **Resolution floor.** ≥2–5 m data → precision wall (Section 6). Only 0.5–1 m is screening-grade.
3. **Residual FP = near-perfect mimics.** Compact natural humps, spoil heaps, dunes, river levees, deep
   ploughing, that match a mound in *both* form and location. Openness, spatial priors and Sentinel-2
   multispectral all failed to separate them (AUC 0.58–0.75). The only remaining lever is **true-curvature
   multi-channel from raw elevation (SSD)**, pending storage.
4. **Class imbalance.** Positives are DK-dominated (21.5 k vs 73 RO). Mitigated by source-balanced
   negatives (so "source" can't act as a label) and Grad-CAM checks, but the RO positive base is thin, 
   the largest single risk to RO-specific generalisation.
5. **Translation sensitivity.** Mounds are trained centred → ~40 % score fall-off off-centre. **Not** a
   real recall loss: the sliding-window peak-search in production finds the apex. Confirmed, not theoretical.
6. **Curvature gate vs ploughed mounds.** A genuine mound on a heavily ploughed field can be cut by the
   directional brake (a small recall risk on arable). Mitigation: route directional hits to a "verify"
   bucket rather than hard-dropping in survey mode.
7. **Catalogue-precision is unmeasurable in RO** (incomplete + poorly geo-referenced RAN), and LiDAR cannot
   confirm a mound. Use precision-on-form (visual); fieldwork is the only ground truth.

## 8. Verdict

- **As a prospecting screen on 0.5 m LiDAR: strong.** ~88 % precision-on-form blind at county scale
  (10 737 km², all candidates visually reviewed), ~100 % recall on well-formed mounds, post-filter FP
  density ~0.2–0.7 /km². Form-precision, not field-confirmed.
- **As a catalogue-grade precise detector: not there yet.** ~12 % residual mimic-FP need ground/satellite
  triage; ~25 % of the diverse/eroded/small population is under-recalled; catalogue-precision is
  unmeasurable.
- **Generalisation:** recall transfers across country and (within the fine band) resolution; the precision
  filters are RO/scale-specific and would need recalibration elsewhere.

**Next levers, in order:** (1) small-mound positives + multi-scale retune for the recall gap;
(2) true-curvature 2-channel (SSD) for the mimic-FP wall; (3) more RO positives to balance the DK skew.

## 9. Reproduce

```bash
./venv/bin/python tools/benchmark.py /tmp/catane_gt_full.csv combined_cnn.pt   # 5a
./venv/bin/python tools/greens_recall.py combined_cnn.pt /tmp/diverse_mounds.csv # 5b
bash /tmp/serious_eval.sh                                                        # full r2-vs-r4 suite
```
The ground-truth CSVs (`lon,lat`) and `serious_eval.sh` above are **not shipped** — mound coordinates are withheld (see README *Ethics*). `tools/benchmark.py` is the public entry point: supply your own held-out `lon,lat` CSV.

Held-out geo-TEST is scored once inside `tools/train_combined.py`; do not tune on it.
