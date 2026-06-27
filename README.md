# Tumulus LiDAR Detector

A small CNN that detects **burial mounds (tumuli)** in 0.5 m airborne-LiDAR-derived hillshade,
for Romania's national LiDAR programme (ANCPI **LAKI III**).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ObuObuHub/tumulus-lidar-detector/blob/main/demo.ipynb) — **run the detector on a sample area in your browser, no install** (Runtime → Run all).

Its design goal is the tumuli that are easiest to overlook: mounds heavily flattened by decades of
ploughing, surviving only as faint, low-relief domes. Rather than keying on height or prominence, it
responds to dome **symmetry**, so a scarified mound still registers as long as its form reads as a tumulus.
On a blind sweep of an entire county (10,737 km² of 0.5 m LiDAR), ~88 % of flagged candidates had tumular
form on review, with ~100 % recall on well-formed mounds. Form is not confirmation: only fieldwork settles
a tumulus. Full numbers and limits: **[EVALUATION.md](EVALUATION.md)**.

## What's here

| File | What |
|---|---|
| `combined_cnn.pt` | Production model weights (release r4, ~23 k params) |
| `curv_gate.json` | Curvature post-filter (logistic, shape features) |
| `EVALUATION.md` | Technical evaluation / model card |
| `MODEL.md` | Architecture & training notes |
| `tools/` | Inference, sweep, evaluation and visualisation scripts |

## How it works

Each 80 m ground window is downsampled to **2 m effective**, rendered as a 6-direction hillshade,
resized to 128², histogram-equalised, and scored by a 3-conv CNN with a global-average-pool head
(size-agnostic). Detections pass two post-filters: **directional coherence** (kills ploughing /
ditches) and a **scale-adaptive curvature gate** (keeps smooth domes, drops rough natural humps).
The 2 m normalisation is what makes recall robust across 0.5–1 m sources and even other countries
(NL/UK generalise without retraining); precision filters are Romania/scale-specific.

## Usage

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
# survey a zone (centre lon, lat, box side in km):
./venv/bin/python tools/scan_zone.py 23.522 44.043 8
# blind county sweep (resumable, disk-managed):
TAG=dolj SWEEP_BBOX="322,429,233,336" ./venv/bin/python tools/sweep_05m.py
# evaluate against your own ground-truth CSV (columns lon,lat; not shipped — see Ethics):
./venv/bin/python tools/benchmark.py ground_truth.csv combined_cnn.pt
```
The scripts read 0.5 m DTM tiles from the public LAKI III service and cache them under `/tmp/laki3`.
Coordinate transforms use `pyproj` (EPSG:4326 ↔ EPSG:3844); no system GDAL/QGIS required.

## Ethics

Coordinates of detected mounds are withheld to avoid facilitating looting; this repo ships the model and method, not site locations.

## Data & credits

- **DTM:** ANCPI, *LAKI II / LAKI III* national LiDAR (Romania).
- **Training positives:** Denmark *Fund og Fortidsminder* (Rundhøj, registry-public) + Romanian
  field-confirmed mounds.
- Author: **Chiper-Leferman Andrei** (ObuObuHub).

Special thanks to **Dr. Alexandru Hegyi** and **Dr. Mehdi Nourelahi** for their guidance and advice.

## License

- **Code** (`tools/`): **MIT**, see [LICENSE](LICENSE).
- **Model weights, evaluation & docs:** **CC-BY-4.0**, free to use, share and adapt, including
  commercially, with attribution to Chiper-Leferman Andrei.

---

**Chiper-Leferman Andrei:** concept, model, data curation, review.
**Claude Opus 4.8:** implementation, training, tooling.
