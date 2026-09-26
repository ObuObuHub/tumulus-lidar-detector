# tumulus-lidar-detector

Detection of plain-field burial mounds (kurgans) in 0.5–1 m LiDAR. A small CNN (~23k parameters)
learns dome shape from Danish mounds trained jointly with Romanian ones (cross-country morphology
transfer). A matched filter proposes candidates; the CNN and a morphometric shape
signature are fused noisy-OR into a single score; shape filters cut the mimics. Designed for mounds
of ~20–40 m diameter. Built for prospection — confirmation remains field survey.

**[STUDY.md](STUDY.md)** — method, evaluation, cross-country transfer, limitations.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ObuObuHub/tumulus-lidar-detector/blob/main/demo.ipynb)
**Run it in your browser, no install**: Runtime → Run all, pick a point on the map (0.5 m coverage:
Oltenia / south-west Romania) and scan.

![The detection chain](assets/figs/fig1_pipeline.png)

## Quick start

```
pip install -r requirements.txt
python tools/tumul_scan.py 23.513 23.537 44.038 44.056 candidates.csv
```

Scans a lon/lat box (W E S N) and writes ranked candidates (lon, lat, fused score, per-signal
columns). Tiles download on demand; missing tiles are reported, and no data stops with an error. The
box above is in Dolj and works while ANCPI is offline. Thresholds and filters: environment
variables in `tools/tumul_scan.py`.

> ⚠ ANCPI's geoportal is temporarily offline after a cyberattack. Until it returns, scanning works across
> Dolj county, served from the project's tile mirror
> ([`tumulus-lidar-tiles`](https://github.com/ObuObuHub/tumulus-lidar-tiles); data © ANCPI, redistributed
> unmodified, with attribution).

## Results

| Benchmark | Result |
|---|---|
| Official register (eISM), 48 unseen tumuli | 31% recovered (50% with visible relief) |
| Catane, development area, 26 labels | 21/26 at the operating point |
| Dolj county scan | 635 of 1,198 candidates mound-shaped on visual review; 93.5% >100 m from any registered site |
| Foreign kurgans, Poland, 1 m | AUROC 0.71; 27% of catalogue kurgans |

Details, caveats and negative results: [STUDY.md](STUDY.md).

## Verify with your own ground truth

```
python tools/benchmark.py your_gt.csv combined_cnn.pt
```

Benchmarks the single-channel core (`combined_cnn.pt`) against your ground-truth CSV at real
prevalence: AUPRC, recall, false positives. Missing tiles are reported.

## Repository layout

- `tools/tumul_scan.py` — production scanner (matched filter → CNN + shape signature → noisy-OR fusion → mimic filters)
- `tools/scan_zone_v4.py` — the engine behind the Colab demo (same chain)
- `tools/lib_tumul.py`, `tools/lib_channels.py` — shared library (footprint, channels)
- `tools/benchmark.py` — benchmark of the single-channel core at real prevalence, user-supplied ground truth
- `assets/` — fusion formula, mimic-filter models, morphometric fingerprint, demo map, figures
- `multichannel_cnn.pt` (production chain), `combined_cnn.pt` (single-channel core) — trained weights

## Ethics

Coordinates of detected mounds are withheld to avoid facilitating looting; this repository ships the
model and the method, not site locations.

## Data & credits

- **DTM:** ANCPI, *LAKI II / LAKI III* national LiDAR (Romania). Test LiDAR: Environment Agency (UK), PDOK AHN (NL), GUGiK (PL).
- **Training positives:** Denmark *Fund og Fortidsminder* (Rundhøj, registry-public) + Romanian register (RAN) and visually reviewed mounds.
- Author: **Chiper-Leferman Andrei** (ObuObuHub).

Special thanks to **Dr. Alexandru Hegyi** and **Dr. Mehdi Nourelahi** for their guidance and advice.

## License

- **Code** (`tools/`): **MIT**, see [LICENSE](LICENSE).
- **Model weights, docs & figures:** **CC-BY-4.0** — free to use, share and adapt, including
  commercially, with attribution to Chiper-Leferman Andrei.
