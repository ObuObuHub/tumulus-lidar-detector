#!/usr/bin/env python3
"""Build assets/laki3_coverage.geojson — the footprint of the 0.5 m LiDAR (LAKI III MNT)
tiles the demo can actually scan.

Source of truth: ANCPI Descarcare/Laki3MNT tile grid. Each tile is a 1 km square named
"{north_km}_{east_km}" in EPSG:3844 (Stereo70) — the exact key scan_zone.py downloads from
https://geoportal.ancpi.ro/laki3_mnt/zip/{north}_{east}.zip. We fetch the tile names only
(cheap), rebuild each 1 km square locally, dissolve them into a handful of rectangles, and
reproject the corners to WGS84. Re-run when ANCPI extends the LAKI III coverage.

Usage:  python tools/build_coverage.py
"""
import json, os, sys, time, urllib.request, urllib.parse
from collections import defaultdict

LYR = ("https://geoportal.ancpi.ro/hosted_services/rest/services/"
       "Descarcare/Laki3MNT/MapServer/0/query")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "assets", "laki3_coverage.geojson")


def _get(params):
    url = LYR + "?" + urllib.parse.urlencode(params)
    last = None
    for _ in range(4):
        try:
            with urllib.request.urlopen(url, timeout=45) as r:
                return json.load(r)
        except Exception as e:  # noqa: BLE001 — transient network, retry
            last = e
            time.sleep(2)
    raise last


def fetch_tile_names():
    names, offset = [], 0
    while True:
        d = _get({"where": "1=1", "outFields": "NUME", "returnGeometry": "false",
                  "resultOffset": offset, "resultRecordCount": 1000, "f": "json"})
        fs = d.get("features", [])
        if not fs:
            break
        names += [f["attributes"]["NUME"] for f in fs]
        offset += len(fs)
        if not d.get("exceededTransferLimit") and len(fs) < 1000:
            break
    return names


def dissolve(cells):
    """Set of (north_km, east_km) 1 km cells -> list of (n_lo, n_hi, e_lo, e_hi) rectangles
    (inclusive km indices). Horizontal run-length per row, then greedy vertical merge."""
    rows = defaultdict(list)
    for nk, ek in cells:
        rows[nk].append(ek)
    runs_by_row = {}
    for nk, eks in rows.items():
        eks.sort()
        runs, s, p = [], eks[0], eks[0]
        for e in eks[1:]:
            if e == p + 1:
                p = e
            else:
                runs.append((s, p)); s = p = e
        runs.append((s, p))
        runs_by_row[nk] = runs
    open_rects, last_row, out = {}, {}, []
    for nk in sorted(runs_by_row):
        cur = set(runs_by_row[nk])
        for run in list(open_rects):
            if run not in cur or last_row[run] != nk - 1:
                out.append((open_rects.pop(run), last_row.pop(run), run[0], run[1]))
        for run in cur:
            if run in open_rects:
                last_row[run] = nk
            else:
                open_rects[run] = nk; last_row[run] = nk
    for run, n_lo in open_rects.items():
        out.append((n_lo, last_row[run], run[0], run[1]))
    return out


def main():
    import pyproj
    print("Fetching LAKI III MNT tile names from ANCPI ...")
    names = fetch_tile_names()
    cells = set()
    for nm in names:
        try:
            a, b = nm.split("_"); cells.add((int(a), int(b)))
        except ValueError:
            pass
    print(f"  {len(cells)} tiles (1 km each)")
    rects = dissolve(cells)
    print(f"  dissolved into {len(rects)} rectangles")

    tf = pyproj.Transformer.from_crs("EPSG:3844", "EPSG:4326", always_xy=True)

    def edge(x0, y0, x1, y1, n=6):
        return [tf.transform(x0 + (x1 - x0) * i / n, y0 + (y1 - y0) * i / n) for i in range(n)]

    feats = []
    for n_lo, n_hi, e_lo, e_hi in rects:
        X0, X1, Y0, Y1 = e_lo * 1000, (e_hi + 1) * 1000, n_lo * 1000, (n_hi + 1) * 1000
        ring = (edge(X0, Y0, X1, Y0) + edge(X1, Y0, X1, Y1)
                + edge(X1, Y1, X0, Y1) + edge(X0, Y1, X0, Y0))
        ring.append(ring[0])
        feats.append({"type": "Feature", "properties": {},
                      "geometry": {"type": "Polygon",
                                   "coordinates": [[[round(lo, 5), round(la, 5)] for lo, la in ring]]}})

    fc = {"type": "FeatureCollection",
          "properties": {"source": "ANCPI LAKI III MNT 0.5 m tile grid (Descarcare/Laki3MNT)",
                         "resolution_m": 0.5, "tiles_1km": len(cells),
                         "note": "Area scannable by this demo (0.5 m DTM). Re-run tools/build_coverage.py to refresh."},
          "features": feats}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(fc, f)
    print(f"Wrote {OUT}  ({round(os.path.getsize(OUT)/1024, 1)} KB, {len(feats)} features)")


if __name__ == "__main__":
    sys.exit(main())
