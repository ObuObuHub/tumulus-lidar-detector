#!/usr/bin/env python3
# sweep_table.py [CSV=review/sweep_dolj_final.csv] [TAG=dolj] — copie a tabelului de candidați cu 2 coloane de
# linkuri 1-click: (1) LiDAR = ArcGIS Map Viewer pe webmap-ul RO-LiDAR (Hegyi) centrat+pin pe punct (hillshade
# 0.5m; ~30s încărcare); (2) Satelit = Google Maps imagine satelitară + pin. Iese HTML (deschis în browser,
# linkuri clicabile) + CSV (import în Sheets, linkuri clicabile). Tag descoperire vs lângă-movilă-cunoscută.
import os,sys,csv,html
H=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSVf=sys.argv[1] if len(sys.argv)>1 else f"{H}/review/sweep_dolj_final.csv"
TAG=sys.argv[2] if len(sys.argv)>2 else 'dolj'
WEBMAP="de24b3ffeeda4209b2229c2a902b6312"  # webmap RO-LiDAR (Hegyi), public
def lidar_url(lon,lat):
    return f"https://www.arcgis.com/apps/mapviewer/index.html?webmap={WEBMAP}&center={lon},{lat}&level=18&marker={lon},{lat}"
def sat_url(lon,lat):
    return f"https://www.google.com/maps/place/{lat},{lon}/@{lat},{lon},450m/data=!3m1!1e3"
rows=list(csv.DictReader(open(CSVf)));rows.sort(key=lambda r:-float(r['score']))
disc=set()
dp=f"{H}/review/sweep_{TAG}_discoveries.csv"
if os.path.exists(dp):
    for r in csv.DictReader(open(dp)):disc.add((round(float(r['lon']),6),round(float(r['lat']),6)))
def istip(r):return 'descoperire' if (round(float(r['lon']),6),round(float(r['lat']),6)) in disc else 'langa-cunoscuta'
# CSV
OUTC=f"{H}/review/sweep_{TAG}_table.csv"
with open(OUTC,'w',newline='') as fo:
    w=csv.writer(fo);w.writerow(['idx','lat','lon','score','coh','pgate','tip','lidar_url','satelit_url'])
    for i,r in enumerate(rows,1):
        lo,la=r['lon'],r['lat'];w.writerow([i,la,lo,r['score'],r['coh'],r['pgate'],istip(r),lidar_url(lo,la),sat_url(lo,la)])
print(f"-> {OUTC} ({len(rows)} rânduri)")
# HTML
nd=sum(1 for r in rows if istip(r)=='descoperire')
parts=["""<!doctype html><html lang=ro><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>Sweep Dolj — candidați movile</title><style>
body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;background:#14161a;color:#e8e8ea;margin:0;padding:16px}
h1{font-size:18px;margin:0 0 4px}.sub{color:#9aa;font-size:13px;margin:0 0 14px}
table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:6px 9px;border-bottom:1px solid #2a2d33;text-align:left;white-space:nowrap}
th{position:sticky;top:0;background:#1d2026;cursor:pointer;user-select:none}tr:hover{background:#1b1e24}
.disc{color:#ffb070}.kn{color:#7fd99a}
a.btn{display:inline-block;padding:3px 9px;border-radius:5px;text-decoration:none;font-weight:600;font-size:12px}
a.lid{background:#2b6cb0;color:#fff}a.sat{background:#2f855a;color:#fff}a.btn:hover{opacity:.85}
.sc{font-weight:700}</style></head><body>
<h1>Sweep Dolj 0.5m — candidați movile</h1>"""]
parts.append(f'<p class=sub>{len(rows)} candidați (sortați după scor) · {nd} descoperiri (portocaliu) · {len(rows)-nd} lângă movilă cunoscută (verde). LiDAR = hillshade 0.5m RO-LiDAR (Hegyi), pin pe punct, ~30s încărcare. Satelit = Google. Click pe antet = sortează.</p>')
parts.append('<table id=t><thead><tr><th onclick="s(0,1)">#</th><th onclick="s(3,1)">scor</th><th onclick="s(4,1)">coh</th><th onclick="s(5,1)">pgate</th><th onclick="s(6,0)">tip</th><th onclick="s(1,0)">lat</th><th onclick="s(2,0)">lon</th><th>LiDAR</th><th>Satelit</th></tr></thead><tbody>')
for i,r in enumerate(rows,1):
    lo,la=r['lon'],r['lat'];tip=istip(r);cls='disc' if tip=='descoperire' else 'kn'
    parts.append(f'<tr><td>{i}</td><td class=sc>{r["score"]}</td><td>{r["coh"]}</td><td>{r["pgate"]}</td><td class={cls}>{"●" if tip=="descoperire" else "○"} {tip}</td><td>{la}</td><td>{lo}</td>'
        f'<td><a class="btn lid" href="{html.escape(lidar_url(lo,la))}" target=_blank>LiDAR</a></td>'
        f'<td><a class="btn sat" href="{html.escape(sat_url(lo,la))}" target=_blank>Satelit</a></td></tr>')
parts.append("""</tbody></table>
<script>function s(c,num){const tb=document.querySelector('#t tbody');const rs=[...tb.rows];
rs.sort((a,b)=>{let x=a.cells[c].innerText,y=b.cells[c].innerText;if(num){return parseFloat(y)-parseFloat(x)||0}return x<y?-1:x>y?1:0});
rs.forEach(r=>tb.appendChild(r));}</script></body></html>""")
OUTH=f"{H}/review/sweep_{TAG}_table.html"
open(OUTH,'w').write("".join(parts));print(f"-> {OUTH}")
