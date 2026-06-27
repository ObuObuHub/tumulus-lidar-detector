import math,subprocess,os,csv
from PIL import Image
R=6378137.0; C=2*math.pi*R; ORIG=-20037508.342787; ORIGY=20037508.342787
REPO=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); OUT=f"{REPO}/vest_pilot"
BASEDIR=f"{OUT}/base"; TILEDIR="/tmp/hegyi_tiles"
os.makedirs(BASEDIR,exist_ok=True); os.makedirs(TILEDIR,exist_ok=True)

# service: (name, org, native_level, res_label) ; extents lon/lat (minlon,minlat,maxlon,maxlat). Prefer finer res first.
SERVICES=[
 ("CS_917","wCvLzGFkz06gCfBg",17,"0.5m",(21.23,44.52,22.79,45.71)),
 ("MH","wCvLzGFkz06gCfBg",17,"0.5m",(21.73,43.91,23.67,45.29)),
 ("DJ","wCvLzGFkz06gCfBg",17,"0.5m",(22.77,43.66,24.28,44.75)),
 ("GJ_917","wCvLzGFkz06gCfBg",17,"0.5m",(22.52,44.53,23.88,45.38)),
 ("AR_MDH_tif","Q2Kmg0bQDn3rySgn",17,"1m",(20.67,45.86,22.77,46.70)),
 ("BH_MDH_tif","Q2Kmg0bQDn3rySgn",17,"1m",(21.37,46.36,22.83,47.61)),
 ("HD_MDH_tif","Q2Kmg0bQDn3rySgn",17,"1m",(22.32,45.23,23.60,46.37)),
 ("AB_MDH_tif","Q2Kmg0bQDn3rySgn",17,"1m",(22.66,45.44,23.82,46.59)),
 ("Banat_3_5_H_tif","Q2Kmg0bQDn3rySgn",16,"3m",(20.03,44.27,23.06,46.34)),
]
def pick(lon,lat):
    for s in SERVICES:
        mnlo,mnla,mxlo,mxla=s[4]
        if mnlo<=lon<=mxlo and mnla<=lat<=mxla: return s
    return None
def merc(lon,lat): return R*math.radians(lon), R*math.log(math.tan(math.pi/4+math.radians(lat)/2))
_cache={}
def tile(svc,org,z,col,row):
    k=(svc,z,col,row)
    if k in _cache: return _cache[k]
    fn=f"{TILEDIR}/{svc}_{z}_{col}_{row}.png"
    if not os.path.exists(fn):
        subprocess.run(["curl","-s","--max-time","25","-o",fn,
            f"https://tiles.arcgis.com/tiles/{org}/arcgis/rest/services/{svc}/MapServer/tile/{z}/{row}/{col}"],check=False)
    try: im=Image.open(fn).convert('L')
    except Exception: im=None
    _cache[k]=im; return im

WIN=384; HALF=WIN//2
rows=list(csv.DictReader(open(f"{REPO}/vest_tumuli.csv")))
def slug(s): return ''.join(c if c.isalnum() else '_' for c in (s or 'NA')).strip('_')[:20]
man=[]; nocov=0
for r in rows:
    lon,lat=float(r['lon']),float(r['lat'])
    s=pick(lon,lat)
    if not s: nocov+=1; continue
    svc,org,z,res,_=s
    x,y=merc(lon,lat); resm=C/(256*2**z)
    gx=(x-ORIG)/resm; gy=(ORIGY-y)/resm
    x0=gx-HALF; y0=gy-HALF
    canvas=Image.new('L',(WIN,WIN),0)
    c0=int(math.floor(x0/256)); c1=int(math.floor((x0+WIN)/256))
    r0=int(math.floor(y0/256)); r1=int(math.floor((y0+WIN)/256))
    for col in range(c0,c1+1):
        for rw in range(r0,r1+1):
            t=tile(svc,org,z,col,rw)
            if t is None: continue
            canvas.paste(t,(col*256-int(x0), rw*256-int(y0)))
    name=f"{r['jud'][:3]}_{r['oid']}_{slug(r['loc'])}.png"
    canvas.save(f"{BASEDIR}/{name}")
    man.append({'oid':r['oid'],'jud':r['jud'],'loc':r['loc'],'nume':r['nume'],'lon':lon,'lat':lat,
                'service':svc,'res':res,'base':f"base/{name}"})
with open(f"{OUT}/manifest_base.csv","w",newline='') as fh:
    w=csv.DictWriter(fh,fieldnames=list(man[0].keys())); w.writeheader(); w.writerows(man)
print("western base stamps:",len(man),"| no coverage:",nocov,"| unique tiles:",len(_cache))
from collections import Counter
print("by service:",Counter((m['service'],m['res']) for m in man))
