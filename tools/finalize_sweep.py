#!/usr/bin/env python3
# finalize_sweep.py TAG — din /tmp/sweep_<TAG>_state.json: dedup candidati (<80m, pastreaza scor max),
# scrie CSV final (lon,lat,score,coh,pgate), un subset "descoperiri" (candidati la >120m de orice movila
# cunoscuta: labels.csv mound + catane_gt_full + confirmed/possible + ran_bulk tumuli) si o harta de ansamblu.
# Coloanele CSV raman DOAR lon,lat,score,coh,pgate (cf. Andrei: fara distanta-RAN, fara judet).
import os,sys,json,csv,math,subprocess
import numpy as np
from PIL import Image,ImageDraw,ImageFont
H=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAG=sys.argv[1] if len(sys.argv)>1 else 'dolj'
st=json.load(open(f"/tmp/sweep_{TAG}_state.json"))
cands=st["cands"]
import pyproj
_TF={}
def _tf(s,t):
    if (s,t) not in _TF:_TF[(s,t)]=pyproj.Transformer.from_crs(s,t,always_xy=True)
    return _TF[(s,t)]
def trans(pts,s,t):
    if not pts:return []
    tf=_tf(s,t);return [tuple(tf.transform(a,b)) for a,b in pts]
# dedup in metri (proiectie) la 80m
ll=[(c[0],c[1]) for c in cands]
en=trans(ll,"EPSG:4326","EPSG:3844") if ll else []
order=sorted(range(len(cands)),key=lambda i:-cands[i][2])
keep=[];kept_en=[]
for i in order:
    e,n=en[i]
    if any((e-ke)**2+(n-kn)**2<80*80 for ke,kn in kept_en):continue
    keep.append(cands[i]);kept_en.append((e,n))
print(f"{len(cands)} raw -> {len(keep)} after dedup 80m")
OUT=f"{H}/review/sweep_{TAG}_final.csv"
with open(OUT,'w',newline='') as fo:
    w=csv.writer(fo);w.writerow(['lon','lat','score','coh','pgate'])
    for c in sorted(keep,key=lambda r:(-r[2])):w.writerow(c)
print(f"-> {OUT}")
# movile cunoscute (pt MARCARE descoperiri, NU coloana in CSV)
kll=[]
for pth in [f'{H}/labeled/labels.csv',f'{H}/labeled/confirmed_new_positives_20260625.csv',f'{H}/labeled/possible_mounds_20260626.csv','/tmp/catane_gt_full.csv']:
    if os.path.exists(pth):
        for r in csv.DictReader(open(pth)):
            if 'verdict' in r and r['verdict'] not in ('mound',''):continue
            try:kll.append((float(r['lon']),float(r['lat'])))
            except:pass
rb='/tmp/ran_bulk.csv'
if os.path.exists(rb):
    rd=csv.reader(open(rb),delimiter='|');hdr=next(rd)
    def idx(name):
        for i,h in enumerate(hdr):
            if h.strip()==name:return i
        return -1
    iln=idx('longitudine (zecimală)');ilt=idx('latitudine (zecimală)');itip=idx('tipul sitului')
    for row in rd:
        if itip>=0 and 'tumul' not in row[itip].lower():continue
        try:
            lo=float(row[iln].replace(',','.'));la=float(row[ilt].replace(',','.'))
            if 20<lo<30 and 43<la<48:kll.append((lo,la))
        except:pass
ken=np.array(trans(kll,"EPSG:4326","EPSG:3844")) if kll else np.empty((0,2))
print(f"known mounds (to flag discoveries): {len(kll)}")
disc=[]
for c,(e,n) in zip([k for k in sorted(keep,key=lambda r:-r[2])],[trans([(k[0],k[1])],"EPSG:4326","EPSG:3844")[0] for k in sorted(keep,key=lambda r:-r[2])]):
    if ken.shape[0]==0 or np.min((ken[:,0]-e)**2+(ken[:,1]-n)**2)>=120*120:disc.append(c)
OUTD=f"{H}/review/sweep_{TAG}_discoveries.csv"
with open(OUTD,'w',newline='') as fo:
    w=csv.writer(fo);w.writerow(['lon','lat','score','coh','pgate'])
    for c in disc:w.writerow(c)
print(f"-> {OUTD} ({len(disc)} discoveries / {len(keep)} candidates)")
# harta ansamblu
E0,E1,N0,N1=st["params"]["bbox"]
lons=[c[0] for c in keep];lats=[c[1] for c in keep]
W=1400;asp=(N1-N0)/max(E1-E0,1);Hh=int(W*asp)
img=Image.new('RGB',(W,Hh+44),(16,18,22));dr=ImageDraw.Draw(img)
def _font(sz):
    for p in ('/System/Library/Fonts/Supplemental/Arial Bold.ttf','/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf','DejaVuSans-Bold.ttf'):
        try:return ImageFont.truetype(p,sz)
        except:continue
    return ImageFont.load_default()
ft=_font(15);ftb=_font(20)
# proiecteaza lon/lat candidati in pixeli prin EPSG:3844 bbox
def xy(e,n):
    x=int((e-E0*1000)/((E1-E0)*1000)*W);y=44+int((N1*1000-n)/((N1-N0)*1000)*Hh);return x,y
discset=set((round(c[0],6),round(c[1],6)) for c in disc)
ken_all=[k for k in keep]
en_all=trans([(c[0],c[1]) for c in ken_all],"EPSG:4326","EPSG:3844") if ken_all else []
for c,(e,n) in zip(ken_all,en_all):
    x,y=xy(e,n)
    isd=(round(c[0],6),round(c[1],6)) in discset
    col=(255,120,80) if isd else (90,200,120)
    r=3 if c[2]>=0.85 else 2
    dr.ellipse([x-r,y-r,x+r,y+r],fill=col)
dr.rectangle([0,0,W,42],fill=(10,10,12))
dr.text((8,4),f"SWEEP {TAG.upper()} 0.5m — {len(keep)} candidates ({st['km2']:.0f}km2, {len(st['done'])}/{st['blocks_total']} blocks)",fill=(255,255,255),font=ftb)
dr.text((8,24),f"green = near a known mound   orange = possible discovery ({len(disc)})",fill=(180,220,180),font=ft)
OUTM=f"{H}/review/sweep_{TAG}_map.jpg"
img.save(OUTM,quality=88);print(f"-> {OUTM}")
