#!/usr/bin/env python3
# mound_context.py CSV [METERS] — pt fiecare coord din CSV (idx,lon,lat,...) face un crop CONTEXT la rezoluție
# NATIVĂ MDH (0.6m/px), lat (default 240m), cu reper central, ca Andrei să judece movilă-vs-șanț cu context.
# Montaj -> review/mound_context.png. Mult mai clar decât stampa de model (80m@2m).
import sys,os,math,subprocess,csv
import numpy as np
from PIL import Image,ImageDraw,ImageFont
H=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV=sys.argv[1];METERS=float(sys.argv[2]) if len(sys.argv)>2 else 240.0
R=6378137.0;ORIG=-20037508.342787;ORIGY=20037508.342787;Z=18;MPP=2*math.pi*R/(256*2**Z)  # 0.6m/px
MDH=[("AR_MDH_tif",(20.67,45.86,22.77,46.70)),("BH_MDH_tif",(21.37,46.36,22.83,47.61)),("HD_MDH_tif",(22.32,45.23,23.60,46.37)),("AB_MDH_tif",(22.66,45.44,23.82,46.59))]
ORG="Q2Kmg0bQDn3rySgn";TDIR="/tmp/mdh_tiles_L18";os.makedirs(TDIR,exist_ok=True)
def pick(lo,la):
    for svc,(a,b,c,d) in MDH:
        if a<=lo<=c and b<=la<=d: return svc
def tile(svc,col,row):
    fn=f"{TDIR}/{svc}_{col}_{row}.png"
    if not os.path.exists(fn): subprocess.run(["curl","-s","--max-time","25","-o",fn,f"https://tiles.arcgis.com/tiles/{ORG}/arcgis/rest/services/{svc}/MapServer/tile/{Z}/{row}/{col}"],check=False)
    try: return Image.open(fn).convert('L')
    except: return None
def crop(lo,la,meters):
    svc=pick(lo,la)
    if not svc: return None
    half=meters/2/MPP;x=R*math.radians(lo);y=R*math.log(math.tan(math.pi/4+math.radians(la)/2))
    px=(x-ORIG)/MPP;py=(ORIGY-y)/MPP;x0=px-half;y0=py-half;W=int(2*half);cv=Image.new('L',(W,W),0);ok=False
    for col in range(int(x0//256),int((x0+W)//256)+1):
        for row in range(int(y0//256),int((y0+W)//256)+1):
            t=tile(svc,col,row)
            if t: cv.paste(t,(col*256-int(x0),row*256-int(y0)));ok=True
    if not ok: return None
    a=np.asarray(cv,np.float32)
    lo2,hi2=np.percentile(a,1),np.percentile(a,99)
    return np.clip((a-lo2)/(hi2-lo2+1e-6)*255,0,255).astype(np.uint8)
rows=list(csv.DictReader(open(CSV)))
disp=int(METERS/MPP)  # px nativi
cell=min(disp,300)
C=3;rw=(len(rows)+C-1)//C;hh=30
img=Image.new('RGB',(C*(cell+6)+6,hh+rw*(cell+22)),(12,12,12));dr=ImageDraw.Draw(img)
try: ft=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial Bold.ttf',16)
except: ft=ImageFont.load_default()
dr.text((6,6),f"CONTEXT nativ {MPP:.1f}m/px, {METERS:.0f}m latime, reper central=movila etichetata. Movila (rotund) sau sant (liniar)?",fill=(255,230,90),font=ft)
for k,r in enumerate(rows):
    lo,la=float(r['lon']),float(r['lat'])
    c=crop(lo,la,METERS)
    if c is None: continue
    im=Image.fromarray(c).convert('RGB').resize((cell,cell))
    d2=ImageDraw.Draw(im);cc=cell//2
    d2.line([(cc-9,cc),(cc-3,cc)],fill=(255,60,60),width=2);d2.line([(cc+3,cc),(cc+9,cc)],fill=(255,60,60),width=2)
    d2.line([(cc,cc-9),(cc,cc-3)],fill=(255,60,60),width=2);d2.line([(cc,cc+3),(cc,cc+9)],fill=(255,60,60),width=2)
    x=(k%C)*(cell+6)+6;y=hh+(k//C)*(cell+22)
    lbl=f"#{r.get('idx',k+1)}"
    if r.get('scor_v0'): lbl+=f" v0:{r['scor_v0']}"
    if r.get('scor_v2'): lbl+=f" v2:{r['scor_v2']}"
    if r.get('coh_liniaritate'): lbl+=f" coh:{r['coh_liniaritate']}"
    dr.text((x,y),lbl,fill=(120,230,255),font=ft);img.paste(im,(x,y+18))
img.save(f"{H}/review/mound_context.png");print(f"-> review/mound_context.png ({len(rows)} crop-uri context {METERS:.0f}m @ {MPP:.2f}m/px)")
