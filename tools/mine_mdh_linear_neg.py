#!/usr/bin/env python3
# mine_mdh_linear_neg.py CLON CLAT KM [N] [--montaj] — minează negative LINIARE (șanț/drum/limită parcelă) din
# hillshade-ul MDH (pt zone fără elevație 0.5m, ex. Arad). Criteriu de FORMĂ = coerență direcțională mare =
# trăsătură liniară = NICIODATĂ tumul (compact). SIGUR (nu pe scorul modelului = fără otravă). Exclude movile known.
# Scop: modelul a învățat „muchie de șanț = pozitiv" de la Rundhoj-urile DK cu șanț inelar -> aceste negative îl dezvață.
import sys,os,math,subprocess,csv,glob
import numpy as np
from PIL import Image
H=os.path.expanduser('~/lidar-match')
CLON=float(sys.argv[1]);CLAT=float(sys.argv[2]);KM=float(sys.argv[3]) if len(sys.argv)>3 else 6.0
NMAX=int(sys.argv[4]) if len(sys.argv)>4 and not sys.argv[4].startswith('--') else 3000
MONTAJ='--montaj' in sys.argv
COH_T=float(os.environ.get('COH_T',0.55)); ENERGY_MIN=float(os.environ.get('ENERGY_MIN',3.0))   # coerență = direcție dominantă (liniar); energie min = evită câmp plat. Override env pt șanțuri mici (COH_T~0.42, ENERGY_MIN~1.3)
R=6378137.0;ORIG=-20037508.342787;ORIGY=20037508.342787;Z=18;MPP=2*math.pi*R/(256*2**Z)
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
def stamp(lo,la,meters=80,eff=2.0,out=128):
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
    if a.std()<0.5: return None
    lo2,hi2=np.percentile(a,2),np.percentile(a,98);a=np.clip((a-lo2)/(hi2-lo2+1e-6),0,1)
    f=max(1,int(round(eff/MPP)));Hh,Ww=a.shape;a=a[:Hh//f*f,:Ww//f*f].reshape(Hh//f,f,Ww//f,f).mean((1,3))
    return np.asarray(Image.fromarray((a*255).astype('uint8')).resize((out,out)),np.uint8)
def coherence(st):
    a=np.asarray(Image.fromarray(st).filter(__import__('PIL.ImageFilter',fromlist=['GaussianBlur']).GaussianBlur(1.0)),np.float32)
    gy,gx=np.gradient(a)
    Jxx=(gx*gx).mean();Jyy=(gy*gy).mean();Jxy=(gx*gy).mean()
    coh=math.sqrt((Jxx-Jyy)**2+4*Jxy*Jxy)/(Jxx+Jyy+1e-9)
    energy=float(np.hypot(gx,gy).mean())
    return coh,energy
def in_mdh(lo,la): return any(a<=lo<=c and b<=la<=d for a,b,c,d in [m[1] for m in MDH])
known=[(float(r['lon']),float(r['lat'])) for r in csv.DictReader(open(f'{H}/labeled/labels.csv')) if r.get('verdict')=='mound' and in_mdh(float(r['lon']),float(r['lat']))]
def near(lo,la,d=150): return any(((lo-a)*111000*math.cos(math.radians(la)))**2+((la-b)*111000)**2<d*d for a,b in known)
OUT=f'{H}/dataset_neg_mdh_linear';os.makedirs(OUT,exist_ok=True)
step_m=50;dlat=step_m/111000;dlon=step_m/(111000*math.cos(math.radians(CLAT)))
nlat=int(KM*1000/2/111000/dlat);nlon=int(KM*1000/2/(111000*math.cos(math.radians(CLAT)))/dlon)
n0=len(glob.glob(f'{OUT}/mdhlin_*.png'));n=0;samples=[]
print(f"minez negative LINIARE MDH la {CLAT},{CLON} ({KM}km), coh>{COH_T}, țintă {NMAX}...",flush=True)
for i in range(-nlat,nlat+1):
    if n>=NMAX: break
    for j in range(-nlon,nlon+1):
        if n>=NMAX: break
        la=CLAT+i*dlat;lo=CLON+j*dlon
        if near(lo,la): continue
        st=stamp(lo,la)
        if st is None: continue
        coh,en=coherence(st)
        if coh<COH_T or en<ENERGY_MIN: continue
        Image.fromarray(st).save(f"{OUT}/mdhlin_{n0+n:05d}.png");n+=1
        if MONTAJ and len(samples)<36: samples.append((st,coh,en))
print(f"-> {n} negative liniare în {os.path.basename(OUT)}/ (total {n0+n})",flush=True)
if MONTAJ and samples:
    from PIL import ImageDraw,ImageFont
    C=6;rw=(len(samples)+C-1)//C;cell=128
    img=Image.new('RGB',(C*cell,rw*cell+22),(15,15,15));dr=ImageDraw.Draw(img)
    try: ft=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial Bold.ttf',12)
    except: ft=ImageFont.load_default()
    dr.text((4,4),f"VERIFICARE negative liniare ({len(samples)}) — toate trebuie să fie linii/șanțuri, NICIO movilă rotundă",fill=(255,230,90),font=ft)
    for k,(st,coh,en) in enumerate(samples):
        x=(k%C)*cell;y=(k//C)*cell+22
        img.paste(Image.fromarray(st).convert('RGB').resize((cell-4,cell-4)),(x+2,y+2))
        dr.text((x+3,y+2),f"c{coh:.2f}",fill=(120,255,120),font=ft)
    img.save(f"{H}/review/mdh_linear_neg_check.png");print(f"-> review/mdh_linear_neg_check.png")
