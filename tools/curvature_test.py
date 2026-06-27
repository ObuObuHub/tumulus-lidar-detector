#!/usr/bin/env python3
# curvature_test.py — TEST DISCRIMINABILITATE: separă CURBURA mușuroaiele compacte FP de tumulii reali?
# Pt fiecare coord (label,idx,lon,lat): calotă în jurul apexului -> 3 metrici + stampe hillshade|curbură.
#   convex = Laplacian mediu în calotă (dom convex-up => negativ)
#   asim   = simetrie radială (std azimutal / relief) — dom artificial = simetric (mic); natural = mare
#   rugoz  = std(Laplacian)/relief în calotă — tumul = neted (mic); mușuroi/natural = aspru (mare)
# Citește /tmp/curv_coords.csv -> /tmp/curv_out.csv + review/curvature_test.png. Fără scipy.
import os,sys,math,subprocess,csv
import numpy as np
from PIL import Image,ImageDraw,ImageFont
H=os.path.expanduser('~/lidar-match');CACHE="/tmp/laki3";CS=0.5;TPX=2000
APP="/Applications/QGIS-final-4_0_3.app/Contents"
ENV=dict(os.environ,DYLD_FRAMEWORK_PATH=f"{APP}/Frameworks",PROJ_DATA=f"{APP}/Resources/qgis/proj",PROJ_LIB=f"{APP}/Resources/qgis/proj",GDAL_DATA=f"{APP}/Resources/qgis/gdal")
GT=f"{APP}/MacOS/gdaltransform"
def trans(p,s,t):
    r=subprocess.run([GT,"-s_srs",s,"-t_srs",t],input=f"{p[0]} {p[1]}\n",capture_output=True,text=True,env=ENV);a,b,_=r.stdout.split();return float(a),float(b)
def hs(dem,cs,azs=(315,45,135,225,270,0),alt=35):
    gy,gx=np.gradient(dem,cs);sl=np.arctan(np.hypot(gx,gy));asp=np.arctan2(-gy,gx);o=np.zeros_like(dem);ar=math.radians(alt)
    for az in azs: azr=math.radians(360-az+90);o+=np.clip(np.sin(ar)*np.cos(sl)+np.cos(ar)*np.sin(sl)*np.cos(azr-asp),0,1)
    return o/len(azs)
def downs(a,fc):
    Hh,Ww=a.shape;return a[:Hh//fc*fc,:Ww//fc*fc].reshape(Hh//fc,fc,Ww//fc,fc).mean((1,3))
WIN=100.0;HALF=int(WIN/CS)
def local_window(est,nord):
    e0=int((est-WIN)//1000);e1=int((est+WIN)//1000);n0=int((nord-WIN)//1000);n1=int((nord+WIN)//1000)
    xll0=e0*1000;ytop0=(n1+1)*1000;W=(e1-e0+1)*TPX;Hh=(n1-n0+1)*TPX
    mos=np.full((Hh,W),np.nan,np.float32)
    for nk in range(n0,n1+1):
        for ek in range(e0,e1+1):
            p=f"{CACHE}/{nk}_{ek}.npy"
            if not os.path.exists(p):continue
            d=np.load(p);ox=int((ek*1000-xll0)/CS);oy=int((ytop0-(nk+1)*1000)/CS);mos[oy:oy+TPX,ox:ox+TPX]=d[:TPX,:TPX]
    px=int((est-xll0)/CS);py=int((ytop0-nord)/CS)
    return mos[py-HALF:py+HALF,px-HALF:px+HALF]
def analyze(est,nord):
    w=local_window(est,nord)
    if w.shape!=(2*HALF,2*HALF) or np.isnan(w).mean()>0.15: return None
    w=np.nan_to_num(w,nan=np.nanmedian(w))
    f=int(round(2.0/CS));z=downs(w,f);cs2=CS*f   # 2m DEM
    # snap apex = max în central 40m
    c0=z.shape[0]//2;rr=int(20/cs2);sub=z[c0-rr:c0+rr,c0-rr:c0+rr]
    off=np.unravel_index(np.argmax(sub),sub.shape);ay=c0-rr+off[0];ax=c0-rr+off[1]
    # Laplacian (curbură medie, m/m²)
    lap=(np.roll(z,1,0)+np.roll(z,-1,0)+np.roll(z,1,1)+np.roll(z,-1,1)-4*z)/(cs2*cs2)
    R=int(25/cs2);ys,xs=np.mgrid[0:z.shape[0],0:z.shape[1]];rad=np.hypot(ys-ay,xs-ax)
    cap=rad<=R
    relief=float(z[cap].max()-np.percentile(z[cap],10))+1e-6
    convex=float(lap[cap].mean())
    rugoz=float(lap[cap].std()/ (relief))
    # simetrie radială: pe inele, std azimutal al elevației / relief
    asim=[]
    for r0 in range(2,R,2):
        ring=(rad>=r0)&(rad<r0+2)&cap
        if ring.sum()>=6: asim.append(z[ring].std())
    asimv=float(np.mean(asim)/relief) if asim else 9.9
    # stampe pt montaj
    shp=hs(z,cs2);lo,hi=np.percentile(shp,2),np.percentile(shp,98)
    shimg=np.clip((shp-lo)/(hi-lo+1e-9)*255,0,255).astype('uint8')
    lo2,hi2=np.percentile(lap,3),np.percentile(lap,97);cur=np.clip((lap-lo2)/(hi2-lo2+1e-9)*255,0,255).astype('uint8')
    return dict(convex=convex,rugoz=rugoz,asim=asimv,relief=relief,sh=shimg,cur=cur)
rows=list(csv.DictReader(open(sys.argv[1] if len(sys.argv)>1 else '/tmp/curv_coords.csv')))
out=open('/tmp/curv_out.csv','w');wr=csv.writer(out);wr.writerow(['label','idx','convex','rugoz','asim','relief_m'])
items=[]
print(f"{'lbl':5}{'idx':>4}{'convex':>9}{'rugoz':>8}{'asim':>8}{'relief':>8}")
for r in rows:
    e,n=trans((float(r['lon']),float(r['lat'])),"EPSG:4326","EPSG:3844");a=analyze(e,n)
    if a is None: print(f"{r['label']:5}{r['idx']:>4}   NA");continue
    wr.writerow([r['label'],r['idx'],f"{a['convex']:.4f}",f"{a['rugoz']:.3f}",f"{a['asim']:.3f}",f"{a['relief']:.2f}"])
    print(f"{r['label']:5}{r['idx']:>4}{a['convex']:>9.4f}{a['rugoz']:>8.3f}{a['asim']:>8.3f}{a['relief']:>8.2f}")
    items.append((r['label'],r['idx'],a))
out.close()
# montaj: 2 coloane (hillshade|curbura) per caz, grupat FP apoi REAL
items.sort(key=lambda t:(t[0]!='FP',int(t[1])))
cellw=130;cellh=150;cols=4
img=Image.new('RGB',(cols*cellw*2,((len(items)+cols-1)//cols)*cellh+24),(12,12,12));dr=ImageDraw.Draw(img)
try: ft=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial Bold.ttf',13)
except: ft=ImageFont.load_default()
dr.text((6,5),"Stanga=hillshade  Dreapta=CURBURA(Laplacian). FP compacte vs REAL tumuli. Separa curbura?",fill=(255,230,90),font=ft)
for k,(lbl,idx,a) in enumerate(items):
    cx=(k%cols)*cellw*2;cy=(k//cols)*cellh+24
    sh=Image.fromarray(a['sh']).resize((120,120));cu=Image.fromarray(a['cur']).resize((120,120))
    img.paste(sh.convert('RGB'),(cx+4,cy+18));img.paste(cu.convert('RGB'),(cx+126,cy+18))
    col=(255,120,120) if lbl=='FP' else (120,255,120)
    dr.text((cx+4,cy+2),f"{lbl}#{idx} ru{a['rugoz']:.2f} as{a['asim']:.2f}",fill=col,font=ft)
img.save(f'{H}/review/curvature_test.png');print("-> review/curvature_test.png")
PY=0
