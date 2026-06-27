#!/usr/bin/env python3
# context_satellite.py CSV [WIN_M] [OUT] — pentru fiecare candidat din CSV (coloane est,nord,idx,score; EPSG:3844),
# montează LADO: context LiDAR (hillshade nativ 0.5m, fereastră WIN_M, reper central) | DREAPTA: satelit Esri World
# Imagery (aceeași fereastră, centrată pe același punct). Ca Andrei să judece movila în context + comparat din satelit.
# -> review/context_satellite.png + coords/linkuri la stdout.
import os,sys,math,subprocess,csv
import numpy as np
from PIL import Image,ImageDraw,ImageFont
H=os.path.expanduser('~/lidar-match');CACHE="/tmp/laki3";CS=0.5;TPX=2000
CSV=sys.argv[1];WIN=float(sys.argv[2]) if len(sys.argv)>2 else 320.0
OUT=sys.argv[3] if len(sys.argv)>3 else f'{H}/review/context_satellite.png'
APP="/Applications/QGIS-final-4_0_3.app/Contents"
ENV=dict(os.environ,DYLD_FRAMEWORK_PATH=f"{APP}/Frameworks",PROJ_DATA=f"{APP}/Resources/qgis/proj",PROJ_LIB=f"{APP}/Resources/qgis/proj",GDAL_DATA=f"{APP}/Resources/qgis/gdal")
GT=f"{APP}/MacOS/gdaltransform"
def trans(p,s,t):
    r=subprocess.run([GT,"-s_srs",s,"-t_srs",t],input=f"{p[0]} {p[1]}\n",capture_output=True,text=True,env=ENV);a,b,_=r.stdout.split();return float(a),float(b)
def dl(nk,ek):
    p=f"{CACHE}/{nk}_{ek}.npy";return np.load(p) if os.path.exists(p) else None
def hs(dem,cs,azs=(315,45,135,225,270,0),alt=35):
    gy,gx=np.gradient(dem,cs);sl=np.arctan(np.hypot(gx,gy));asp=np.arctan2(-gy,gx);o=np.zeros_like(dem);ar=math.radians(alt)
    for az in azs: azr=math.radians(360-az+90);o+=np.clip(np.sin(ar)*np.cos(sl)+np.cos(ar)*np.sin(sl)*np.cos(azr-asp),0,1)
    return o/len(azs)
rows=list(csv.DictReader(open(CSV)))
cand=[]
for r in rows:
    e=float(r['est']);n=float(r['nord']);lo,la=trans((e,n),"EPSG:3844","EPSG:4326")
    cand.append({'idx':r.get('idx','?'),'score':r.get('score','?'),'e':e,'n':n,'lon':lo,'lat':la})
print(f"{len(cand)} candidați, fereastră {WIN}m",flush=True)
# --- LiDAR mosaic acoperind toți ---
es=[c['e'] for c in cand];ns=[c['n'] for c in cand]
e0=int((min(es)-WIN)//1000);e1=int((max(es)+WIN)//1000);n0=int((min(ns)-WIN)//1000);n1=int((max(ns)+WIN)//1000)
xll0=e0*1000;ytop0=(n1+1)*1000;W=(e1-e0+1)*TPX;Hh=(n1-n0+1)*TPX
mos=np.full((Hh,W),np.nan,np.float32)
for nk in range(n0,n1+1):
    for ek in range(e0,e1+1):
        d=dl(nk,ek)
        if d is None: continue
        ox=int((ek*1000-xll0)/CS);oy=int((ytop0-(nk+1)*1000)/CS);mos[oy:oy+TPX,ox:ox+TPX]=d[:TPX,:TPX]
def lidar_crop(e,n,win):
    px=int((e-xll0)/CS);py=int((ytop0-n)/CS);hw=int(win/2/CS)
    w=mos[py-hw:py+hw,px-hw:px+hw]
    if w.shape[0]<10 or w.shape[1]<10 or np.isnan(w).mean()>0.3: return None
    h=hs(np.nan_to_num(w,nan=np.nanmedian(w)),CS);lo,hi=np.percentile(h,2),np.percentile(h,98)
    return np.clip((h-lo)/(hi-lo+1e-6),0,1)
# --- satelit Esri World Imagery (web mercator z19) ---
Z=18;TS=256;Rm=6378137.0;Cm=2*math.pi*Rm
def deg2px(lon,lat,z):
    n=2**z;x=(lon+180)/360*n*TS;y=(1-math.log(math.tan(math.radians(lat))+1/math.cos(math.radians(lat)))/math.pi)/2*n*TS;return x,y
mpp_sat=Cm/(TS*2**Z)*math.cos(math.radians(cand[0]['lat']))  # ~0.3m/px la z19
SDIR="/tmp/esri_sat";os.makedirs(SDIR,exist_ok=True)
def sat_tile(z,col,row):
    fn=f"{SDIR}/{z}_{col}_{row}.jpg"
    if not os.path.exists(fn): subprocess.run(["curl","-s","--max-time","25","-o",fn,f"https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{row}/{col}"],check=False)
    try: return Image.open(fn).convert('RGB')
    except: return None
def sat_crop(lon,lat,win):
    cx,cy=deg2px(lon,lat,Z);hw=win/2/mpp_sat
    x0=cx-hw;y0=cy-hw;ww=int(2*hw)
    im=Image.new('RGB',(ww,ww),(20,20,20))
    for col in range(int(x0//TS),int((x0+ww)//TS)+1):
        for row in range(int(y0//TS),int((y0+ww)//TS)+1):
            t=sat_tile(Z,col,row)
            if t: im.paste(t,(col*TS-int(x0),row*TS-int(y0)))
    return im
# --- montaj ---
CW=300;PAD=8;HDR=22;ncol=2
img=Image.new('RGB',(2*CW+3*PAD, len(cand)*(CW+HDR)+PAD),(12,12,12));dr=ImageDraw.Draw(img)
try: ft=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial Bold.ttf',15)
except: ft=ImageFont.load_default()
for i,c in enumerate(cand):
    y=i*(CW+HDR)+HDR
    dr.text((PAD,y-18),f"#{c['idx']} scor {c['score']} | {c['lat']:.5f},{c['lon']:.5f}  (stg=LiDAR, dr=satelit)",fill=(255,255,90),font=ft)
    lc=lidar_crop(c['e'],c['n'],WIN)
    if lc is not None:
        l=Image.fromarray((lc*255).astype('uint8')).convert('RGB').resize((CW,CW));dd=ImageDraw.Draw(l)
        dd.line([(CW//2-12,CW//2),(CW//2+12,CW//2)],fill=(255,60,60),width=2);dd.line([(CW//2,CW//2-12),(CW//2,CW//2+12)],fill=(255,60,60),width=2)
        img.paste(l,(PAD,y))
    sc=sat_crop(c['lon'],c['lat'],WIN).resize((CW,CW));ds=ImageDraw.Draw(sc)
    ds.line([(CW//2-12,CW//2),(CW//2+12,CW//2)],fill=(255,60,60),width=2);ds.line([(CW//2,CW//2-12),(CW//2,CW//2+12)],fill=(255,60,60),width=2)
    img.paste(sc,(2*PAD+CW,y))
    print(f"  #{c['idx']} {c['lat']:.5f},{c['lon']:.5f}  https://maps.google.com/?q={c['lat']:.5f},{c['lon']:.5f}")
img.save(OUT);print(f"-> {OUT}")
