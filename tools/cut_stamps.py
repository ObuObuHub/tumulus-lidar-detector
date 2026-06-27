#!/usr/bin/env python3
# cut_stamps.py [OUTPX=128] [METERS=120]
# Taie din labels.csv o stampa UNIFORMA per movila: fereastra fixa in METRI (scara reala egala),
# SLRM, redimensionata la OUTPX x OUTPX. 1 movila centrata. Serviciu auto-pick dupa coord.
# Iese dataset_v2/<source>/*.png + contact_<source>.png + manifest.csv
import sys,math,subprocess,os,csv
import numpy as np
from PIL import Image,ImageFilter,ImageDraw,ImageFont
OUTPX=int(sys.argv[1]) if len(sys.argv)>1 else 128
METERS=float(sys.argv[2]) if len(sys.argv)>2 else 120.0
R=6378137.0;C=2*math.pi*R;ORIG=-20037508.342787;ORIGY=20037508.342787
SERVICES=[
 ("CS_917","wCvLzGFkz06gCfBg",17,0.5,(21.23,44.52,22.79,45.71)),("MH","wCvLzGFkz06gCfBg",17,0.5,(21.73,43.91,23.67,45.29)),
 ("DJ","wCvLzGFkz06gCfBg",17,0.5,(22.77,43.66,24.28,44.75)),("GJ_917","wCvLzGFkz06gCfBg",17,0.5,(22.52,44.53,23.88,45.38)),
 ("AR_MDH_tif","Q2Kmg0bQDn3rySgn",17,1,(20.67,45.86,22.77,46.70)),("BH_MDH_tif","Q2Kmg0bQDn3rySgn",17,1,(21.37,46.36,22.83,47.61)),
 ("HD_MDH_tif","Q2Kmg0bQDn3rySgn",17,1,(22.32,45.23,23.60,46.37)),("AB_MDH_tif","Q2Kmg0bQDn3rySgn",17,1,(22.66,45.44,23.82,46.59)),
 ("Banat_3_5_H_tif","Q2Kmg0bQDn3rySgn",17,3,(20.03,44.27,23.06,46.34))]
def pick(lon,lat):
    for s in SERVICES:
        a,b,c,d=s[4]
        if a<=lon<=c and b<=lat<=d: return s
    return None
_c={}
def tile(svc,org,z,col,row):
    k=(svc,z,col,row)
    if k in _c: return _c[k]
    fn=f'/tmp/hegyi_tiles/{svc}_{z}_{col}_{row}.png'
    if not os.path.exists(fn):
        subprocess.run(['curl','-s','--max-time','25','-o',fn,f'https://tiles.arcgis.com/tiles/{org}/arcgis/rest/services/{svc}/MapServer/tile/{z}/{row}/{col}'],check=False)
    try: im=Image.open(fn).convert('L')
    except: im=None
    _c[k]=im; return im
def stamp(lon,lat):
    s=pick(lon,lat)
    if not s: return None,None
    SVC,ORG,Z,RES,_=s
    res=C/(256*2**Z); mpp=res
    half_m=METERS/2.0; margin=90.0  # m extra pt context blur SLRM
    halfpx=int((half_m+margin)/mpp); Wc=2*halfpx
    x=R*math.radians(lon); y=R*math.log(math.tan(math.pi/4+math.radians(lat)/2))
    x0=(x-ORIG)/res-halfpx; y0=(ORIGY-y)/res-halfpx
    cv=Image.new('L',(Wc,Wc),0)
    for col in range(int(x0//256),int((x0+Wc)//256)+1):
        for row in range(int(y0//256),int((y0+Wc)//256)+1):
            t=tile(SVC,ORG,Z,col,row)
            if t: cv.paste(t,(col*256-int(x0),row*256-int(y0)))
    a=np.asarray(cv,dtype=np.float32); b=np.asarray(cv.filter(ImageFilter.GaussianBlur(24)),dtype=np.float32)
    rel=a-b; lo,hi=np.percentile(rel,2),np.percentile(rel,98)
    sl=Image.fromarray(np.clip((rel-lo)/(hi-lo+1e-6)*255,0,255).astype('uint8'))
    cropr=int(half_m/mpp)  # central METERS window
    cc=Wc//2; sl=sl.crop((cc-cropr,cc-cropr,cc+cropr,cc+cropr)).resize((OUTPX,OUTPX))
    return sl,RES
H=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); OUT=f'{H}/dataset_v2'
rows=list(csv.DictReader(open(f'{H}/labeled/labels.csv')))
from collections import defaultdict
bag=defaultdict(list); man=open(f'{OUT}/manifest.csv','w') if os.path.isdir(OUT) else None
os.makedirs(OUT,exist_ok=True); man=open(f'{OUT}/manifest.csv','w'); man.write('file,lon,lat,source,res_m\n')
miss=0
for i,r in enumerate(rows):
    grp=r['source']; st,res=stamp(float(r['lon']),float(r['lat']))
    if st is None: miss+=1; continue
    os.makedirs(f'{OUT}/{grp}',exist_ok=True)
    fn=f'{OUT}/{grp}/{grp}_{i:03d}_{res}m.png'; st.save(fn)
    bag[grp].append((st,res)); man.write(f'{fn},{r["lon"]},{r["lat"]},{grp},{res}\n')
man.close()
try: fnt=ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf",13)
except: fnt=ImageFont.load_default()
for grp,items in bag.items():
    cols=12; cell=OUTPX+4; rn=math.ceil(len(items)/cols)
    sh=Image.new('RGB',(cols*cell,rn*cell+18),(20,20,20)); dr=ImageDraw.Draw(sh)
    dr.text((4,2),f"{grp}: {len(items)} stampe {OUTPX}px = {METERS:.0f} m",fill=(0,255,255),font=fnt)
    for i,(st,res) in enumerate(items):
        x=(i%cols)*cell+2; y=(i//cols)*cell+16; sh.paste(st.convert('RGB'),(x,y))
        dr.text((x+1,y+1),f"{res}",fill=(255,200,0),font=fnt)
    sh.save(f'{OUT}/contact_{grp}.png')
print(f'stampe {OUTPX}px @ {METERS:.0f}m | per grup:',{k:len(v) for k,v in bag.items()},'| fara acoperire:',miss)
print('->',OUT)
