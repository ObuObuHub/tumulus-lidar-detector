#!/usr/bin/env python3
# grid_slrm.py CLON CLAT OUTDIR [N=3]
# Generează o grilă NxN de plăci SLRM curate (auto-pick serviciu Hegyi) în jurul unui punct.
# Fiecare placă: 1000px, etichetă "#idx lon,lat" în colț, manifest.csv pt extragere.
import sys,math,subprocess,os
import numpy as np
from PIL import Image,ImageFilter,ImageDraw,ImageFont
CLON=float(sys.argv[1]); CLAT=float(sys.argv[2]); OUTDIR=sys.argv[3]
N=int(sys.argv[4]) if len(sys.argv)>4 else 3
R=6378137.0;C=2*math.pi*R;ORIG=-20037508.342787;ORIGY=20037508.342787
SERVICES=[
 ("CS_917","wCvLzGFkz06gCfBg",17,0.5,(21.23,44.52,22.79,45.71)),
 ("MH","wCvLzGFkz06gCfBg",17,0.5,(21.73,43.91,23.67,45.29)),
 ("DJ","wCvLzGFkz06gCfBg",17,0.5,(22.77,43.66,24.28,44.75)),
 ("GJ_917","wCvLzGFkz06gCfBg",17,0.5,(22.52,44.53,23.88,45.38)),
 ("AR_MDH_tif","Q2Kmg0bQDn3rySgn",17,1,(20.67,45.86,22.77,46.70)),
 ("BH_MDH_tif","Q2Kmg0bQDn3rySgn",17,1,(21.37,46.36,22.83,47.61)),
 ("HD_MDH_tif","Q2Kmg0bQDn3rySgn",17,1,(22.32,45.23,23.60,46.37)),
 ("AB_MDH_tif","Q2Kmg0bQDn3rySgn",17,1,(22.66,45.44,23.82,46.59)),
 ("Banat_3_5_H_tif","Q2Kmg0bQDn3rySgn",17,3,(20.03,44.27,23.06,46.34)),
]
svc=None
for s in SERVICES:
    a,b,c,d=s[4]
    if a<=CLON<=c and b<=CLAT<=d: svc=s; break
if not svc: print('NO COVERAGE at',CLON,CLAT); sys.exit(1)
SVC,ORG,Z,RES,_=svc
WIN=1000; res=C/(256*2**Z)
mlon=(WIN*res)/(111320*math.cos(math.radians(CLAT))); mlat=(WIN*res)/110540
step_lon=mlon*0.92; step_lat=mlat*0.92  # ~8% overlap
os.makedirs(OUTDIR,exist_ok=True); os.makedirs('/tmp/hegyi_tiles',exist_ok=True)
try: fnt=ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf",26)
except: fnt=ImageFont.load_default()
def gettile(col,row):
    fn=f'/tmp/hegyi_tiles/{SVC}_{Z}_{col}_{row}.png'
    if not os.path.exists(fn):
        subprocess.run(['curl','-s','--max-time','25','-o',fn,f'https://tiles.arcgis.com/tiles/{ORG}/arcgis/rest/services/{SVC}/MapServer/tile/{Z}/{row}/{col}'],check=False)
    try: return Image.open(fn).convert('L')
    except: return None
man=open(f'{OUTDIR}/manifest.csv','w'); man.write('file,lon,lat,win,service\n')
half=N//2; idx=0
for dlat in range(half,-half-1,-1):       # north -> south
    for dlon in range(-half,half+1):       # west -> east
        idx+=1
        lon=CLON+dlon*step_lon; lat=CLAT+dlat*step_lat
        xC=R*math.radians(lon); yC=R*math.log(math.tan(math.pi/4+math.radians(lat)/2))
        x0=(xC-ORIG)/res-WIN//2; y0=(ORIGY-yC)/res-WIN//2
        cv=Image.new('L',(WIN,WIN),0)
        for col in range(int(x0//256),int((x0+WIN)//256)+1):
            for row in range(int(y0//256),int((y0+WIN)//256)+1):
                t=gettile(col,row)
                if t: cv.paste(t,(col*256-int(x0),row*256-int(y0)))
        a=np.asarray(cv,dtype=np.float32); b=np.asarray(cv.filter(ImageFilter.GaussianBlur(28)),dtype=np.float32)
        rel=a-b; lo,hi=np.percentile(rel,2),np.percentile(rel,98)
        sl=Image.fromarray(np.clip((rel-lo)/(hi-lo+1e-6)*255,0,255).astype('uint8')).convert('RGB')
        dr=ImageDraw.Draw(sl)
        dr.rectangle([0,0,330,34],fill=(0,0,0)); dr.text((5,3),f"#{idx}  {lon:.4f},{lat:.4f}",fill=(0,255,255),font=fnt)
        fname=f'{OUTDIR}/g{idx}_{lon:.4f}_{lat:.4f}.png'
        sl.save(fname); man.write(f'{fname},{lon:.6f},{lat:.6f},{WIN},{SVC}\n')
        print("saved",fname)
man.close(); print(f'{idx} plăci | serviciu {SVC} ({RES}m) | manifest {OUTDIR}/manifest.csv')
