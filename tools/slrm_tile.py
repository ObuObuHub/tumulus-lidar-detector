import sys,math,subprocess,os
import numpy as np
from PIL import Image,ImageFilter
LON=float(sys.argv[1]); LAT=float(sys.argv[2]); OUT=sys.argv[3]
# auto-pick service by extent (finer res first): (name,org,level,res,(minlon,minlat,maxlon,maxlat))
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
    if a<=LON<=c and b<=LAT<=d: svc=s; break
if not svc: print('NO COVERAGE at',LON,LAT); sys.exit(1)
SVC,ORG,Z,RES,_=svc
WIN=1000; R=6378137.0;C=2*math.pi*R;ORIG=-20037508.342787;ORIGY=20037508.342787
res=C/(256*2**Z); xC=R*math.radians(LON); yC=R*math.log(math.tan(math.pi/4+math.radians(LAT)/2))
x0=(xC-ORIG)/res-WIN//2; y0=(ORIGY-yC)/res-WIN//2
cv=Image.new('L',(WIN,WIN),0); os.makedirs('/tmp/hegyi_tiles',exist_ok=True)
for col in range(int(x0//256),int((x0+WIN)//256)+1):
    for row in range(int(y0//256),int((y0+WIN)//256)+1):
        fn=f'/tmp/hegyi_tiles/{SVC}_{Z}_{col}_{row}.png'
        if not os.path.exists(fn):
            subprocess.run(['curl','-s','--max-time','25','-o',fn,f'https://tiles.arcgis.com/tiles/{ORG}/arcgis/rest/services/{SVC}/MapServer/tile/{Z}/{row}/{col}'],check=False)
        try: cv.paste(Image.open(fn).convert('L'),(col*256-int(x0),row*256-int(y0)))
        except: pass
a=np.asarray(cv,dtype=np.float32); b=np.asarray(cv.filter(ImageFilter.GaussianBlur(28)),dtype=np.float32)
rel=a-b; lo,hi=np.percentile(rel,2),np.percentile(rel,98)
Image.fromarray(np.clip((rel-lo)/(hi-lo+1e-6)*255,0,255).astype('uint8')).save(OUT)
print(f'SLRM saved: {OUT} | service {SVC} ({RES}m) | center {LON},{LAT}')
