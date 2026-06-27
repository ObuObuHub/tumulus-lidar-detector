import math,subprocess,os,csv
import numpy as np
from PIL import Image,ImageFilter,ImageDraw
R=6378137.0;C=2*math.pi*R;ORIG=-20037508.342787;ORIGY=20037508.342787
SERVICES=[("CS_917","wCvLzGFkz06gCfBg",17,0.5,(21.23,44.52,22.79,45.71)),("MH","wCvLzGFkz06gCfBg",17,0.5,(21.73,43.91,23.67,45.29)),
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
def slrm_stamp(lon,lat):
    s=pick(lon,lat)
    if not s: return None
    SVC,ORG,Z,RES,_=s; W=512
    res=C/(256*2**Z); x=R*math.radians(lon); y=R*math.log(math.tan(math.pi/4+math.radians(lat)/2))
    x0=(x-ORIG)/res-W//2; y0=(ORIGY-y)/res-W//2
    cv=Image.new('L',(W,W),0)
    for col in range(int(x0//256),int((x0+W)//256)+1):
        for row in range(int(y0//256),int((y0+W)//256)+1):
            t=tile(SVC,ORG,Z,col,row)
            if t: cv.paste(t,(col*256-int(x0),row*256-int(y0)))
    a=np.asarray(cv,dtype=np.float32); b=np.asarray(cv.filter(ImageFilter.GaussianBlur(24)),dtype=np.float32)
    rel=a-b; lo,hi=np.percentile(rel,2),np.percentile(rel,98)
    sl=Image.fromarray(np.clip((rel-lo)/(hi-lo+1e-6)*255,0,255).astype('uint8'))
    return sl.crop((128,128,384,384))   # center 256
H=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); REV=f'{H}/review'
rows=list(csv.DictReader(open(f'{H}/labeled/labels.csv')))
from collections import defaultdict
bygrp=defaultdict(list)
for i,r in enumerate(rows):
    grp=r['source']; st=slrm_stamp(float(r['lon']),float(r['lat']))
    if st is None: continue
    os.makedirs(f'{REV}/{grp}',exist_ok=True)
    name=f"{r['tile']}_{i}.png"; st.save(f'{REV}/{grp}/{name}')
    bygrp[grp].append((name,st))
# contact sheets per group
for grp,items in bygrp.items():
    cols=10; cell=86; rn=math.ceil(len(items)/cols)
    sh=Image.new('RGB',(cols*cell,rn*cell+18),(18,18,18)); dr=ImageDraw.Draw(sh)
    dr.text((4,2),f"{grp}: {len(items)}",fill=(255,255,0))
    for i,(name,st) in enumerate(items):
        im=st.convert('RGB').resize((82,82)); x=(i%cols)*cell+2; y=(i//cols)*cell+16; sh.paste(im,(x,y))
        dr.line([(x+41-4,y+41),(x+41+4,y+41)],fill=(255,40,40)); dr.line([(x+41,y+41-4),(x+41,y+41+4)],fill=(255,40,40))
        dr.text((x+1,y+1),str(i+1),fill=(0,255,255))
    sh.save(f'{REV}/contact_{grp}.png'); print(f'  {grp}: {len(items)} -> contact_{grp}.png')
print('review built in',REV)
