#!/usr/bin/env python3
# mdh_dataset.py {neg N | pos | test}
# Stampe din serviciile Hegyi _MDH (Arad/Bihor/Hunedoara/Alba, 1m hillshade MULTIDIRECTIONAL).
# Rețetă IDENTICĂ pt pozitivi și negativi: fetch tile MDH -> fereastră 80m -> AUTOCONTRAST
# (percentile 2-98, MDH e spălăcit) -> downsample 2m efectiv -> 128px.
import sys,os,math,subprocess,random,csv,glob
import numpy as np
from PIL import Image
random.seed(20260620)
R=6378137.0;C=2*math.pi*R;ORIG=-20037508.342787;ORIGY=20037508.342787;Z=17
MPP=C/(256*2**Z)  # ~1.19 m/px
MDH=[("AR_MDH_tif",(20.67,45.86,22.77,46.70)),("BH_MDH_tif",(21.37,46.36,22.83,47.61)),
     ("HD_MDH_tif",(22.32,45.23,23.60,46.37)),("AB_MDH_tif",(22.66,45.44,23.82,46.59))]
ORG="Q2Kmg0bQDn3rySgn"; TDIR="/tmp/mdh_tiles"; os.makedirs(TDIR,exist_ok=True)
H=os.path.expanduser('~/lidar-match')
def pick(lon,lat):
    for svc,(a,b,c,d) in MDH:
        if a<=lon<=c and b<=lat<=d: return svc
    return None
_tc={}
def tile(svc,col,row):
    k=(svc,col,row)
    if k in _tc: return _tc[k]
    fn=f"{TDIR}/{svc}_{col}_{row}.png"
    if not os.path.exists(fn):
        subprocess.run(["curl","-s","--max-time","25","-o",fn,f"https://tiles.arcgis.com/tiles/{ORG}/arcgis/rest/services/{svc}/MapServer/tile/{Z}/{row}/{col}"],check=False)
    try: im=Image.open(fn).convert('L')
    except: im=None
    _tc[k]=im; return im
def stamp(lon,lat,meters=80,eff=2.0,out=128):
    svc=pick(lon,lat)
    if not svc: return None
    half=meters/2/MPP; x=R*math.radians(lon); y=R*math.log(math.tan(math.pi/4+math.radians(lat)/2))
    px=(x-ORIG)/MPP; py=(ORIGY-y)/MPP; x0=px-half; y0=py-half; W=int(2*half)
    cv=Image.new('L',(W,W),0); ok=False
    for col in range(int(x0//256),int((x0+W)//256)+1):
        for row in range(int(y0//256),int((y0+W)//256)+1):
            t=tile(svc,col,row)
            if t: cv.paste(t,(col*256-int(x0),row*256-int(y0))); ok=True
    if not ok: return None
    a=np.asarray(cv,dtype=np.float32)
    if a.std()<0.5: return None  # tile gol/uniform (în afara acoperirii)
    lo,hi=np.percentile(a,2),np.percentile(a,98)  # autocontrast (MDH spălăcit)
    a=np.clip((a-lo)/(hi-lo+1e-6),0,1)
    f=max(1,int(round(eff/MPP)))  # downsample 2m efectiv
    Hh,Ww=a.shape; a=a[:Hh//f*f,:Ww//f*f].reshape(Hh//f,f,Ww//f,f).mean((1,3))
    return Image.fromarray((a*255).astype('uint8')).resize((out,out))
mode=sys.argv[1] if len(sys.argv)>1 else 'test'
if mode=='test':
    # 6 negative random + 6 pozitivi MDH
    mounds=[(float(r['lon']),float(r['lat'])) for r in csv.DictReader(open(f'{H}/labeled/labels.csv')) if r['verdict']=='mound' and pick(float(r['lon']),float(r['lat']))]
    print(f"pozitivi în acoperire MDH: {len(mounds)}")
    posim=[stamp(lo,la) for lo,la in mounds[:6]]; posim=[p for p in posim if p]
    negim=[]
    while len(negim)<6:
        svc,(a,b,c,d)=random.choice(MDH); lo=random.uniform(a,c); la=random.uniform(b,d)
        s=stamp(lo,la)
        if s: negim.append(s)
    from PIL import ImageDraw
    sh=Image.new('RGB',(6*134,2*134+20),(15,15,15)); dr=ImageDraw.Draw(sh)
    dr.text((4,2),"SUS pozitivi MDH | JOS negative MDH (80m@2m, autocontrast)",fill=(0,255,255))
    for i,p in enumerate(posim): sh.paste(p.convert('RGB').resize((130,130)),(i*134+2,18))
    for i,p in enumerate(negim): sh.paste(p.convert('RGB').resize((130,130)),(i*134+2,18+134))
    sh.save('/tmp/mdh_test.png'); print("-> /tmp/mdh_test.png")
elif mode=='neg':
    N=int(sys.argv[2]) if len(sys.argv)>2 else 2000
    OUT=f'{H}/dataset_neg_mdh'; os.makedirs(OUT,exist_ok=True)
    for f in glob.glob(f'{OUT}/*.png'): os.remove(f)
    mounds=[(float(r['lon']),float(r['lat'])) for r in csv.DictReader(open(f'{H}/labeled/labels.csv')) if r['verdict']=='mound']
    def near(lo,la):
        return any(abs(lo-m[0])<0.0015 and abs(la-m[1])<0.0011 for m in mounds)  # ~120m
    man=open(f'{OUT}/manifest.csv','w'); man.write('file,lon,lat\n')
    saved=0; tries=0
    while saved<N and tries<N*6:
        tries+=1
        svc,(a,b,c,d)=random.choice(MDH); lo=random.uniform(a,c); la=random.uniform(b,d)
        if near(lo,la): continue
        s=stamp(lo,la)
        if s:
            fn=f'{OUT}/mdhneg_{saved:04d}.png'; s.save(fn); man.write(f'{fn},{lo:.6f},{la:.6f}\n'); saved+=1
        if saved and saved%500==0: print(f"  {saved}/{N}",flush=True)
    man.close(); print(f"GATA: {saved} negative MDH -> {OUT}")
elif mode=='pos':
    OUT=f'{H}/dataset_pos_mdh'; os.makedirs(OUT,exist_ok=True)
    for f in glob.glob(f'{OUT}/*.png'): os.remove(f)
    mounds=[r for r in csv.DictReader(open(f'{H}/labeled/labels.csv')) if r['verdict']=='mound' and pick(float(r['lon']),float(r['lat']))]
    n=0
    for i,r in enumerate(mounds):
        s=stamp(float(r['lon']),float(r['lat']))
        if s: s.save(f'{OUT}/mdhpos_{i:03d}.png'); n+=1
    print(f"GATA: {n}/{len(mounds)} pozitivi MDH -> {OUT}")
