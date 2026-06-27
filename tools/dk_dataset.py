#!/usr/bin/env python3
# dk_dataset.py [NPOS=300] [NNEG=500] — construiește setul DANEZ omogenizat la proiect.
# Pozitivi = movile Rundhøj (WFS) ; negative = teren danez random (departe de movile).
# Fiecare: WCS DTM 0.4m -> hillshade multidir -> 2m efectiv -> 128px (rețeta RO identică).
import sys,os,urllib.request,urllib.parse,json,socket,io,math,random,glob
import numpy as np
from PIL import Image
socket.setdefaulttimeout(60); random.seed(20260621)
NPOS=int(sys.argv[1]) if len(sys.argv)>1 else 300
NNEG=int(sys.argv[2]) if len(sys.argv)>2 else 500
H=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); TK=open(f'{H}/.dk_token').read().strip()
UA={'User-Agent':'Mozilla/5.0'}
# bbox-uri peste Danemarca (EPSG:25832): Jutland N/C/S, Fyn, Sjælland
BOXES=[(480000,6300000,540000,6360000),(500000,6180000,560000,6240000),(520000,6080000,580000,6140000),
       (560000,6240000,620000,6300000),(580000,6120000,640000,6180000),(680000,6160000,720000,6200000),
       (440000,6260000,500000,6320000),(620000,6300000,680000,6360000)]
def wfs_rundhoj(bbox,want=3000,page=1000,maxstart=60000):
    out=[];start=0
    while len(out)<want and start<maxstart:
        url="https://www.kulturarv.dk/ffpublic/wfs?"+urllib.parse.urlencode({
         "service":"WFS","version":"2.0.0","request":"GetFeature","typeNames":"public:fundogfortidsminder_punkt_fredet",
         "count":str(page),"startIndex":str(start),"outputFormat":"application/json","srsName":"EPSG:25832","bbox":f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]},EPSG:25832"})
        try:
            d=json.load(urllib.request.urlopen(urllib.request.Request(url,headers=UA)))
            feats=d.get('features',[])
        except Exception as e: print("wfs err",str(e)[:50]); break
        if not feats: break
        out+=[tuple(f['geometry']['coordinates']) for f in feats if f['properties'].get('anlaegstype')=='Rundhøj']
        if len(feats)<page: break
        start+=page
    return out
def hs(dem,cs,azs=(315,45,135,225,270,0),alt=35):
    gy,gx=np.gradient(dem,cs);sl=np.arctan(np.hypot(gx,gy));asp=np.arctan2(-gy,gx);o=np.zeros_like(dem);ar=math.radians(alt)
    for az in azs:
        azr=math.radians(360-az+90);o+=np.clip(np.sin(ar)*np.cos(sl)+np.cos(ar)*np.sin(sl)*np.cos(azr-asp),0,1)
    return o/len(azs)
def downs(a,f):
    if f<=1: return a
    Hh,Ww=a.shape;return a[:Hh//f*f,:Ww//f*f].reshape(Hh//f,f,Ww//f,f).mean((1,3))
def stamp(e,n,M=80,marg=90,native=0.4,eff=2.0,out=128):
    half=M/2+marg;px=int(2*half/native)
    url="https://api.dataforsyningen.dk/dhm_wcs_DAF?"+urllib.parse.urlencode({"service":"WCS","version":"1.0.0","request":"GetCoverage","coverage":"dhm_terraen","crs":"EPSG:25832","bbox":f"{e-half},{n-half},{e+half},{n+half}","width":px,"height":px,"format":"GTiff","token":TK})
    try: raw=urllib.request.urlopen(urllib.request.Request(url,headers=UA)).read()
    except: return None
    try: a=np.array(Image.open(io.BytesIO(raw)),dtype=np.float32)
    except: return None
    if a.shape[0]<px-2: return None
    nod=(a<-1000)|(a>9000)
    if nod.mean()>0.2: return None  # mare/nodata
    a=np.where(nod,np.nanmean(a[~nod]) if (~nod).any() else 0,a)
    cr=int(M/2/native);c=a.shape[0]//2;w=a[c-cr:c+cr,c-cr:c+cr]
    if w.std()<0.01: return None
    hh=hs(w,native);d2=downs(hh,int(round(eff/native)));lo,hi=np.percentile(d2,2),np.percentile(d2,98)
    return Image.fromarray(np.clip((d2-lo)/(hi-lo+1e-6)*255,0,255).astype('uint8')).resize((out,out))
# 1. pozitivi Rundhøj — adun din TOATE cutiile pt diversitate geografică (anti-leakage)
mounds=[]
perbox=max(400,NPOS//len(BOXES)*3)
for bbox in BOXES:
    r=wfs_rundhoj(bbox,want=perbox); mounds+=r; print(f"  bbox {bbox[0]},{bbox[1]} -> {len(r)} Rundhoj (cumul {len(mounds)})",flush=True)
# dedup ~30m
uniq=[]
for e,n in mounds:
    if all((e-u[0])**2+(n-u[1])**2>30*30 for u in uniq): uniq.append((e,n))
random.shuffle(uniq); uniq=uniq[:NPOS]
print(f"Rundhøj de procesat: {len(uniq)}",flush=True)
OUTP=f'{H}/dataset_pos_dk';os.makedirs(OUTP,exist_ok=True)
for f in glob.glob(f'{OUTP}/*.png'): os.remove(f)
mp=open(f'{OUTP}/manifest.csv','w');mp.write('file,est,nord\n');np_=0
for i,(e,n) in enumerate(uniq):
    s=stamp(e,n)
    if s: fn=f'{OUTP}/dkpos_{np_:04d}.png';s.save(fn);mp.write(f'{fn},{e:.1f},{n:.1f}\n');np_+=1
    if (i+1)%50==0: print(f"  poz {np_}/{i+1}",flush=True)
mp.close();print(f"POZITIVI DK: {np_}",flush=True)
# 2. negative random (în aceleași bbox-uri, departe de movile)
mset=uniq
def near(e,n): return any((e-m[0])**2+(n-m[1])**2<120*120 for m in mset)
OUTN=f'{H}/dataset_neg_dk';os.makedirs(OUTN,exist_ok=True)
for f in glob.glob(f'{OUTN}/*.png'): os.remove(f)
mn=open(f'{OUTN}/manifest.csv','w');mn.write('file,est,nord\n');nn=0;tries=0
while nn<NNEG and tries<NNEG*8:
    tries+=1
    bbox=random.choice(BOXES);e=random.uniform(bbox[0],bbox[2]);n=random.uniform(bbox[1],bbox[3])
    if near(e,n): continue
    s=stamp(e,n)
    if s: fn=f'{OUTN}/dkneg_{nn:04d}.png';s.save(fn);mn.write(f'{fn},{e:.1f},{n:.1f}\n');nn+=1
    if nn and nn%100==0: print(f"  neg {nn}/{NNEG}",flush=True)
mn.close();print(f"NEGATIVE DK: {nn}")
print(f"GATA: {np_} pozitivi + {nn} negative daneze")
