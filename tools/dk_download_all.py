#!/usr/bin/env python3
# dk_download_all.py — descarca TOT setul danez Rundhoj (coordonate din /tmp/dk_all_rundhoj.json),
# stampa identica cu rețeta RO (DTM 0.4m WCS -> hillshade multidir -> 2m -> 128px). REZUMABIL:
# sare peste fisierele deja descarcate (dkpos_{i:05d}.png) -> daca pica, re-rulezi si continua.
import os,urllib.request,urllib.parse,json,socket,io,math
import numpy as np
from PIL import Image
socket.setdefaulttimeout(60)
H=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); TK=open(f'{H}/.dk_token').read().strip(); UA={'User-Agent':'Mozilla/5.0'}
coords=json.load(open('/tmp/dk_all_rundhoj.json'))
OUTP=f'{H}/dataset_pos_dk'; os.makedirs(OUTP,exist_ok=True)
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
    if nod.mean()>0.2: return None
    a=np.where(nod,np.nanmean(a[~nod]) if (~nod).any() else 0,a)
    cr=int(M/2/native);c=a.shape[0]//2;w=a[c-cr:c+cr,c-cr:c+cr]
    if w.std()<0.01: return None
    hh=hs(w,native);d2=downs(hh,int(round(eff/native)));lo,hi=np.percentile(d2,2),np.percentile(d2,98)
    return Image.fromarray(np.clip((d2-lo)/(hi-lo+1e-6)*255,0,255).astype('uint8')).resize((out,out))
done=skip=fail=0
print(f"TOTAL Rundhoj de descarcat: {len(coords)} (rezumabil)",flush=True)
for i,(e,n) in enumerate(coords):
    fn=f'{OUTP}/dkpos_{i:05d}.png'
    if os.path.exists(fn): skip+=1; continue
    s=stamp(e,n)
    if s: s.save(fn); done+=1
    else: fail+=1
    if (done+fail)%200==0: print(f"  i={i+1}/{len(coords)} | nou {done} sarit {skip} esuat {fail}",flush=True)
# manifest complet din ce exista pe disc
mp=open(f'{OUTP}/manifest.csv','w'); mp.write('file,est,nord\n'); have=0
for i,(e,n) in enumerate(coords):
    fn=f'{OUTP}/dkpos_{i:05d}.png'
    if os.path.exists(fn): mp.write(f'{fn},{e:.1f},{n:.1f}\n'); have+=1
mp.close()
print(f"GATA: pe disc {have} pozitivi danezi (nou {done}, sarit {skip}, esuat {fail})",flush=True)
