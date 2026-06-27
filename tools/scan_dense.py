#!/usr/bin/env python3
# scan_dense.py — scaneaza acoperirea MDH dupa campuri DENSE de movile (blob-uri rotunde intunecate),
# alege spoturile cele mai dense (departe de movile existente), randeaza placi pt incercuit.
import os,math,subprocess,csv,json
import numpy as np
from PIL import Image,ImageFilter,ImageDraw,ImageFont
R=6378137.0;C=2*math.pi*R;ORIG=-20037508.342787;ORIGY=20037508.342787;Z=17;MPP=C/(256*2**Z)
ORG="Q2Kmg0bQDn3rySgn";TDIR="/tmp/mdh_tiles";os.makedirs(TDIR,exist_ok=True)
H=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MDH=[("AR_MDH_tif",(20.67,45.86,22.77,46.70)),("BH_MDH_tif",(21.37,46.36,22.83,47.61)),("HD_MDH_tif",(22.32,45.23,23.60,46.37))]
def pick(lo,la):
    for s,(a,b,c,d) in MDH:
        if a<=lo<=c and b<=la<=d: return s
    return None
def tile(svc,col,row):
    fn=f"{TDIR}/{svc}_{col}_{row}.png"
    if not os.path.exists(fn):
        subprocess.run(["curl","-s","--max-time","20","-o",fn,f"https://tiles.arcgis.com/tiles/{ORG}/arcgis/rest/services/{svc}/MapServer/tile/{Z}/{row}/{col}"],check=False)
    try: return Image.open(fn).convert('L')
    except: return None
def window(svc,lon,lat,M):
    half=M/2/MPP;x=R*math.radians(lon);y=R*math.log(math.tan(math.pi/4+math.radians(lat)/2))
    px=(x-ORIG)/MPP;py=(ORIGY-y)/MPP;x0=px-half;y0=py-half;W=int(2*half)
    cv=Image.new('L',(W,W),0);ok=0
    for col in range(int(x0//256),int((x0+W)//256)+1):
        for row in range(int(y0//256),int((y0+W)//256)+1):
            t=tile(svc,col,row)
            if t: cv.paste(t,(col*256-int(x0),row*256-int(y0)));ok+=1
    return np.asarray(cv,dtype=np.float32) if ok else None
def blobcount(a):
    if a is None or a.std()<0.5: return 0
    lo,hi=np.percentile(a,2),np.percentile(a,98);n=np.clip((a-lo)/(hi-lo+1e-6)*255,0,255)
    im=Image.fromarray(n.astype('uint8'))
    g1=np.asarray(im.filter(ImageFilter.GaussianBlur(2)),float);g2=np.asarray(im.filter(ImageFilter.GaussianBlur(8)),float)
    dog=g2-g1  # dark blob -> pozitiv
    # local maxima: egal cu max pe vecinatate (MaxFilter)
    mx=np.asarray(Image.fromarray(np.clip(dog-dog.min(),0,255).astype('uint8')).filter(ImageFilter.MaxFilter(9)),float)
    base=np.clip(dog-dog.min(),0,255)
    peaks=(base>=mx-0.5)&(dog>np.percentile(dog,98.5))&(base>8)
    ys,xs=np.where(peaks)
    # dedup peaks apropiate
    pts=[]
    for y,x in zip(ys.tolist(),xs.tolist()):
        if all((x-px)**2+(y-py)**2>=100 for px,py in pts): pts.append((x,y))
    return len(pts)
existing=[(float(r['lon']),float(r['lat'])) for r in csv.DictReader(open(f'{H}/labeled/labels.csv')) if r['verdict']=='mound']
def far(lo,la): return all((lo-e[0])**2+(la-e[1])**2>(0.012)**2 for e in existing)
cands=[]
for clon,clat,span in [(21.42,46.16,0.06),(21.65,46.61,0.05),(21.6,46.85,0.07),(21.5,46.3,0.08)]:
    s=0.022; lo=clon-span
    while lo<=clon+span:
        la=clat-span
        while la<=clat+span:
            if pick(lo,la) and far(lo,la): cands.append((round(lo,4),round(la,4)))
            la+=s
        lo+=s
# dedup candidate apropiate
uniq=[]
for lo,la in cands:
    if all((lo-u[0])**2+(la-u[1])**2>(0.018)**2 for u in uniq): uniq.append((lo,la))
print(f"scanez {len(uniq)} centre (probă 450m)...",flush=True)
scored=[]
for i,(lo,la) in enumerate(uniq):
    svc=pick(lo,la); a=window(svc,lo,la,450); bc=blobcount(a)
    scored.append((bc,lo,la,svc))
    if (i+1)%15==0: print(f"  {i+1}/{len(uniq)}",flush=True)
scored.sort(reverse=True)
print("TOP spoturi dense:")
for bc,lo,la,svc in scored[:10]: print(f"  {bc} blob-uri @ {lo},{la} [{svc[:2]}]")
# randez placi pt top 6 cu >=4 blob-uri
top=[s for s in scored if s[0]>=4][:6]
try: fnt=ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf",22)
except: fnt=ImageFont.load_default()
out=[]
for k,(bc,lo,la,svc) in enumerate(top):
    a=window(svc,lo,la,1100)
    if a is None: continue
    plo,phi=np.percentile(a,2),np.percentile(a,98);n=np.clip((a-plo)/(phi-plo+1e-6)*255,0,255).astype('uint8')
    im=Image.fromarray(n).resize((1000,1000)).convert('RGB');dr=ImageDraw.Draw(im)
    dr.rectangle([0,0,520,30],fill=(0,0,0));dr.text((4,4),f"DENS{k+1} {lo:.4f},{la:.4f} (~{bc} candidati) MDH",fill=(0,255,255),font=fnt)
    fn=f"{H}/review/dense_{k+1}.png";im.save(fn);out.append((fn,lo,la,svc,bc))
json.dump(out,open('/tmp/dense_spots.json','w'))
print(f"RANDAT {len(out)} plăci dense -> review/dense_*.png")
