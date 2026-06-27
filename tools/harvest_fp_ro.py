#!/usr/bin/env python3
# harvest_fp_ro.py [N_TARGET] [STEP_M] [MAXPERTILE] — extrage negative FP din cache-ul laki3 RO,
# CLASIFICATE: plough(araturi) / ditch(santuri) / stream(parauri). EXCLUDE dome-like (posibili tumuli).
# Recipe stampe IDENTIC cu neg_stamp training: 80m DEM 0.5m -> downs la 2m -> hillshade@2m -> resize 128 RAW.
# -> dataset_neg_ro_fp5k/<class>_<i>.png + manifest.csv ; boards review/fpneg_<class>.jpg + fpneg_excluded_dome.jpg
import os,sys,math,subprocess,csv,glob,random
import numpy as np
from PIL import Image,ImageDraw,ImageFont
H=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE="/tmp/laki3";CS=0.5;TPX=2000;f=int(round(2.0/CS))      # f=4 -> 2m effective (EXACT training)
WIN_M=80;WPX=int(WIN_M/CS)                                    # 160 px = 80 m
OUT=f"{H}/dataset_neg_ro_fp5k"
N_TARGET=int(sys.argv[1]) if len(sys.argv)>1 else 5000
STEP_M=int(sys.argv[2]) if len(sys.argv)>2 else 60
MAXPERTILE=int(sys.argv[3]) if len(sys.argv)>3 else 25
random.seed(0);np.random.seed(0)
APP=os.environ.get("QGIS_APP","/Applications/QGIS-final-4_0_3.app/Contents")
ENV=dict(os.environ,DYLD_FRAMEWORK_PATH=f"{APP}/Frameworks",PROJ_DATA=f"{APP}/Resources/qgis/proj",PROJ_LIB=f"{APP}/Resources/qgis/proj",GDAL_DATA=f"{APP}/Resources/qgis/gdal")
GT=f"{APP}/MacOS/gdaltransform"
def hs(dem,cs,azs=(315,45,135,225,270,0),alt=35):
    gy,gx=np.gradient(dem,cs);sl=np.arctan(np.hypot(gx,gy));asp=np.arctan2(-gy,gx);o=np.zeros_like(dem);ar=math.radians(alt)
    for az in azs: azr=math.radians(360-az+90);o+=np.clip(np.sin(ar)*np.cos(sl)+np.cos(ar)*np.sin(sl)*np.cos(azr-asp),0,1)
    return o/len(azs)
def downs(a,fc):
    Hh,Ww=a.shape;return a[:Hh//fc*fc,:Ww//fc*fc].reshape(Hh//fc,fc,Ww//fc,fc).mean((1,3))
def neg_stamp(w):  # w = 160x160 DEM 0.5m -> EXACT recipe neg_stamp training
    if w.shape!=(WPX,WPX) or np.isnan(w).mean()>0.05: return None
    d2=downs(np.nan_to_num(w,nan=np.nanmedian(w)),f);h=hs(d2,CS*f);lo,hi=np.percentile(h,2),np.percentile(h,98)
    if hi-lo<1e-6: return None
    return np.asarray(Image.fromarray(np.clip((h-lo)/(hi-lo)*255,0,255).astype('uint8')).resize((128,128)),np.uint8)
def feats(w):
    z=downs(np.nan_to_num(w,nan=np.nanmedian(w)),f);n=z.shape[0]      # 40x40 @ 2m
    ys,xs=np.mgrid[0:n,0:n].astype(float)
    A=np.c_[xs.ravel(),ys.ravel(),np.ones(n*n)];co,_,_,_=np.linalg.lstsq(A,z.ravel(),rcond=None)
    r=z-(co[0]*xs+co[1]*ys+co[2])                                    # detrend (plan scos)
    cx=cy=n/2.0;rad=np.hypot(xs-cx,ys-cy)*CS*f
    inner=rad<=14;ring=(rad>=22)&(rad<=36)
    if inner.sum()<6 or ring.sum()<6: return None
    prom=float(r[inner].mean())                                      # + convex (dom) / - concav (sant/parau)
    relief=float(np.percentile(r,98)-np.percentile(r,2))
    rough=float(r.std())
    valley=float(r[inner].mean()-r[ring].mean())                     # centru sub inel = vale/albie
    gy,gx=np.gradient(z,CS*f)                                        # coerenta directionala (structure tensor global)
    Sxx=float((gx*gx).mean());Syy=float((gy*gy).mean());Sxy=float((gx*gy).mean())
    tr=Sxx+Syy;dsc=max((tr/2)**2-(Sxx*Syy-Sxy*Sxy),0.0);l1=tr/2+math.sqrt(dsc);l2=tr/2-math.sqrt(dsc)
    coh=float((l1-l2)/(l1+l2+1e-9))
    # simetrie radiala (dom = profil radial neted): varianta pe inele / medie
    sym=0.0
    prof=[]
    for k in range(0,28,4):
        m=(rad>=k)&(rad<k+4)
        if m.sum()>=4: prof.append(r[m].mean())
    if len(prof)>=4:
        prof=np.array(prof);mono=float(np.mean(np.diff(prof)<=0))     # descreste din centru = dom
    else: mono=0.0
    return dict(prom=prom,relief=relief,rough=rough,valley=valley,coh=coh,mono=mono)
def label(F):
    p,relief,rough,valley,coh,mono=F['prom'],F['relief'],F['rough'],F['valley'],F['coh'],F['mono']
    if relief<0.18: return None                                      # plat mort = inutil
    # DOME-VETO (posibil tumul): centru convex + simetric/monoton + nedirectional -> EXCLUDE din TOATE clasele
    if p>0.22 and mono>=0.75 and coh<0.5 and relief>0.35: return 'EXCL_dome'
    # PLOUGH (araturi): textura directionala fina, fara movila neta
    if coh>=0.55 and abs(p)<0.22 and rough<0.55: return 'plough'
    # DITCH (santuri/canale): directional + centru concav (sant)
    if coh>=0.45 and p<=-0.15: return 'ditch'
    # STREAM (parauri/albii): centru in vale, concav, mai putin directional
    if valley<=-0.28 and p<0.05 and coh<0.6: return 'stream'
    return None
# ---- iterate tiles ----
tiles=glob.glob(f"{CACHE}/*.npy");random.shuffle(tiles)
os.makedirs(OUT,exist_ok=True)
targets={'plough':N_TARGET//3+ (N_TARGET%3>0),'ditch':N_TARGET//3+ (N_TARGET%3>1),'stream':N_TARGET//3}
buckets={'plough':[],'ditch':[],'stream':[]};excl=[]
step=int(STEP_M/CS);hw=WPX//2;ntiles=0
for tp in tiles:
    if all(len(buckets[c])>=targets[c] for c in buckets): break
    try: nk,ek=os.path.basename(tp)[:-4].split('_');nk=int(nk);ek=int(ek)
    except: continue
    try: T=np.load(tp)
    except: continue
    if T.shape[0]<WPX or T.shape[1]<WPX: continue
    ntiles+=1;percls={'plough':0,'ditch':0,'stream':0}
    ys=list(range(hw,T.shape[0]-hw,step));xs=list(range(hw,T.shape[1]-hw,step));random.shuffle(ys)
    for py in ys:
        for px in xs:
            if sum(percls.values())>=MAXPERTILE: break
            w=T[py-hw:py+hw,px-hw:px+hw]
            if w.shape!=(WPX,WPX) or np.isnan(w).mean()>0.05: continue
            F=feats(w)
            if F is None: continue
            lab=label(F)
            if lab is None: continue
            est=ek*1000+px*CS;nord=(nk+1)*1000-py*CS                 # EPSG:3844 (stereo70)
            if lab=='EXCL_dome':
                if len(excl)<400:
                    st=neg_stamp(w)
                    if st is not None: excl.append((st,est,nord,F))
                continue
            if len(buckets[lab])>=targets[lab] or percls[lab]>=MAXPERTILE//2: continue
            st=neg_stamp(w)
            if st is None: continue
            buckets[lab].append((st,est,nord,F));percls[lab]+=1
        if sum(percls.values())>=MAXPERTILE: break
    if ntiles%50==0: print(f"  {ntiles} tiles | plough {len(buckets['plough'])} ditch {len(buckets['ditch'])} stream {len(buckets['stream'])} | excl_dome {len(excl)}",flush=True)
print(f"SCAN gata: {ntiles} tiles | plough {len(buckets['plough'])} ditch {len(buckets['ditch'])} stream {len(buckets['stream'])} | excluse dome {len(excl)}",flush=True)
# ---- batch transform stereo70 -> lon/lat ----
allpts=[(c,i,e,nn) for c in buckets for i,(st,e,nn,F) in enumerate(buckets[c])]
inp="".join(f"{e} {nn}\n" for _,_,e,nn in allpts)
r=subprocess.run([GT,"-s_srs","EPSG:3844","-t_srs","EPSG:4326"],input=inp,capture_output=True,text=True,env=ENV)
ll=[(float(x.split()[0]),float(x.split()[1])) for x in r.stdout.strip().split("\n")] if r.stdout.strip() else [(0,0)]*len(allpts)
llmap={(c,i):(lo,la) for (c,i,_,_),(lo,la) in zip(allpts,ll)}
# ---- save + manifest ----
mf=open(f"{OUT}/manifest.csv","w");mw=csv.writer(mf);mw.writerow(['file','class','lon','lat','prom','coh','relief','valley','rough','mono']);ntot=0
for c in buckets:
    for i,(st,e,nn,F) in enumerate(buckets[c]):
        fn=f"{c}_{i:05d}.png";Image.fromarray(st).save(f"{OUT}/{fn}")
        lo,la=llmap.get((c,i),(0,0));mw.writerow([fn,c,f"{lo:.6f}",f"{la:.6f}",f"{F['prom']:.3f}",f"{F['coh']:.3f}",f"{F['relief']:.3f}",f"{F['valley']:.3f}",f"{F['rough']:.3f}",f"{F['mono']:.2f}"]);ntot+=1
mf.close();print(f"SALVAT: {ntot} stampe -> {OUT}/ (manifest.csv)",flush=True)
# ---- verification boards (contact sheet per clasa) ----
def board(items,title,path,cols=20,rows=15,thumb=72):
    sel=items[:cols*rows]
    W=cols*thumb;Hh=rows*thumb+30
    im=Image.new('RGB',(W,Hh),(15,15,18));d=ImageDraw.Draw(im)
    try: ft=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial Bold.ttf',16)
    except: ft=ImageFont.load_default()
    for k,(st,e,nn,F) in enumerate(sel):
        r_,c_=divmod(k,cols);x=c_*thumb;y=30+r_*thumb
        th=Image.fromarray(st).resize((thumb-2,thumb-2));im.paste(th,(x+1,y+1))
    d.rectangle([0,0,W,28],fill=(12,12,14));d.text((6,5),title,fill=(255,255,255),font=ft)
    im.save(path,quality=88);print(f"-> {path} ({len(sel)} thumbs / {len(items)} total)")
for c in buckets:
    random.shuffle(buckets[c]);board(buckets[c],f"NEG {c.upper()} (RO laki3 0.5m) — {len(buckets[c])} total, arata {min(300,len(buckets[c]))}",f"{H}/review/fpneg_{c}.jpg")
if excl: board(excl,f"EXCLUSE ca DOME (posibili tumuli, NU intra in negative) — {len(excl)}",f"{H}/review/fpneg_excluded_dome.jpg")
