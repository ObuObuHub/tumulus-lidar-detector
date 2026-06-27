#!/usr/bin/env python3
# refilter_domes.py [APPLY] — re-filtrează dataset_neg_ro_fp5k scoțând domurile ROTUNDE (posibile movile)
# pe care veto-ul de fereastră le-a ratat (movile pe câmp arat: coerența ferestrei mare din arătură maschează
# centrul rotund). Măsoară ROTUNJIMEA CENTRULUI (coerență + simetrie pe miezul ~24m). Rotund+convex+simetric=EXCLUS.
# Fără APPLY = doar raportează + board-uri. Cu APPLY=1 = mută PNG excluse în _domes_excluded/, rescrie manifest,
# re-randează board-uri etichetate (index + ordine stabilă + board_map.csv). Re-extrage ferestre DEM din laki3.
import os,sys,math,subprocess,csv,glob,collections
import numpy as np
from PIL import Image,ImageDraw,ImageFont
H=os.path.dirname(os.path.dirname(os.path.abspath(__file__)));CACHE="/tmp/laki3";CS=0.5;f=int(round(2.0/CS));WPX=int(80/CS);TPX=2000
APPLY=len(sys.argv)>1 and sys.argv[1] in('1','apply','APPLY')
OUT=os.environ.get('OUTDIR',f"{H}/dataset_neg_ro_fp5k");EXC=OUT+"_domes_excluded"
APP=os.environ.get("QGIS_APP","/Applications/QGIS-final-4_0_3.app/Contents")
ENV=dict(os.environ,DYLD_FRAMEWORK_PATH=f"{APP}/Frameworks",PROJ_DATA=f"{APP}/Resources/qgis/proj",PROJ_LIB=f"{APP}/Resources/qgis/proj",GDAL_DATA=f"{APP}/Resources/qgis/gdal")
GT=f"{APP}/MacOS/gdaltransform"
def downs(a,fc):
    Hh,Ww=a.shape;return a[:Hh//fc*fc,:Ww//fc*fc].reshape(Hh//fc,fc,Ww//fc,fc).mean((1,3))
def central(w):  # w = 160x160 DEM 0.5m. Întoarce trăsături de CENTRU (rotunjime/elongare/convexitate).
    z=downs(np.nan_to_num(w,nan=np.nanmedian(w)),f);n=z.shape[0]
    ys,xs=np.mgrid[0:n,0:n].astype(float);A=np.c_[xs.ravel(),ys.ravel(),np.ones(n*n)];co,_,_,_=np.linalg.lstsq(A,z.ravel(),rcond=None)
    r=z-(co[0]*xs+co[1]*ys+co[2]);cx=cy=n/2.0;rad=np.hypot(xs-cx,ys-cy)*CS*f
    inner=rad<=14;mid=rad<=24
    if inner.sum()<6 or mid.sum()<12: return None
    prom=float(r[inner].mean())
    # coerență CENTRALĂ (doar miez <=24m): rotund=izotrop(jos), alungit=directional(sus)
    gy,gx=np.gradient(r,CS*f);mm=mid
    Sxx=float((gx*gx)[mm].mean());Syy=float((gy*gy)[mm].mean());Sxy=float((gx*gy)[mm].mean())
    tr=Sxx+Syy;dsc=max((tr/2)**2-(Sxx*Syy-Sxy*Sxy),0.0);l1=tr/2+math.sqrt(dsc);l2=tr/2-math.sqrt(dsc)
    ccoh=float((l1-l2)/(l1+l2+1e-9))
    # simetrie radială: variația pe inele a reziduului (jos = simetric/rotund)
    cvs=[]
    for k in range(0,24,4):
        m=(rad>=k)&(rad<k+4)
        if m.sum()>=6:
            v=r[m];mu=v.mean()
            if abs(mu)>1e-3: cvs.append(v.std()/abs(mu))
    sym=float(1.0/(1.0+np.mean(cvs))) if cvs else 0.0
    prof=[];
    for k in range(0,26,3):
        m=(rad>=k)&(rad<k+3)
        if m.sum()>=4: prof.append(r[m].mean())
    mono=float(np.mean(np.diff(np.array(prof))<=0)) if len(prof)>=4 else 0.0
    return dict(prom=prom,ccoh=ccoh,sym=sym,mono=mono)
_P=float(os.environ.get('PROM','0.06'));_C=float(os.environ.get('CCOH','0.55'));_S=float(os.environ.get('SYM','0.45'));_M=float(os.environ.get('MONO','0.50'))
def is_dome(F):  # rotund + convex + simetric + radial-descrescător = posibilă movilă -> EXCLUDE (prag asimetric: mai bine pierd un FP decât otrăvesc cu o movilă)
    return F is not None and F['prom']>_P and F['ccoh']<_C and F['sym']>=_S and F['mono']>=_M
# citește manifest, recuperează Stereo70 din lon/lat
rows=list(csv.DictReader(open(f"{OUT}/manifest.csv")))
inp="".join(f"{r['lon']} {r['lat']}\n" for r in rows)
res=subprocess.run([GT,"-s_srs","EPSG:4326","-t_srs","EPSG:3844"],input=inp,capture_output=True,text=True,env=ENV)
en=[(float(x.split()[0]),float(x.split()[1])) for x in res.stdout.strip().split("\n")]
for r,(e,nn) in zip(rows,en): r['_e']=e;r['_n']=nn;r['_nk']=int(nn//1000);r['_ek']=int(e//1000)
bytile=collections.defaultdict(list)
for i,r in enumerate(rows): bytile[(r['_nk'],r['_ek'])].append(i)
verdict={}  # idx -> (is_dome, F)
loaded=0
for (nk,ek),idxs in bytile.items():
    tp=f"{CACHE}/{nk}_{ek}.npy"
    if not os.path.exists(tp):
        for i in idxs: verdict[i]=(False,None)  # nu pot reverifica -> păstrez
        continue
    try: T=np.load(tp)
    except:
        for i in idxs: verdict[i]=(False,None)
        continue
    loaded+=1
    for i in idxs:
        e=rows[i]['_e'];nn=rows[i]['_n'];ox=int(round((e-ek*1000)/CS));oy=int(round(((nk+1)*1000-nn)/CS));hw=WPX//2
        if ox-hw<0 or oy-hw<0 or ox+hw>T.shape[1] or oy+hw>T.shape[0]:
            verdict[i]=(False,None);continue
        w=T[oy-hw:oy+hw,ox-hw:ox+hw]
        F=central(w) if w.shape==(WPX,WPX) else None
        verdict[i]=(is_dome(F),F)
ndome=sum(1 for i in verdict if verdict[i][0])
print(f"tiles re-citite {loaded} | {len(rows)} mostre | DOME detectate (de exclus) {ndome} ({100*ndome/len(rows):.1f}%)")
byc=collections.Counter();
for i,r in enumerate(rows):
    if verdict.get(i,(False,))[0]: byc[r['class']]+=1
print("excluse pe clasă:",dict(byc))
# board-uri: excluse (domuri) + sample kept, etichetate index + ordine stabilă
def load_png(fn): return np.asarray(Image.open(f"{OUT}/{fn}").convert('L'))
def board(items,title,path,cols=18,rows_=14,thumb=80):
    sel=items[:cols*rows_];W=cols*thumb;Hh=rows_*thumb+30;im=Image.new('RGB',(W,Hh),(15,15,18));d=ImageDraw.Draw(im)
    try: ft=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial Bold.ttf',15);fs=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',11)
    except: ft=fs=ImageFont.load_default()
    for k,(fn,lab) in enumerate(sel):
        r_,c_=divmod(k,cols);x=c_*thumb;y=30+r_*thumb
        try: im.paste(Image.fromarray(load_png(fn)).resize((thumb-2,thumb-2)),(x+1,y+1))
        except: pass
        d.rectangle([x,y,x+34,y+12],fill=(0,0,0));d.text((x+1,y),lab,fill=(120,255,120),font=fs)
    d.rectangle([0,0,W,28],fill=(12,12,14));d.text((6,5),title,fill=(255,255,255),font=ft);im.save(path,quality=88);print(f"-> {path} ({len(sel)}/{len(items)})")
dome_items=[(rows[i]['file'],f"{rows[i]['class'][:2]}{i}") for i in range(len(rows)) if verdict.get(i,(False,))[0]]
keep_items=[(rows[i]['file'],f"{rows[i]['class'][:2]}{i}") for i in range(len(rows)) if not verdict.get(i,(False,))[0]]
board(dome_items,f"DOMURI ROTUNDE de EXCLUS (re-filtru centru) — {len(dome_items)}",f"{H}/review/refilter_domes_excluded.jpg")
# map index->file pt verificare ulterioară
mm=open(f"{OUT}/board_map.csv","w");mw=csv.writer(mm);mw.writerow(['index','file','class','keep']);
for i,r in enumerate(rows): mw.writerow([i,r['file'],r['class'],0 if verdict.get(i,(False,))[0] else 1])
mm.close();print(f"-> {OUT}/board_map.csv (index->file)")
if APPLY:
    os.makedirs(EXC,exist_ok=True)
    kept=[]
    for i,r in enumerate(rows):
        if verdict.get(i,(False,))[0]:
            try: os.rename(f"{OUT}/{r['file']}",f"{EXC}/{r['file']}")
            except: pass
        else: kept.append(r)
    fields=[k for k in rows[0].keys() if not k.startswith('_')]
    mw=csv.writer(open(f"{OUT}/manifest.csv","w"));mw.writerow(fields)
    for r in kept: mw.writerow([r[k] for k in fields])
    print(f"APPLY: {len(kept)} păstrate, {ndome} mutate în {EXC}/. manifest rescris.")
    # re-randez board-uri curate pe clasă, etichetate
    bycl=collections.defaultdict(list)
    for i,r in enumerate(rows):
        if not verdict.get(i,(False,))[0]: bycl[r['class']].append((r['file'],f"{r['class'][:2]}{i}"))
    for c in bycl: board(bycl[c],f"HARD-NEG {c.upper()} CURAT (post re-filtru) — {len(bycl[c])}",f"{H}/review/fphard2_{c}.jpg")
