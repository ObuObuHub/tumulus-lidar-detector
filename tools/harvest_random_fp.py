#!/usr/bin/env python3
# harvest_random_fp.py PREFIX [THR] — din /tmp/zone_dets.csv (scan zonă RANDOM fără movile), ia detecțiile ≥THR,
# aplică DOME-VETO (central roundness) ca să nu bage movile reale, taie restul ca hard-negative în
# dataset_neg_expert_fp/ (PREFIX_NNN.png) + board de verificare review/<PREFIX>_board.jpg. Pt bucla FP-harvest.
import os,sys,math,subprocess,csv,glob
import numpy as np
from PIL import Image,ImageFilter,ImageDraw,ImageFont
PREFIX=sys.argv[1] if len(sys.argv)>1 else 'rndfp';THR=float(sys.argv[2]) if len(sys.argv)>2 else 0.85
CACHE="/tmp/laki3";CS=0.5;TPX=2000;H=os.path.dirname(os.path.dirname(os.path.abspath(__file__)));f=int(round(2.0/CS));hw=int(40/CS)
APP=os.environ.get("QGIS_APP","/Applications/QGIS-final-4_0_3.app/Contents")
ENV=dict(os.environ,DYLD_FRAMEWORK_PATH=f"{APP}/Frameworks",PROJ_DATA=f"{APP}/Resources/qgis/proj",PROJ_LIB=f"{APP}/Resources/qgis/proj",GDAL_DATA=f"{APP}/Resources/qgis/gdal")
GTb=f"{APP}/MacOS/gdaltransform"
def trans(lo,la):
    r=subprocess.run([GTb,"-s_srs","EPSG:4326","-t_srs","EPSG:3844"],input=f"{lo} {la}\n",capture_output=True,text=True,env=ENV);a,b,_=r.stdout.split();return float(a),float(b)
def load_one(nk,ek):
    p=f"{CACHE}/{nk}_{ek}.npy";return np.load(p) if os.path.exists(p) else None
def hs(dem,cs,azs=(315,45,135,225,270,0),alt=35):
    gy,gx=np.gradient(dem,cs);sl=np.arctan(np.hypot(gx,gy));asp=np.arctan2(-gy,gx);o=np.zeros_like(dem);ar=math.radians(alt)
    for az in azs: azr=math.radians(360-az+90);o+=np.clip(np.sin(ar)*np.cos(sl)+np.cos(ar)*np.sin(sl)*np.cos(azr-asp),0,1)
    return o/len(azs)
def downs(a,fc):
    Hh,Ww=a.shape;return a[:Hh//fc*fc,:Ww//fc*fc].reshape(Hh//fc,fc,Ww//fc,fc).mean((1,3))
def homog(a):
    a2=np.asarray(Image.fromarray(a).filter(ImageFilter.GaussianBlur(0.8)),np.uint8);cdf=np.bincount(a2.ravel(),minlength=256).astype(np.float64).cumsum()
    return (cdf[a2]/cdf[-1]*255).astype(np.uint8) if cdf[-1]>0 else a2
def central(w):
    z=downs(np.nan_to_num(w,nan=np.nanmedian(w)),f);n=z.shape[0]
    ys,xs=np.mgrid[0:n,0:n].astype(float);A=np.c_[xs.ravel(),ys.ravel(),np.ones(n*n)];co,_,_,_=np.linalg.lstsq(A,z.ravel(),rcond=None)
    r=z-(co[0]*xs+co[1]*ys+co[2]);cx=cy=n/2.0;rad=np.hypot(xs-cx,ys-cy)*CS*f
    inner=rad<=14;mid=rad<=24
    if inner.sum()<6 or mid.sum()<12: return None
    prom=float(r[inner].mean());gy,gx=np.gradient(r,CS*f);mm=mid
    Sxx=float((gx*gx)[mm].mean());Syy=float((gy*gy)[mm].mean());Sxy=float((gx*gy)[mm].mean())
    tr=Sxx+Syy;dsc=max((tr/2)**2-(Sxx*Syy-Sxy*Sxy),0.0);l1=tr/2+math.sqrt(dsc);l2=tr/2-math.sqrt(dsc);ccoh=float((l1-l2)/(l1+l2+1e-9))
    cvs=[]
    for k in range(0,24,4):
        m=(rad>=k)&(rad<k+4)
        if m.sum()>=6:
            v=r[m];mu=v.mean()
            if abs(mu)>1e-3: cvs.append(v.std()/abs(mu))
    sym=float(1.0/(1.0+np.mean(cvs))) if cvs else 0.0
    prof=[]
    for k in range(0,26,3):
        m=(rad>=k)&(rad<k+3)
        if m.sum()>=4: prof.append(r[m].mean())
    mono=float(np.mean(np.diff(np.array(prof))<=0)) if len(prof)>=4 else 0.0
    return dict(prom=prom,ccoh=ccoh,sym=sym,mono=mono)
def is_dome(F): return F is not None and F['prom']>0.06 and F['ccoh']<0.55 and F['sym']>=0.45 and F['mono']>=0.50
def mosaic(est,nord,half=300):
    e0=int((est-half)//1000);e1=int((est+half)//1000);n0=int((nord-half)//1000);n1=int((nord+half)//1000)
    xll0=e0*1000;ytop0=(n1+1)*1000;W=(e1-e0+1)*TPX;Hh=(n1-n0+1)*TPX;mos=np.full((Hh,W),np.nan,np.float32)
    for nk in range(n0,n1+1):
        for ek in range(e0,e1+1):
            d=load_one(nk,ek)
            if d is None:continue
            ox=int((ek*1000-xll0)/CS);oy=int((ytop0-(nk+1)*1000)/CS);mos[oy:oy+TPX,ox:ox+TPX]=d[:TPX,:TPX]
    return mos,xll0,ytop0
rows=[r for r in csv.DictReader(open('/tmp/zone_dets.csv')) if float(r['score'])>=THR];rows.sort(key=lambda r:-float(r['score']))
outdir=f"{H}/dataset_neg_expert_fp";os.makedirs(outdir,exist_ok=True);start=len(glob.glob(f"{outdir}/*.png"))
WIN=140;DISP=170;cols=6;rr=max(1,(len(rows)+cols-1)//cols)
bd=Image.new('RGB',(cols*DISP,rr*DISP+26),(15,15,15));dr=ImageDraw.Draw(bd)
try: fnt=ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf",13)
except: fnt=ImageFont.load_default()
harv=0;veto=0
for k,r in enumerate(rows):
    lo=float(r['lon']);la=float(r['lat']);est,nord=trans(lo,la);mos,xll0,ytop0=mosaic(est,nord)
    px=int((est-xll0)/CS);py=int((ytop0-nord)/CS);w=mos[py-hw:py+hw,px-hw:px+hw]
    x=(k%cols)*DISP;y=(k//cols)*DISP+26;dome=False
    if w.shape==(2*hw,2*hw) and np.isnan(w).mean()<0.05:
        h=hs(np.nan_to_num(w,nan=np.nanmedian(w)),CS);l2,h2=np.percentile(h,2),np.percentile(h,98)
        im=Image.fromarray(np.clip((h-l2)/(h2-l2)*255,0,255).astype('uint8')).resize((DISP-6,DISP-26)).convert('RGB');bd.paste(im,(x+3,y+3))
        if is_dome(central(w)): dome=True;veto+=1
        else:
            d2=downs(np.nan_to_num(w,nan=np.nanmedian(w)),f);hh=hs(d2,CS*f);ll,hi=np.percentile(hh,2),np.percentile(hh,98)
            st=homog(np.asarray(Image.fromarray(np.clip((hh-ll)/(hi-ll)*255,0,255).astype('uint8')).resize((128,128)),np.uint8))
            Image.fromarray(st).save(f"{outdir}/{PREFIX}_{start+harv:03d}.png");harv+=1
    col=(255,80,80) if dome else (0,255,120)
    dr.text((x+4,y+DISP-22),f"#{k+1} s{float(r['score']):.2f}{' VETO' if dome else ''}",fill=col,font=fnt)
bd.save(f"{H}/review/{PREFIX}_board.jpg",quality=90)
print(f"≥{THR}: {len(rows)} | HARVESTED {harv} | DOME-VETO {veto} | dataset_neg_expert_fp={len(glob.glob(f'{outdir}/*.png'))}")
print(f"-> review/{PREFIX}_board.jpg")
