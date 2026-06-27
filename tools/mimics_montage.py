#!/usr/bin/env python3
# mimics_montage.py — arată FP-urile reziduale (mimici) lângă movilele reale, crisp 0.5m hillshade+SLRM,
# ca să se VADĂ că-s aproape identice. Din /tmp/dets_v3_gated.csv (Catane gate-passing @recall100). -> review/mimics.jpg
import os,math,subprocess,csv
import numpy as np
from PIL import Image,ImageDraw,ImageFont
H=os.path.dirname(os.path.dirname(os.path.abspath(__file__)));CACHE="/tmp/laki3";CS=0.5;TPX=2000
APP=os.environ.get("QGIS_APP","/Applications/QGIS-final-4_0_3.app/Contents")
ENV=dict(os.environ,DYLD_FRAMEWORK_PATH=f"{APP}/Frameworks",PROJ_DATA=f"{APP}/Resources/qgis/proj",PROJ_LIB=f"{APP}/Resources/qgis/proj",GDAL_DATA=f"{APP}/Resources/qgis/gdal")
GTb=f"{APP}/MacOS/gdaltransform"
def trans(pts,s,t):
    inp="\n".join(f"{a} {b}" for a,b in pts)+"\n";r=subprocess.run([GTb,"-s_srs",s,"-t_srs",t],input=inp,capture_output=True,text=True,env=ENV)
    return [tuple(map(float,l.split()[:2])) for l in r.stdout.strip().split("\n") if l.split()]
_tc={}
def tile(nk,ek):
    k=(nk,ek)
    if k not in _tc:
        if len(_tc)>120:_tc.clear()
        p=f"{CACHE}/{nk}_{ek}.npy";_tc[k]=np.load(p) if os.path.exists(p) else None
    return _tc[k]
def hs(dem,cs,azs=(315,45,135,225,270,0),alt=35):
    gy,gx=np.gradient(dem,cs);sl=np.arctan(np.hypot(gx,gy));asp=np.arctan2(-gy,gx);o=np.zeros_like(dem);ar=math.radians(alt)
    for az in azs: azr=math.radians(360-az+90);o+=np.clip(np.sin(ar)*np.cos(sl)+np.cos(ar)*np.sin(sl)*np.cos(azr-asp),0,1)
    return o/len(azs)
def boxblur1(a,r):
    Hh,Ww=a.shape;ii=np.zeros((Hh+1,Ww+1),np.float64);ii[1:,1:]=np.cumsum(np.cumsum(a,0),1)
    ys=np.arange(Hh);xs=np.arange(Ww);y0=np.clip(ys-r,0,Hh);y1=np.clip(ys+r+1,0,Hh);x0=np.clip(xs-r,0,Ww);x1=np.clip(xs+r+1,0,Ww)
    A=ii[y1][:,x1]-ii[y0][:,x1]-ii[y1][:,x0]+ii[y0][:,x0];cnt=((y1-y0)[:,None]*(x1-x0)[None,:]).astype(np.float64)
    return (A/cnt).astype(np.float32)
WIN=140.0;HALF=int(WIN/CS)
def crops(est,nord):
    e0=int((est-WIN)//1000);e1=int((est+WIN)//1000);n0=int((nord-WIN)//1000);n1=int((nord+WIN)//1000)
    xll=e0*1000;ytop=(n1+1)*1000;W=(e1-e0+1)*TPX;Hh=(n1-n0+1)*TPX;mos=np.full((Hh,W),np.nan,np.float32);got=False
    for nk in range(n0,n1+1):
        for ek in range(e0,e1+1):
            d=tile(nk,ek)
            if d is None:continue
            got=True;ox=int((ek*1000-xll)/CS);oy=int((ytop-(nk+1)*1000)/CS);mos[oy:oy+TPX,ox:ox+TPX]=d[:TPX,:TPX]
    if not got:return None,None
    px=int((est-xll)/CS);py=int((ytop-nord)/CS);w=mos[py-HALF:py+HALF,px-HALF:px+HALF]
    if w.shape!=(2*HALF,2*HALF) or np.isnan(w).mean()>0.2:return None,None
    w=np.nan_to_num(w,nan=np.nanmedian(w))
    sh=hs(w,CS);lo,hi=np.percentile(sh,2),np.percentile(sh,98);shi=np.clip((sh-lo)/(hi-lo+1e-9)*255,0,255).astype('uint8')
    sl=w-boxblur1(boxblur1(boxblur1(w,int(30/CS)),int(30/CS)),int(30/CS));lo2,hi2=np.percentile(sl,2),np.percentile(sl,98)
    sli=np.clip((sl-lo2)/(hi2-lo2+1e-9)*255,0,255).astype('uint8')
    return shi,sli
rows=list(csv.DictReader(open('/tmp/dets_v3_gated.csv')))
s1=np.array([float(r['score']) for r in rows]);istp=np.array([int(r['istp']) for r in rows]);pg=np.array([float(r['pgate']) if r['pgate']!='NA' else 1.0 for r in rows])
thr=s1[(istp==1)&(pg>=0.70)].min();keep=(s1>=thr)&(pg>=0.70)
items=[(float(rows[i]['lon']),float(rows[i]['lat']),int(istp[i]),float(pg[i])) for i in range(len(rows)) if keep[i]]
items.sort(key=lambda x:(-x[2],-x[3]))  # REAL first, then FP
en=trans([(lo,la) for lo,la,t,g in items],"EPSG:4326","EPSG:3844")
def F(s,b=True):
    try:return ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial Bold.ttf' if b else '/System/Library/Fonts/Supplemental/Arial.ttf',s)
    except:return ImageFont.load_default()
cols=5;CW=220;CHH=480;n=len(items);rn=(n+cols-1)//cols
img=Image.new('RGB',(cols*CW,rn*CHH+56),(16,16,18));dr=ImageDraw.Draw(img)
nreal=sum(1 for x in items if x[2]==1);nfp=len(items)-nreal
dr.text((10,8),f"FP REZIDUALE (mimici) vs MOVILE REALE — Catane, crisp 0.5m. Sus=hillshade, jos=SLRM.",fill=(255,230,90),font=F(17))
dr.text((10,30),f"VERDE = {nreal} movile REALE | ROȘU = {nfp} FP reziduale. Trec AMBELE de filtru (scor g≥0.70). Vezi cât de identice-s.",fill=(225,225,225),font=F(14,False))
for k,((lo,la,t,g),(e,n)) in enumerate(zip(items,en)):
    cx=(k%cols)*CW;cy=(k//cols)*CHH+56;shi,sli=crops(e,n)
    if shi is not None:
        img.paste(Image.fromarray(shi).resize((210,210)).convert('RGB'),(cx+5,cy+22))
        img.paste(Image.fromarray(sli).resize((210,210)).convert('RGB'),(cx+5,cy+234))
    col=(60,235,60) if t==1 else (240,80,80)
    dr.rectangle([cx+4,cy+20,cx+216,cy+446],outline=col,width=3)
    dr.text((cx+6,cy+3),f"{'REALĂ' if t==1 else 'FP-mimic'}  g={g:.2f}",fill=col,font=F(15))
    dr.text((cx+6,cy+448),f"{la:.4f}, {lo:.4f}",fill=(170,170,175),font=F(11,False))
img.save(f'{H}/review/mimics.jpg',quality=90);print(f"-> review/mimics.jpg {img.size} | {nreal} reale + {nfp} FP")
