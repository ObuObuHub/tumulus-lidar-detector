#!/usr/bin/env python3
# neg_audit.py — clasifică negativele marcate (batch hardneg) după FORMĂ (principiul Andrei): LINIAR (coerență>0.70 =
# arătură/șanț = negativ CORECT) vs COMPACT/DOM (coerență<=0.70 = formă tumulară = etichetat GREȘIT negativ).
# Raport counts + planșe de REVIEW pt cele compacte (crisp hillshade+SLRM, numerotate) + map coord. -> review/negaudit_*.jpg
import os,math,subprocess,csv,sys
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
        if len(_tc)>140:_tc.clear()
        p=f"{CACHE}/{nk}_{ek}.npy";_tc[k]=np.load(p) if os.path.exists(p) else None
    return _tc[k]
def window(est,nord,half_m):
    h=int(half_m/CS);eks=sorted({int((est-half_m)//1000),int((est+half_m)//1000)});nks=sorted({int((nord-half_m)//1000),int((nord+half_m)//1000)})
    e0=min(eks);n1=max(nks);xll=e0*1000;ytop=(n1+1)*1000;W=(max(eks)-e0+1)*TPX;Hh=(n1-min(nks)+1)*TPX;mos=np.full((Hh,W),np.nan,np.float32);got=False
    for nk in nks:
        for ek in eks:
            d=tile(nk,ek)
            if d is None:continue
            got=True;ox=int((ek*1000-xll)/CS);oy=int((ytop-(nk+1)*1000)/CS);mos[oy:oy+TPX,ox:ox+TPX]=d[:TPX,:TPX]
    if not got:return None
    px=int((est-xll)/CS);py=int((ytop-nord)/CS);w=mos[py-h:py+h,px-h:px+h]
    return w if w.shape==(2*h,2*h) else None
def coherence(est,nord,rad_m=22):
    w=window(est,nord,rad_m)
    if w is None or np.isnan(w).mean()>0.1:return 0.0
    w=np.nan_to_num(w,nan=np.nanmedian(w));gy,gx=np.gradient(w);Jxx=(gx*gx).mean();Jyy=(gy*gy).mean();Jxy=(gx*gy).mean();den=Jxx+Jyy
    return math.sqrt((Jxx-Jyy)**2+4*Jxy*Jxy)/den if den>1e-12 else 0.0
def hs(dem,cs,azs=(315,45,135,225,270,0),alt=35):
    gy,gx=np.gradient(dem,cs);sl=np.arctan(np.hypot(gx,gy));asp=np.arctan2(-gy,gx);o=np.zeros_like(dem);ar=math.radians(alt)
    for az in azs: azr=math.radians(360-az+90);o+=np.clip(np.sin(ar)*np.cos(sl)+np.cos(ar)*np.sin(sl)*np.cos(azr-asp),0,1)
    return o/len(azs)
def boxblur1(a,r):
    Hh,Ww=a.shape;ii=np.zeros((Hh+1,Ww+1),np.float64);ii[1:,1:]=np.cumsum(np.cumsum(a,0),1)
    ys=np.arange(Hh);xs=np.arange(Ww);y0=np.clip(ys-r,0,Hh);y1=np.clip(ys+r+1,0,Hh);x0=np.clip(xs-r,0,Ww);x1=np.clip(xs+r+1,0,Ww)
    A=ii[y1][:,x1]-ii[y0][:,x1]-ii[y1][:,x0]+ii[y0][:,x0];cnt=((y1-y0)[:,None]*(x1-x0)[None,:]).astype(np.float64)
    return (A/cnt).astype(np.float32)
def crops(est,nord):
    w=window(est,nord,70)
    if w is None or np.isnan(w).mean()>0.2:return None,None
    w=np.nan_to_num(w,nan=np.nanmedian(w))
    sh=hs(w,CS);lo,hi=np.percentile(sh,2),np.percentile(sh,98);shi=np.clip((sh-lo)/(hi-lo+1e-9)*255,0,255).astype('uint8')
    sl=w-boxblur1(boxblur1(boxblur1(w,int(30/CS)),int(30/CS)),int(30/CS));lo2,hi2=np.percentile(sl,2),np.percentile(sl,98)
    sli=np.clip((sl-lo2)/(hi2-lo2+1e-9)*255,0,255).astype('uint8')
    return shi,sli
# mod: dacă primește un CSV (ex /tmp/neg_domelike.csv) → toate sunt de review (ordonate cum vin); altfel split pe coerență
INP=sys.argv[1] if len(sys.argv)>1 else None
if INP:
    neg=[(r['src'],float(r['lon']),float(r['lat'])) for r in csv.DictReader(open(INP))]
    en=trans([(lo,la) for s,lo,la in neg],"EPSG:4326","EPSG:3844")
    compact=[(s,lo,la,e,n,0.0) for (s,lo,la),(e,n) in zip(neg,en)]  # păstrează ordinea (dome-first)
    print(f"REVIEW din {INP}: {len(compact)} dome-candidate (ordonate dome-first)")
else:
    neg=[(r['src'],float(r['lon']),float(r['lat'])) for r in csv.DictReader(open('/tmp/curv_pool.csv')) if r['label']=='neg']
    en=trans([(lo,la) for s,lo,la in neg],"EPSG:4326","EPSG:3844")
    recs=[]
    for (s,lo,la),(e,n) in zip(neg,en):
        coh=coherence(e,n);recs.append((s,lo,la,e,n,coh))
        if len(recs)%200==0: print(f"  coh {len(recs)}/{len(neg)}",flush=True)
    linear=[r for r in recs if r[5]>0.70];compact=[r for r in recs if r[5]<=0.70]
    print(f"\nNEGATIVE marcate: {len(recs)}")
    print(f"  LINIARE (coh>0.70 = arătură/șanț = negativ CORECT, păstrăm): {len(linear)}")
    print(f"  COMPACTE (coh<=0.70): {len(compact)}")
    compact.sort(key=lambda r:r[5])
def F(s,b=True):
    try:return ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial Bold.ttf' if b else '/System/Library/Fonts/Supplemental/Arial.ttf',s)
    except:return ImageFont.load_default()
mapf=open('/tmp/negaudit_map.csv','w',newline='');mw=csv.writer(mapf);mw.writerow(['gidx','src','lon','lat','coh'])
PER=24;cols=6;CW=180;CHH=380;page=0;gidx=0
for start in range(0,len(compact),PER):
    chunk=compact[start:start+PER];page+=1;rn=(len(chunk)+cols-1)//cols
    img=Image.new('RGB',(cols*CW,rn*CHH+50),(16,16,18));dr=ImageDraw.Draw(img)
    dr.text((10,8),f"NEGATIVE compacte (formă de dom) — REVIEW. Pagina {page}. Sus=hillshade, jos=SLRM.",fill=(255,230,90),font=F(15))
    dr.text((10,28),"Marchează care sunt DOMURI reale (de scos din negative). Nemarcat = rămâne negativ.",fill=(220,220,220),font=F(13,False))
    for k,(s,lo,la,e,n,coh) in enumerate(chunk):
        gidx+=1;mw.writerow([gidx,s,f"{lo:.6f}",f"{la:.6f}",f"{coh:.3f}"])
        cx=(k%cols)*CW;cy=(k//cols)*CHH+50;shi,sli=crops(e,n)
        if shi is not None:
            img.paste(Image.fromarray(shi).resize((170,170)).convert('RGB'),(cx+4,cy+20))
            img.paste(Image.fromarray(sli).resize((170,170)).convert('RGB'),(cx+4,cy+192))
        dr.rectangle([cx+3,cy+18,cx+175,cy+364],outline=(245,170,40),width=2)
        dr.text((cx+5,cy+3),f"#{gidx} coh{coh:.2f}",fill=(245,180,60),font=F(13))
    out=f"{H}/review/negaudit_{page:02d}.jpg";img.save(out,quality=86);print(f"-> {out} ({len(chunk)})",flush=True)
mapf.close();print(f"map -> /tmp/negaudit_map.csv | {page} pagini, {len(compact)} compacte de review")
