#!/usr/bin/env python3
# greens_hires.py [WHICH=red|all|green] — crop-uri CLARE (0.5m nativ) hillshade+SLRM pt fiecare green din
# /tmp/greens_qa_map.csv (montage_idx,gate,lon,lat). Paginat 20/pagină -> review/greens_hi_PP.jpg
import os,sys,math,subprocess,csv
import numpy as np
from PIL import Image,ImageDraw,ImageFont
H=os.path.dirname(os.path.dirname(os.path.abspath(__file__)));CACHE="/tmp/laki3";CS=0.5;TPX=2000
WHICH=sys.argv[1] if len(sys.argv)>1 else 'red'
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
WIN=140.0;HALF=int(WIN/CS)  # 140m @0.5m = 280px nativ
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
rows=list(csv.DictReader(open('/tmp/greens_qa_map.csv')))
sel=[r for r in rows if (WHICH=='all' or (WHICH=='red' and float(r['gate'])<0.70) or (WHICH=='green' and float(r['gate'])>=0.70))]
en=trans([(float(r['lon']),float(r['lat'])) for r in sel],"EPSG:4326","EPSG:3844")
def F(s,b=True):
    try:return ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial Bold.ttf' if b else '/System/Library/Fonts/Supplemental/Arial.ttf',s)
    except:return ImageFont.load_default()
PER=20;cols=5;CW=250;CHH=560;page=0
for start in range(0,len(sel),PER):
    chunk=list(zip(sel[start:start+PER],en[start:start+PER]));page+=1
    rn=(len(chunk)+cols-1)//cols
    img=Image.new('RGB',(cols*CW,rn*CHH+50),(18,18,20));dr=ImageDraw.Draw(img)
    dr.text((10,8),f"Movile marcate de tine — CLAR (hillshade 0.5m sus / SLRM jos), 140m. Pagina {page}.",fill=(255,230,90),font=F(18))
    dr.text((10,30),"Spune care # sunt movile REALE. Roșu=filtrul le-ar tăia.",fill=(220,220,220),font=F(15,False))
    for k,(r,(e,n)) in enumerate(chunk):
        cx=(k%cols)*CW;cy=(k//cols)*CHH+50;shi,sli=crops(e,n)
        if shi is not None:
            img.paste(Image.fromarray(shi).resize((240,240)).convert('RGB'),(cx+5,cy+24))
            img.paste(Image.fromarray(sli).resize((240,240)).convert('RGB'),(cx+5,cy+268))
        g=float(r['gate']);col=(90,235,90) if g>=0.70 else (240,80,80)
        dr.rectangle([cx+3,cy+22,cx+247,cy+510],outline=col,width=3)
        dr.text((cx+6,cy+3),f"#{r['montage_idx']}  scor {g:.2f}",fill=col,font=F(16))
        dr.text((cx+6,cy+512),f"{float(r['lat']):.4f}, {float(r['lon']):.4f}",fill=(180,180,185),font=F(12,False))
    out=f"{H}/review/greens_hi_{page:02d}.jpg";img.save(out,quality=90);print(f"-> {out} {img.size} ({os.path.getsize(out)//1024}KB, {len(chunk)} movile)")
print(f"WHICH={WHICH}, {len(sel)} movile, {page} pagini")
