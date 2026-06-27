#!/usr/bin/env python3
# sweep_board.py [CSV=review/sweep_dolj_final.csv] [TAG=dolj] [WIN_M=200] — planse cu crop-uri hillshade pt
# FIECARE candidat din CSV (lon,lat,score,coh,pgate), sortate dupa scor, paginate. Re-descarca dalele laki3
# necesare (cele sterse de prune in sweep). Index pe fiecare thumbnail = randul din index-CSV livrat alaturi.
# Border: VERDE = langa movila cunoscuta, PORTOCALIU = descoperire (din <TAG>_discoveries.csv).
# -> review/sweep_<TAG>_board_pNN.jpg + review/sweep_<TAG>_board_index.csv (idx,page,lon,lat,score,coh,pgate,tip)
import os,sys,math,subprocess,csv
import numpy as np
from PIL import Image,ImageDraw,ImageFont
H=os.path.dirname(os.path.dirname(os.path.abspath(__file__)));CACHE="/tmp/laki3";CS=0.5;TPX=2000
CSVf=sys.argv[1] if len(sys.argv)>1 else f"{H}/review/sweep_dolj_final.csv"
TAG=sys.argv[2] if len(sys.argv)>2 else 'dolj'
WIN_M=float(sys.argv[3]) if len(sys.argv)>3 else 200.0
APP=os.environ.get("QGIS_APP","/Applications/QGIS-final-4_0_3.app/Contents")
ENV=dict(os.environ,DYLD_FRAMEWORK_PATH=f"{APP}/Frameworks",PROJ_DATA=f"{APP}/Resources/qgis/proj",PROJ_LIB=f"{APP}/Resources/qgis/proj",GDAL_DATA=f"{APP}/Resources/qgis/gdal")
GT=f"{APP}/MacOS/gdaltransform"
def trans(pts,s,t):
    if not pts:return []
    inp="\n".join(f"{a} {b}" for a,b in pts)+"\n";r=subprocess.run([GT,"-s_srs",s,"-t_srs",t],input=inp,capture_output=True,text=True,env=ENV)
    return [tuple(map(float,l.split()[:2])) for l in r.stdout.strip().split("\n") if l.split()]
def hs(dem,cs,azs=(315,45,135,225,270,0),alt=35):
    gy,gx=np.gradient(dem,cs);sl=np.arctan(np.hypot(gx,gy));asp=np.arctan2(-gy,gx);o=np.zeros_like(dem);ar=math.radians(alt)
    for az in azs: azr=math.radians(360-az+90);o+=np.clip(np.sin(ar)*np.cos(sl)+np.cos(ar)*np.sin(sl)*np.cos(azr-asp),0,1)
    return o/len(azs)
def load_one(nk,ek):
    p=f"{CACHE}/{nk}_{ek}.npy"
    if os.path.exists(p):
        try:return np.load(p)
        except:pass
    z=f"{CACHE}/{nk}_{ek}.zip"
    if not os.path.exists(z):subprocess.run(["curl","-s","--max-time","120","-o",z,f"https://geoportal.ancpi.ro/laki3_mnt/zip/{nk}_{ek}.zip"],check=False)
    try:import zipfile;zf=zipfile.ZipFile(z)
    except:
        if os.path.exists(z):os.remove(z)
        return None
    asc=[n for n in zf.namelist() if n.lower().endswith('.asc')]
    if not asc:return None
    raw=zf.read(asc[0]).decode('latin-1').replace(',','.');lines=raw.split('\n');hdr={};i=0
    while i<len(lines):
        pp=lines[i].split()
        if len(pp)>=2 and pp[0].lower() in ('ncols','nrows','xllcorner','yllcorner','cellsize','nodata_value'):hdr[pp[0].lower()]=float(pp[1]);i+=1
        else:break
    nc=int(hdr['ncols']);nr=int(hdr['nrows']);nd=hdr.get('nodata_value',-9999)
    d=np.fromstring(' '.join(lines[i:]),sep=' ',dtype=np.float32)[:nc*nr].reshape(nr,nc);d[d==nd]=np.nan;np.save(p,d)
    if os.path.exists(z):os.remove(z)
    return d
def crop_hs(E,N,win_m,out=150):
    half=win_m/2.0
    ek0=int((E-half)//1000);ek1=int((E+half)//1000);nk0=int((N-half)//1000);nk1=int((N+half)//1000)
    xll0=ek0*1000;ytop0=(nk1+1)*1000;W=(ek1-ek0+1)*TPX;Hh=(nk1-nk0+1)*TPX
    mos=np.full((Hh,W),np.nan,np.float32);nt=0
    for nk in range(nk0,nk1+1):
        for ek in range(ek0,ek1+1):
            d=load_one(nk,ek)
            if d is None:continue
            nt+=1;ox=int((ek-ek0)*TPX);oy=int((nk1-nk)*TPX);mos[oy:oy+TPX,ox:ox+TPX]=d[:TPX,:TPX]
    if nt==0:return None
    cx=int((E-xll0)/CS);cy=int((ytop0-N)/CS);h=int(half/CS)
    w=mos[cy-h:cy+h,cx-h:cx+h]
    if w.shape!=(2*h,2*h) or np.isnan(w).mean()>0.5:return None
    w=np.nan_to_num(w,nan=float(np.nanmin(w)));sh=hs(w,CS);lo,hi=np.percentile(sh,2),np.percentile(sh,98)
    if hi-lo<1e-6:return None
    return np.asarray(Image.fromarray(np.clip((sh-lo)/(hi-lo)*255,0,255).astype('uint8')).resize((out,out)),np.uint8)

rows=list(csv.DictReader(open(CSVf)))
rows.sort(key=lambda r:-float(r['score']))
disc=set()
dpath=f"{H}/review/sweep_{TAG}_discoveries.csv"
if os.path.exists(dpath):
    for r in csv.DictReader(open(dpath)):disc.add((round(float(r['lon']),6),round(float(r['lat']),6)))
en=trans([(float(r['lon']),float(r['lat'])) for r in rows],"EPSG:4326","EPSG:3844")
print(f"{len(rows)} candidați, render crop-uri (win {WIN_M:.0f}m)...",flush=True)
THUMB=156;COLS=8;ROWS=6;PER=COLS*ROWS;LBL=20
try:ft=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial Bold.ttf',13);ftb=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial Bold.ttf',18)
except:ft=ftb=ImageFont.load_default()
idxrows=[];pages=(len(rows)+PER-1)//PER
for pg in range(pages):
    sub=rows[pg*PER:(pg+1)*PER];sen=en[pg*PER:(pg+1)*PER]
    bw=COLS*THUMB;bh=ROWS*(THUMB+LBL)+34;im=Image.new('RGB',(bw,bh),(16,16,20));dr=ImageDraw.Draw(im)
    for k,(r,(E,N)) in enumerate(zip(sub,sen)):
        gi=pg*PER+k+1
        cr=crop_hs(E,N,WIN_M);x=(k%COLS)*THUMB;y=34+(k//COLS)*(THUMB+LBL)
        if cr is not None:im.paste(Image.fromarray(cr),(x+3,y+LBL))
        else:dr.rectangle([x+3,y+LBL,x+THUMB-3,y+THUMB+LBL-3],fill=(40,40,46))
        isd=(round(float(r['lon']),6),round(float(r['lat']),6)) in disc
        col=(255,150,70) if isd else (90,210,120)
        dr.rectangle([x+2,y+LBL-1,x+THUMB-2,y+THUMB+LBL-2],outline=col,width=2)
        dr.text((x+5,y+1),f"{gi}  s{float(r['score']):.2f}",fill=col,font=ft)
        idxrows.append([gi,pg+1,r['lon'],r['lat'],r['score'],r['coh'],r['pgate'],'descoperire' if isd else 'langa-cunoscuta'])
    dr.rectangle([0,0,bw,30],fill=(10,10,12))
    dr.text((6,5),f"SWEEP {TAG.upper()} — candidați {pg*PER+1}-{pg*PER+len(sub)} / {len(rows)} (sortați după scor)   verde=lângă movilă cunoscută  portocaliu=descoperire",fill=(255,255,255),font=ftb)
    p=f"{H}/review/sweep_{TAG}_board_p{pg+1:02d}.jpg";im.save(p,quality=86);print(f"-> {p} ({len(sub)} crop-uri)",flush=True)
with open(f"{H}/review/sweep_{TAG}_board_index.csv",'w',newline='') as fo:
    w=csv.writer(fo);w.writerow(['idx','page','lon','lat','score','coh','pgate','tip']);w.writerows(idxrows)
print(f"-> review/sweep_{TAG}_board_index.csv ({pages} planșe)")
