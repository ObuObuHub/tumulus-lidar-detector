import os,sys,math,subprocess,zipfile,csv
import numpy as np
from PIL import Image,ImageDraw,ImageFont
H=os.path.dirname(os.path.dirname(os.path.abspath(__file__)));CACHE="/tmp/laki3";CS=0.5;TPX=2000
# board_from_detections.py [det_csv] [out_board_png] [out_map_csv] [titlu_zona] — board numerotat scor-ASCUNS
IN_CSV  = sys.argv[1] if len(sys.argv)>1 else f'{H}/labeled/eval_session_22iun/catane_detections.csv'
OUT_PNG = sys.argv[2] if len(sys.argv)>2 else f'{H}/review/dolj_roundfp_board.png'
OUT_MAP = sys.argv[3] if len(sys.argv)>3 else '/tmp/dolj_roundfp_map.csv'
TITLU   = sys.argv[4] if len(sys.argv)>4 else 'Dolj nord'
APP=os.environ.get("QGIS_APP","/Applications/QGIS-final-4_0_3.app/Contents")
ENV=dict(os.environ,DYLD_FRAMEWORK_PATH=f"{APP}/Frameworks",PROJ_DATA=f"{APP}/Resources/qgis/proj",PROJ_LIB=f"{APP}/Resources/qgis/proj",GDAL_DATA=f"{APP}/Resources/qgis/gdal")
GT=f"{APP}/MacOS/gdaltransform"
def trans(p,s,t):
    r=subprocess.run([GT,"-s_srs",s,"-t_srs",t],input=f"{p[0]} {p[1]}\n",capture_output=True,text=True,env=ENV);a,b,_=r.stdout.split();return float(a),float(b)
def dl(nk,ek):
    p=f"{CACHE}/{nk}_{ek}.npy"
    if os.path.exists(p): return np.load(p)
    z=f"{CACHE}/{nk}_{ek}.zip"
    if not os.path.exists(z): subprocess.run(["curl","-s","--max-time","60","-o",z,f"https://geoportal.ancpi.ro/laki3_mnt/zip/{nk}_{ek}.zip"],check=False)
    try: zf=zipfile.ZipFile(z);asc=[n for n in zf.namelist() if n.lower().endswith('.asc')]
    except: return None
    if not asc: return None
    raw=zf.read(asc[0]).decode('latin-1').replace(',','.');L=raw.split('\n');i=0
    while i<len(L) and L[i].split() and L[i].split()[0].lower() in('ncols','nrows','xllcorner','yllcorner','cellsize','nodata_value'):i+=1
    d=np.fromstring(' '.join(L[i:]),sep=' ',dtype=np.float32)[:TPX*TPX].reshape(TPX,TPX);d[d==-9999]=np.nan;np.save(p,d);return d
def hs(dem,cs,azs=(315,45,135,225,270,0),alt=35):
    gy,gx=np.gradient(dem,cs);sl=np.arctan(np.hypot(gx,gy));asp=np.arctan2(-gy,gx);o=np.zeros_like(dem);ar=math.radians(alt)
    for az in azs: azr=math.radians(360-az+90);o+=np.clip(np.sin(ar)*np.cos(sl)+np.cos(ar)*np.sin(sl)*np.cos(azr-asp),0,1)
    return o/len(azs)

# load pool, dedup 150m, top-N by score
rows=list(csv.DictReader(open(IN_CSV)))
pts=[(float(r['score']),float(r['lon']),float(r['lat'])) for r in rows]
pts.sort(key=lambda x:-x[0])
kept=[]
for s,lo,la in pts:
    if any(math.hypot((lo-k[1])*111000*math.cos(math.radians(la)),(la-k[2])*111000)<150 for k in kept): continue
    kept.append((s,lo,la))
N=min(54,len(kept)); kept=kept[:N]
print(f"deduped -> {len(kept)} candidați (top {N} by score)")

# bbox mosaic
lons=[k[1] for k in kept]; lats=[k[2] for k in kept]
clon=(min(lons)+max(lons))/2; clat=(min(lats)+max(lats))/2
ests=[];nords=[]
for s,lo,la in kept:
    e,n=trans((lo,la),"EPSG:4326","EPSG:3844");ests.append(e);nords.append(n)
e0=int(min(ests)//1000)-1;e1=int(max(ests)//1000)+1;n0=int(min(nords)//1000)-1;n1=int(max(nords)//1000)+1
xll0=e0*1000;ytop0=(n1+1)*1000;W=(e1-e0+1)*TPX;Hh=(n1-n0+1)*TPX
print(f"mosaic {W}x{Hh} tiles E{e0}-{e1} N{n0}-{n1}")
mos=np.full((Hh,W),np.nan,np.float32);miss=0
for nk in range(n0,n1+1):
    for ek in range(e0,e1+1):
        d=dl(nk,ek)
        if d is None: miss+=1; continue
        ox=int((ek*1000-xll0)/CS);oy=int((ytop0-(nk+1)*1000)/CS);mos[oy:oy+TPX,ox:ox+TPX]=d[:TPX,:TPX]
print(f"tiles lipsă: {miss}")
WIN=int(120/CS)
def stamp(e,n):
    px=int((e-xll0)/CS);py=int((ytop0-n)/CS);w=mos[py-WIN//2:py+WIN//2,px-WIN//2:px+WIN//2]
    if w.shape!=(WIN,WIN) or np.isnan(w).mean()>0.1: return None
    w=np.nan_to_num(w,nan=np.nanmedian(w));h=hs(w,CS);lo,hi=np.percentile(h,2),np.percentile(h,98)
    return np.asarray(Image.fromarray(np.clip((h-lo)/(hi-lo+1e-9)*255,0,255).astype('uint8')).resize((150,150)),np.uint8)

cols=9;cell=156;mp=open(OUT_MAP,'w');mw=csv.writer(mp);mw.writerow(['idx','lon','lat','score'])
items=[]
for i,(s,lo,la) in enumerate(kept,1):
    e,n=trans((lo,la),"EPSG:4326","EPSG:3844");im=stamp(e,n)
    items.append((i,lo,la,s,im));mw.writerow([i,f"{lo:.5f}",f"{la:.5f}",f"{s:.3f}"])
mp.close()
rowsN=(len(items)+cols-1)//cols
img=Image.new('RGB',(cols*cell,rowsN*cell+26),(15,15,15));dr=ImageDraw.Draw(img)
try: ft=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial Bold.ttf',16)
except: ft=ImageFont.load_default()
dr.text((6,4),f"{TITLU} ({clat:.3f},{clon:.3f}) — {len(items)} candidati model, NU langa movile known.  ROSU=fals-pozitiv  ·  VERDE=de investigat/posibil real",fill=(255,255,80),font=ft)
for i,lo,la,s,im in items:
    x=((i-1)%cols)*cell;y=((i-1)//cols)*cell+26
    if im is not None: img.paste(Image.fromarray(im).convert('RGB'),(x+3,y+20))
    else: dr.rectangle([x+3,y+20,x+3+150,y+170],outline=(80,80,80))
    dr.text((x+4,y+2),f"#{i}",fill=(120,230,255),font=ft)
img.save(OUT_PNG)
print(f"-> {OUT_PNG} ({len([1 for it in items if it[4] is not None])}/{len(items)} randate); map -> {OUT_MAP}")
