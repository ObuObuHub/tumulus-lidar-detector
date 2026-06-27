import os,sys,math,subprocess,csv,json
import numpy as np
from PIL import Image,ImageDraw,ImageFont
H=os.path.expanduser('~/lidar-match');CACHE="/tmp/laki3";CS=0.5;TPX=2000
# extract_marked.py [marks_json] [map_csv] [prefix]
#   roșii -> negative dome-FP (recipe training) cu nume `domefp_<prefix>_NNN.png` (NU suprascrie alte zone)
#   verzii -> randate @200m pt verdict în review/green_investigate_<prefix>.png
MARKS  = sys.argv[1] if len(sys.argv)>1 else '/tmp/dolj_marks.json'
MAPCSV = sys.argv[2] if len(sys.argv)>2 else '/tmp/dolj_roundfp_map.csv'
PREFIX = sys.argv[3] if len(sys.argv)>3 else 'cot'
NEG_KEY= sys.argv[4] if len(sys.argv)>4 else 'red'   # 'red' (vechi) sau 'blank' (convenția verde-only)
APP="/Applications/QGIS-final-4_0_3.app/Contents"
ENV=dict(os.environ,DYLD_FRAMEWORK_PATH=f"{APP}/Frameworks",PROJ_DATA=f"{APP}/Resources/qgis/proj",PROJ_LIB=f"{APP}/Resources/qgis/proj",GDAL_DATA=f"{APP}/Resources/qgis/gdal")
GT=f"{APP}/MacOS/gdaltransform"
def trans(p,s,t):
    r=subprocess.run([GT,"-s_srs",s,"-t_srs",t],input=f"{p[0]} {p[1]}\n",capture_output=True,text=True,env=ENV);a,b,_=r.stdout.split();return float(a),float(b)
def dl(nk,ek):
    p=f"{CACHE}/{nk}_{ek}.npy";return np.load(p) if os.path.exists(p) else None
def hs(dem,cs,azs=(315,45,135,225,270,0),alt=35):
    gy,gx=np.gradient(dem,cs);sl=np.arctan(np.hypot(gx,gy));asp=np.arctan2(-gy,gx);o=np.zeros_like(dem);ar=math.radians(alt)
    for az in azs: azr=math.radians(360-az+90);o+=np.clip(np.sin(ar)*np.cos(sl)+np.cos(ar)*np.sin(sl)*np.cos(azr-asp),0,1)
    return o/len(azs)
def downs(a,fc):
    Hh,Ww=a.shape;return a[:Hh//fc*fc,:Ww//fc*fc].reshape(Hh//fc,fc,Ww//fc,fc).mean((1,3))

marks=json.load(open(MARKS))
mapc={int(r['idx']):(float(r['lon']),float(r['lat']),r['score']) for r in csv.DictReader(open(MAPCSV))}
red=[(b[0] if isinstance(b,(list,tuple)) else b) for b in marks.get(NEG_KEY,marks.get('red',[]))]; green=marks.get('green',[])

# build mosaic over bbox of ALL 54 points
allpts=[mapc[i] for i in mapc]
ests=[];nords=[]
for lo,la,s in allpts:
    e,n=trans((lo,la),"EPSG:4326","EPSG:3844");ests.append(e);nords.append(n)
e0=int(min(ests)//1000)-1;e1=int(max(ests)//1000)+1;n0=int(min(nords)//1000)-1;n1=int(max(nords)//1000)+1
xll0=e0*1000;ytop0=(n1+1)*1000;W=(e1-e0+1)*TPX;Hh=(n1-n0+1)*TPX
mos=np.full((Hh,W),np.nan,np.float32)
for nk in range(n0,n1+1):
    for ek in range(e0,e1+1):
        d=dl(nk,ek)
        if d is None: continue
        ox=int((ek*1000-xll0)/CS);oy=int((ytop0-(nk+1)*1000)/CS);mos[oy:oy+TPX,ox:ox+TPX]=d[:TPX,:TPX]

# ---- A) extract 50 red as NEGATIVES (training recipe: 80m -> 2m -> hillshade -> 2-98 -> 128px RAW) ----
f=int(round(2.0/CS));wpx=int(80/CS)
def neg_stamp(e,n):
    px=int((e-xll0)/CS);py=int((ytop0-n)/CS);w=mos[py-wpx//2:py+wpx//2,px-wpx//2:px+wpx//2]
    if w.shape!=(wpx,wpx) or np.isnan(w).mean()>0.05: return None
    d2=downs(np.nan_to_num(w,nan=np.nanmedian(w)),f);h=hs(d2,CS*f);lo,hi=np.percentile(h,2),np.percentile(h,98)
    if hi-lo<1e-6: return None
    return np.asarray(Image.fromarray(np.clip((h-lo)/(hi-lo)*255,0,255).astype('uint8')).resize((128,128)),np.uint8)
mf=open(f'{H}/dataset_neg_domefp/manifest_{PREFIX}.csv','w');mw=csv.writer(mf);mw.writerow(['file','est','nord','idx','score'])
nw=0
for i in red:
    lo,la,s=mapc[i];e,n=trans((lo,la),"EPSG:4326","EPSG:3844");st=neg_stamp(e,n)
    if st is None: print(f"  ⚠ #{i} neg skip (acoperire)"); continue
    fn=f"dataset_neg_domefp/domefp_{PREFIX}_{i:03d}.png";Image.fromarray(st).save(f"{H}/{fn}");mw.writerow([fn,f"{e:.1f}",f"{n:.1f}",i,s]);nw+=1
mf.close()
print(f"A) negative dome-FP scrise: {nw}/{len(red)} -> dataset_neg_domefp/ (prefix {PREFIX}, NU suprascrie alte zone)")

# ---- B) render 4 green candidates @ 200m for investigation ----
WIN=int(200/CS)
def view(e,n):
    px=int((e-xll0)/CS);py=int((ytop0-n)/CS);w=mos[py-WIN//2:py+WIN//2,px-WIN//2:px+WIN//2]
    if w.shape!=(WIN,WIN): return None
    w=np.nan_to_num(w,nan=np.nanmedian(w));h=hs(w,CS);lo,hi=np.percentile(h,2),np.percentile(h,98)
    return np.asarray(Image.fromarray(np.clip((h-lo)/(hi-lo+1e-9)*255,0,255).astype('uint8')).resize((300,300)),np.uint8)
cell=310;img=Image.new('RGB',(len(green)*cell,cell+26),(15,15,15));dr=ImageDraw.Draw(img)
try: ft=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial Bold.ttf',15)
except: ft=ImageFont.load_default()
for j,i in enumerate(green):
    lo,la,s=mapc[i];e,n=trans((lo,la),"EPSG:4326","EPSG:3844");v=view(e,n)
    if v is not None: img.paste(Image.fromarray(v).convert('RGB'),(j*cell+5,26))
    dr.text((j*cell+6,4),f"#{i}  scor {s}  ({la:.4f},{lo:.4f})",fill=(120,255,120),font=ft)
img.save(f'{H}/review/green_investigate_{PREFIX}.png')
print(f"B) verzi @200m -> review/green_investigate_{PREFIX}.png ({green})")
