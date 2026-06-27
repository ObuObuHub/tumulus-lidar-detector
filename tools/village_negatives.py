#!/usr/bin/env python3
# village_negatives.py [N=10000] [RADIUS_M=500] [MAX_VILLAGES=600]
# NEGATIVE random din SATE (preponderenta sat = sursa #1 de fals-pozitive: case, parcele, canale).
# Esantionare RANDOM in jurul centrului fiecarui sat (NU mining pe scor -> mining-ul ar culege fix
# movilele reale si le-ar eticheta gresit ca negativ). Random = movila e rara -> contaminare neglijabila.
# Sate din OSM (/tmp/laki3_villages.json). Reteta IDENTICA pozitivilor/celorlalte negative:
# LAKI III DTM 0.5m -> fereastra 80m -> 2m efectiv -> hillshade multidir -> 128px. Exclude <150m movile.
# Scrie in dataset_neg_village/ (NU atinge dataset_neg). -> + review/village_neg_sample.png
import os,sys,math,subprocess,random,csv,zipfile,glob,json
import numpy as np
from PIL import Image
random.seed(20260621)
N=int(sys.argv[1]) if len(sys.argv)>1 else 10000
RADIUS_M=float(sys.argv[2]) if len(sys.argv)>2 else 500.0
MAX_VILLAGES=int(sys.argv[3]) if len(sys.argv)>3 else 600
H=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE="/tmp/laki3"; CS=0.5; TPX=2000; os.makedirs(CACHE,exist_ok=True)
OUT=f"{H}/dataset_neg_village"; os.makedirs(OUT,exist_ok=True)
APP=os.environ.get("QGIS_APP","/Applications/QGIS-final-4_0_3.app/Contents")
ENV=dict(os.environ,DYLD_FRAMEWORK_PATH=f"{APP}/Frameworks",PROJ_DATA=f"{APP}/Resources/qgis/proj",PROJ_LIB=f"{APP}/Resources/qgis/proj",GDAL_DATA=f"{APP}/Resources/qgis/gdal")
GT=f"{APP}/MacOS/gdaltransform"
def trans(pts,s,t):
    if not pts: return []
    inp="\n".join(f"{a} {b}" for a,b in pts)+"\n"
    r=subprocess.run([GT,"-s_srs",s,"-t_srs",t],input=inp,capture_output=True,text=True,env=ENV)
    return [(float(l.split()[0]),float(l.split()[1])) for l in r.stdout.strip().split("\n") if l.split()]
# coverage LAKI III 0.5m
BOXES=[(22.77,43.66,24.28,44.75),(21.73,43.91,23.67,45.29),(21.23,44.52,22.79,45.71),(22.52,44.53,23.88,45.38)]
def in_boxes(lo,la): return any(a<=lo<=c and b<=la<=d for a,b,c,d in BOXES)
# movile cunoscute (INCL. cele 7 noi din labels.csv) -> Stereo70, exclude 150m
mounds=[(float(r['lon']),float(r['lat'])) for r in csv.DictReader(open(f'{H}/labeled/labels.csv')) if r['verdict']=='mound']
ran=[]
for r in list(csv.reader(open(f'{H}/vest_tumuli.csv')))[1:]:
    try: ran.append((float(r[4]),float(r[5])))
    except: pass
known=np.array(trans(mounds+ran,"EPSG:4326","EPSG:3844")) if (mounds+ran) else np.empty((0,2))
def near(e,n,d=150):
    if known.shape[0]==0: return False
    return bool(np.any((known[:,0]-e)**2+(known[:,1]-n)**2<d*d))
def hillshade(dem,cs,azs=(315,45,135,225,270,0),alt=35):
    gy,gx=np.gradient(dem,cs);slope=np.arctan(np.hypot(gx,gy));aspect=np.arctan2(-gy,gx)
    o=np.zeros_like(dem);ar=math.radians(alt)
    for az in azs:
        azr=math.radians(360-az+90);o+=np.clip(np.sin(ar)*np.cos(slope)+np.cos(ar)*np.sin(slope)*np.cos(azr-aspect),0,1)
    return o/len(azs)
def downs(dem,f):
    if f<=1: return dem,CS
    Hh,Ww=dem.shape;H2,W2=Hh//f,Ww//f
    return dem[:H2*f,:W2*f].reshape(H2,f,W2,f).mean((1,3)),CS*f
def cut(dem,r,c,meters=80,eff=2.0,out=128):
    half=int((meters/2)/CS); w=dem[r-half:r+half,c-half:c+half]
    if w.shape!=(2*half,2*half) or np.isnan(w).mean()>0.05: return None
    d,cs2=downs(w.astype(np.float32),int(round(eff/CS)))
    h=hillshade(d,cs2);lo,hi=np.percentile(h,2),np.percentile(h,98)
    if hi-lo<1e-6: return None
    return Image.fromarray(np.clip((h-lo)/(hi-lo)*255,0,255).astype('uint8')).resize((out,out))
def dl(NUME):
    npy=f"{CACHE}/{NUME}.npy"
    if os.path.exists(npy): return np.load(npy)
    z=f"{CACHE}/{NUME}.zip"
    if not os.path.exists(z):
        subprocess.run(["curl","-s","--max-time","60","-o",z,f"https://geoportal.ancpi.ro/laki3_mnt/zip/{NUME}.zip"],check=False)
    try: zf=zipfile.ZipFile(z)
    except:
        if os.path.exists(z): os.remove(z)
        return None
    asc=[n for n in zf.namelist() if n.lower().endswith('.asc')]
    if not asc: return None
    raw=zf.read(asc[0]).decode('latin-1').replace(',','.'); lines=raw.split('\n'); i=0
    while i<len(lines) and lines[i].split() and lines[i].split()[0].lower() in ('ncols','nrows','xllcorner','yllcorner','cellsize','nodata_value'): i+=1
    data=np.fromstring(' '.join(lines[i:]),sep=' ',dtype=np.float32)
    try: data=data[:TPX*TPX].reshape(TPX,TPX)
    except: return None
    data[data==-9999]=np.nan; np.save(npy,data); return data
# sate din OSM, filtrate la acoperirea LAKI III
vj=json.load(open('/tmp/laki3_villages.json'))
vills=[(e['lon'],e['lat']) for e in vj.get('elements',[]) if 'lon' in e and in_boxes(e['lon'],e['lat'])]
random.shuffle(vills); vills=vills[:MAX_VILLAGES]
print(f"sate in LAKI III: {len(vills)} (din {len(vj.get('elements',[]))} OSM) | tinta {N} negative, raza {RADIUS_M}m",flush=True)
st=trans(vills,"EPSG:4326","EPSG:3844")  # centre Stereo70
Rpx=int(RADIUS_M/CS)
saved=0; used=0; idx=0; sample=[]
man=open(f"{OUT}/manifest.csv","w"); man.write("file,village_e,village_n,est,nord\n")
per_v=15  # cap/sat; multe sate ies din acoperirea LAKI III reala -> mai multe sate, mai putine/sat
for (e0,n0) in st:
    if saved>=N: break
    nkm,ekm=int(n0//1000),int(e0//1000); NUME=f"{nkm}_{ekm}"
    d=dl(NUME)
    if d is None or np.isnan(d).mean()>0.6: continue
    used+=1; got=0; tries=0
    cc=(e0-ekm*1000)/CS; rc=TPX-(n0-nkm*1000)/CS  # pixel centru sat in dala
    while got<per_v and saved<N and tries<per_v*6:
        tries+=1
        c=int(cc+random.uniform(-Rpx,Rpx)); r=int(rc+random.uniform(-Rpx,Rpx))
        if not (80<=c<TPX-80 and 80<=r<TPX-80): continue
        e=ekm*1000+c*CS; n=nkm*1000+(TPX-r)*CS
        if near(e,n): continue
        im=cut(d,r,c)
        if im is None: continue
        fn=f"{OUT}/vneg_{idx:05d}.png"; im.save(fn); man.write(f"{fn},{e0:.1f},{n0:.1f},{e:.1f},{n:.1f}\n")
        idx+=1; saved+=1; got+=1
        if len(sample)<48: sample.append(np.asarray(im))
    if used%25==0: print(f"  {used} sate, {saved} negative",flush=True)
man.close()
print(f"GATA: {saved} negative din {used} sate -> {OUT}",flush=True)
# montaj de control
if sample:
    cols=8;rows=(len(sample)+cols-1)//cols;M=Image.new('L',(cols*132,rows*132),40)
    for i,a in enumerate(sample): M.paste(Image.fromarray(a),((i%cols)*132+2,(i//cols)*132+2))
    M.save(f"{H}/review/village_neg_sample.png");print("-> review/village_neg_sample.png")
