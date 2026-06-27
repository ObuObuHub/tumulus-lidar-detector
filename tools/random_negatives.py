#!/usr/bin/env python3
# random_negatives.py [N=1000] [MAX_TILES=80]
# Eșantionează N căsuțe NEGATIVE random din LiDAR-ul accesibil (LAKI III DTM 0.5m, SV: DJ/MH/CS/GJ),
# homogenizate la 2 m efectiv, hillshade-DTM, 80 m -> 128px. Exclude <120 m de movile cunoscute.
# Fără analiză per-căsuță (movilele sunt extrem de rare -> contaminare neglijabilă la 1000).
import os,sys,math,subprocess,random,csv,zipfile,glob
import numpy as np
from PIL import Image
random.seed(20260620)
N=int(sys.argv[1]) if len(sys.argv)>1 else 1000
MAX_TILES=int(sys.argv[2]) if len(sys.argv)>2 else 80
CACHE="/tmp/laki3"; CS=0.5; TPX=2000; os.makedirs(CACHE,exist_ok=True)
OUT=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset_neg"); os.makedirs(OUT,exist_ok=True)
APP=os.environ.get("QGIS_APP","/Applications/QGIS-final-4_0_3.app/Contents")
ENV=dict(os.environ,DYLD_FRAMEWORK_PATH=f"{APP}/Frameworks",PROJ_DATA=f"{APP}/Resources/qgis/proj",PROJ_LIB=f"{APP}/Resources/qgis/proj",GDAL_DATA=f"{APP}/Resources/qgis/gdal")
GT=f"{APP}/MacOS/gdaltransform"
def trans(pts,s,t):
    if not pts: return []
    inp="\n".join(f"{a} {b}" for a,b in pts)+"\n"
    r=subprocess.run([GT,"-s_srs",s,"-t_srs",t],input=inp,capture_output=True,text=True,env=ENV)
    return [(float(l.split()[0]),float(l.split()[1])) for l in r.stdout.strip().split("\n") if l.split()]
# coverage LAKI III 0.5m (lon/lat boxes)
BOXES=[(22.77,43.66,24.28,44.75),(21.73,43.91,23.67,45.29),(21.23,44.52,22.79,45.71),(22.52,44.53,23.88,45.38)]
# known mounds -> Stereo70 (exclude)
mounds=[(float(r['lon']),float(r['lat'])) for r in csv.DictReader(open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'labeled/labels.csv'))) if r['verdict']=='mound']
ran=[]
for r in list(csv.reader(open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'vest_tumuli.csv'))))[1:]:
    try: ran.append((float(r[4]),float(r[5])))
    except: pass
known=np.array(trans(mounds+ran,"EPSG:4326","EPSG:3844")) if (mounds+ran) else np.empty((0,2))
def near(e,n,d=120):
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
    H,W=dem.shape;H2,W2=H//f,W//f
    return dem[:H2*f,:W2*f].reshape(H2,f,W2,f).mean((1,3)),CS*f
def cut(dem,r,c,meters=80,eff=2.0,out=128):
    half=int((meters/2)/CS); w=dem[r-half:r+half,c-half:c+half]
    if w.shape!=(2*half,2*half) or np.isnan(w).mean()>0.05: return None
    d,cs2=downs(w.astype(np.float32),int(round(eff/CS)))
    h=hillshade(d,cs2);lo,hi=np.percentile(h,2),np.percentile(h,98)
    return Image.fromarray(np.clip((h-lo)/(hi-lo+1e-6)*255,0,255).astype('uint8')).resize((out,out))
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
# candidate tiles: cached + random within boxes
cached=[os.path.basename(f)[:-4] for f in glob.glob(f"{CACHE}/*.npy")]
cand_pts=[]
for _ in range(MAX_TILES*4):
    b=random.choice(BOXES); cand_pts.append((random.uniform(b[0],b[2]),random.uniform(b[1],b[3])))
st=trans(cand_pts,"EPSG:4326","EPSG:3844")
numes=list(dict.fromkeys(cached+[f"{int(n//1000)}_{int(e//1000)}" for e,n in st]))
random.shuffle(numes)
saved=0; used_tiles=0; idx=0
man=open(f"{OUT}/manifest.csv","w"); man.write("file,tile,est,nord\n")
per_tile=max(8,N//min(MAX_TILES,len(numes))+2)
for NUME in numes:
    if saved>=N or used_tiles>=MAX_TILES: break
    nkm,ekm=[int(x) for x in NUME.split("_")]
    d=dl(NUME)
    if d is None or np.isnan(d).mean()>0.6: continue
    used_tiles+=1; got=0; tries=0
    while got<per_tile and saved<N and tries<per_tile*5:
        tries+=1
        r=random.randint(80,TPX-80); c=random.randint(80,TPX-80)
        e=ekm*1000+c*CS; n=nkm*1000+(TPX-r)*CS
        if near(e,n): continue
        im=cut(d,r,c)
        if im is None: continue
        fn=f"{OUT}/neg_{idx:04d}.png"; im.save(fn); man.write(f"{fn},{NUME},{e:.1f},{n:.1f}\n")
        idx+=1; saved+=1; got+=1
    if used_tiles%10==0: print(f"  {used_tiles} dale, {saved} negative",flush=True)
man.close()
print(f"GATA: {saved} negative din {used_tiles} dale -> {OUT}")
