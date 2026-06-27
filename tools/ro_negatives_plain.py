#!/usr/bin/env python3
# ro_negatives_plain.py [N=5000] [MAX_TILES=220]
# Baza MARE de NEGATIVE din LiDAR-ul RO (LAKI III DTM 0.5m: DJ/MH/CS/GJ), in zone FARA movile
# confirmate (exclud 800 m de orice RAN/etichetat) si EVITAND MUNTII (filtru de relief: amplitudine
# elevatie pe context 300 m < FLAT_MAX -> doar campie/terase). Reteta identica cu pozitivii
# (80 m @ 2 m hillshade multidir DTM luminos -> 128px). -> dataset_neg_ro_plain/ + manifest.
import os,sys,math,subprocess,random,csv,zipfile,glob
import numpy as np
from PIL import Image
N=int(sys.argv[1]) if len(sys.argv)>1 else 5000
MAX_TILES=int(sys.argv[2]) if len(sys.argv)>2 else 220
SEED=int(sys.argv[3]) if len(sys.argv)>3 else 20260621
# BANDĂ de amplitudine pe context 300m: campie [0,20] / deal [20,90]. argv[4]=min argv[5]=max argv[6]=label
AMP_MIN=float(sys.argv[4]) if len(sys.argv)>4 else 0.0
AMP_MAX=float(sys.argv[5]) if len(sys.argv)>5 else 20.0
LABEL=sys.argv[6] if len(sys.argv)>6 else "plain"
random.seed(SEED)
EXCL=800.0          # m fata de orice movila cunoscuta (zona FARA movile)
FLAT_WIN=300.0      # m context pt calcul relief
CACHE="/tmp/laki3"; CS=0.5; TPX=2000; os.makedirs(CACHE,exist_ok=True)
OUT=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), f"dataset_neg_ro_{LABEL}"); os.makedirs(OUT,exist_ok=True)
APP=os.environ.get("QGIS_APP","/Applications/QGIS-final-4_0_3.app/Contents")
ENV=dict(os.environ,DYLD_FRAMEWORK_PATH=f"{APP}/Frameworks",PROJ_DATA=f"{APP}/Resources/qgis/proj",PROJ_LIB=f"{APP}/Resources/qgis/proj",GDAL_DATA=f"{APP}/Resources/qgis/gdal")
GT=f"{APP}/MacOS/gdaltransform"
def trans(pts,s,t):
    if not pts: return []
    inp="\n".join(f"{a} {b}" for a,b in pts)+"\n"
    r=subprocess.run([GT,"-s_srs",s,"-t_srs",t],input=inp,capture_output=True,text=True,env=ENV)
    return [(float(l.split()[0]),float(l.split()[1])) for l in r.stdout.strip().split("\n") if l.split()]
BOXES=[(22.77,43.66,24.28,44.75),(21.73,43.91,23.67,45.29),(21.23,44.52,22.79,45.71),(22.52,44.53,23.88,45.38)]
mounds=[(float(r['lon']),float(r['lat'])) for r in csv.DictReader(open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'labeled/labels.csv'))) if r['verdict']=='mound']
ran=[]
for r in list(csv.reader(open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'vest_tumuli.csv'))))[1:]:
    try: ran.append((float(r[4]),float(r[5])))
    except: pass
known=np.array(trans(mounds+ran,"EPSG:4326","EPSG:3844")) if (mounds+ran) else np.empty((0,2))
def near(e,n,d=EXCL):
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
def relief_amp(d,r,c,winpx):
    w=d[r-winpx:r+winpx,c-winpx:c+winpx]
    if w.size==0 or np.isnan(w).mean()>0.1: return 9e9
    return float(np.nanpercentile(w,98)-np.nanpercentile(w,2))
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
cached=[os.path.basename(f)[:-4] for f in glob.glob(f"{CACHE}/*.npy")]
cand_pts=[]
for _ in range(MAX_TILES*4):
    b=random.choice(BOXES); cand_pts.append((random.uniform(b[0],b[2]),random.uniform(b[1],b[3])))
stp=trans(cand_pts,"EPSG:4326","EPSG:3844")
numes=list(dict.fromkeys(cached+[f"{int(n//1000)}_{int(e//1000)}" for e,n in stp]))
random.shuffle(numes)
flatpx=int(FLAT_WIN/CS)
existing=sorted(glob.glob(f"{OUT}/roneg_*.png"))
idx=(int(os.path.basename(existing[-1])[6:11])+1) if existing else 0   # APPEND: continua indexarea
saved=0; used=0; rej_mt=0; rej_mound=0
newman=not os.path.exists(f"{OUT}/manifest.csv")
man=open(f"{OUT}/manifest.csv","a"); mw=csv.writer(man)
if newman: mw.writerow(["file","tile","est","nord","amp"])
print(f"APPEND de la idx={idx} (existente {len(existing)})",flush=True)
per_tile=max(10,N//min(MAX_TILES,len(numes))+2)
for NUME in numes:
    if saved>=N or used>=MAX_TILES: break
    nkm,ekm=[int(x) for x in NUME.split("_")]
    d=dl(NUME)
    if d is None or np.isnan(d).mean()>0.6: continue
    used+=1; got=0; tries=0
    while got<per_tile and saved<N and tries<per_tile*6:
        tries+=1
        r=random.randint(flatpx,TPX-flatpx); c=random.randint(flatpx,TPX-flatpx)
        e=ekm*1000+c*CS; n=nkm*1000+(TPX-r)*CS
        if near(e,n): rej_mound+=1; continue
        amp=relief_amp(d,r,c,flatpx)
        if amp<AMP_MIN or amp>AMP_MAX: rej_mt+=1; continue   # in afara benzii (campie/deal) -> respins
        im=cut(d,r,c)
        if im is None: continue
        fn=f"{OUT}/roneg_{idx:05d}.png"; im.save(fn); mw.writerow([os.path.basename(fn),NUME,f"{e:.1f}",f"{n:.1f}",f"{amp:.1f}"])
        idx+=1; saved+=1; got+=1
    if used%10==0: print(f"  {used} dale, {saved} negative (respins munte {rej_mt}, langa movila {rej_mound})",flush=True)
man.close()
# montaj control
fs=sorted(glob.glob(f"{OUT}/*.png")); random.seed(3); random.shuffle(fs); fs=fs[:48]
cols=8;rows=(len(fs)+cols-1)//cols
M=Image.new('L',(cols*132+4,rows*132+4),40)
for i,f in enumerate(fs): M.paste(Image.open(f).resize((128,128)),((i%cols)*132+2,(i//cols)*132+2))
SAMPLE=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), f"review/ro_{LABEL}_neg_sample.png"); M.save(SAMPLE)
print(f"GATA: {saved} negative '{LABEL}' (amp {AMP_MIN}-{AMP_MAX}m) din {used} dale (respins banda {rej_mt}, langa movila {rej_mound}) -> {OUT} + {SAMPLE}")
