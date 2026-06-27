#!/usr/bin/env python3
# sweep_05m.py — SWEEP de productie pe acoperirea LiDAR 0.5m (LAKI3), bloc-cu-bloc, REZUMABIL + disc-managed.
# Model de productie (combined_cnn.pt=r4) single-scara 80m + NMS + filtru coerenta (coh22<=0.70) + curbura (gate>=0.70).
# Emite, pt FIECARE candidat pastrat, DOAR: lon,lat,score,coh,pgate (fara distanta-RAN, fara judet — cf. Andrei 26.06).
#
# Grila: blocuri patrate de BLOCK_KM tile (1 tile=1km=2000px@0.5m), pas STEP_TILES (suprapunere = BLOCK_KM-STEP_TILES km,
#   >80m => fara cusatura ratata). Procesare pe RANDURI (N ascendent), in rand pe COLOANE (E ascendent).
# Disc: dalele descarcate de ACEST run se sterg cand raman in urma frontului (nu se mai ating de blocuri viitoare);
#   dalele preexistente in cache NU se sterg niciodata (activ de proiect).
# Stare: /tmp/sweep_<TAG>_state.json (blocuri facute, candidati acumulati, km2, timp, dale 404) => resume curat.
#
# Env:
#   SWEEP_BBOX="e0,e1,n0,n1"  (km EPSG:3844; default Dolj 322,429,233,336)
#   TAG=dolj  BLOCK_KM=8  STEP_TILES=7  STEP_M=12  CANDTHR=0.60  MINDISK_GB=6
#   MODEL=<repo>/combined_cnn.pt  KEEP_DOWNLOADS=0 (1=nu sterge nimic, pt teste)  MAXBLOCKS=0 (0=toate)
import os,sys,math,subprocess,csv,json,time,shutil,glob
import numpy as np
from PIL import Image,ImageFilter,ImageDraw,ImageFont
import torch,torch.nn as nn

H=os.path.dirname(os.path.dirname(os.path.abspath(__file__)));dev=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
CACHE="/tmp/laki3";CS=0.5;TPX=2000;os.makedirs(CACHE,exist_ok=True)
TAG=os.environ.get('TAG','dolj')
BLOCK_KM=int(os.environ.get('BLOCK_KM','8'))
STEP_TILES=int(os.environ.get('STEP_TILES','7'))      # suprapunere = BLOCK_KM-STEP_TILES km
STEP_M=float(os.environ.get('STEP_M','12'))
CANDTHR=float(os.environ.get('CANDTHR','0.60'))
MINDISK_GB=float(os.environ.get('MINDISK_GB','6'))
KEEP_DL=os.environ.get('KEEP_DOWNLOADS','0')=='1'
MAXBLOCKS=int(os.environ.get('MAXBLOCKS','0'))
MODEL=os.environ.get('MODEL',f'{H}/combined_cnn.pt')
bb=os.environ.get('SWEEP_BBOX','322,429,233,336').split(',')
E0,E1,N0,N1=[int(float(x)) for x in bb]
STATE=f"/tmp/sweep_{TAG}_state.json"
OUTCSV=f"{H}/review/sweep_{TAG}_candidates.csv"
import pyproj
_TF={}
def _tf(s,t):
    if (s,t) not in _TF:_TF[(s,t)]=pyproj.Transformer.from_crs(s,t,always_xy=True)
    return _TF[(s,t)]
def trans(pts,s,t):
    if not pts: return []
    tf=_tf(s,t);return [tuple(tf.transform(a,b)) for a,b in pts]
def hs(dem,cs,azs=(315,45,135,225,270,0),alt=35):
    gy,gx=np.gradient(dem,cs);sl=np.arctan(np.hypot(gx,gy));asp=np.arctan2(-gy,gx);o=np.zeros_like(dem);ar=math.radians(alt)
    for az in azs: azr=math.radians(360-az+90);o+=np.clip(np.sin(ar)*np.cos(sl)+np.cos(ar)*np.sin(sl)*np.cos(azr-asp),0,1)
    return o/len(azs)
def downs(a,fc):
    Hh,Ww=a.shape;return a[:Hh//fc*fc,:Ww//fc*fc].reshape(Hh//fc,fc,Ww//fc,fc).mean((1,3))
def homog(a):
    a2=np.asarray(Image.fromarray(a).filter(ImageFilter.GaussianBlur(0.8)),np.uint8);cdf=np.bincount(a2.ravel(),minlength=256).astype(np.float64).cumsum()
    return (cdf[a2]/cdf[-1]*255).astype(np.uint8) if cdf[-1]>0 else a2

DOWNLOADED=set()  # tile keys (nk,ek) descarcate de acest run (deletable)
NODATA=set()      # tile keys care au dat 404/eroare (skip la resume)
def load_one(nk,ek):
    if (nk,ek) in NODATA: return None
    p=f"{CACHE}/{nk}_{ek}.npy"
    if os.path.exists(p):
        try:return np.load(p)
        except:pass
    z=f"{CACHE}/{nk}_{ek}.zip"
    if not os.path.exists(z):subprocess.run(["curl","-s","--max-time","120","-o",z,f"https://geoportal.ancpi.ro/laki3_mnt/zip/{nk}_{ek}.zip"],check=False)
    try:import zipfile;zf=zipfile.ZipFile(z)
    except:
        if os.path.exists(z):os.remove(z)
        NODATA.add((nk,ek));return None
    asc=[n for n in zf.namelist() if n.lower().endswith('.asc')]
    if not asc:NODATA.add((nk,ek));return None
    raw=zf.read(asc[0]).decode('latin-1').replace(',','.');lines=raw.split('\n');hdr={};i=0
    while i<len(lines):
        pp=lines[i].split()
        if len(pp)>=2 and pp[0].lower() in ('ncols','nrows','xllcorner','yllcorner','cellsize','nodata_value'):hdr[pp[0].lower()]=float(pp[1]);i+=1
        else:break
    nc=int(hdr['ncols']);nr=int(hdr['nrows']);nd=hdr.get('nodata_value',-9999)
    d=np.fromstring(' '.join(lines[i:]),sep=' ',dtype=np.float32)[:nc*nr].reshape(nr,nc);d[d==nd]=np.nan
    np.save(p,d)
    if os.path.exists(z):os.remove(z)
    DOWNLOADED.add((nk,ek))
    return d

class Net(nn.Module):
    def __init__(s):
        super().__init__();s.c=nn.Sequential(nn.Conv2d(1,16,3,2,1),nn.ReLU(),nn.Conv2d(16,32,3,2,1),nn.ReLU(),nn.Conv2d(32,64,3,2,1),nn.ReLU(),nn.AdaptiveAvgPool2d(1));s.f=nn.Linear(64,1)
    def forward(s,x): return s.f(s.c(x).flatten(1)).squeeze(1)
net=Net().to(dev);net.load_state_dict(torch.load(MODEL,map_location=dev,weights_only=True));net.eval()

def coh22(mos,px,py,r_m=22):
    r=int(r_m/CS);w=mos[py-r:py+r,px-r:px+r]
    if w.shape!=(2*r,2*r) or np.isnan(w).mean()>0.1:return 0.0
    w=np.nan_to_num(w,nan=np.nanmedian(w));gy,gx=np.gradient(w);Jxx=(gx*gx).mean();Jyy=(gy*gy).mean();Jxy=(gx*gy).mean();den=Jxx+Jyy
    return math.sqrt((Jxx-Jyy)**2+4*Jxy*Jxy)/den if den>1e-12 else 0.0

def scan_block(e,n):
    """Scaneaza blocul cu origine tile (e,n) km, latura BLOCK_KM tile. Intoarce (cands, km2)."""
    e0=e;e1=e+BLOCK_KM-1;n0=n;n1=n+BLOCK_KM-1
    xll0=e0*1000;ytop0=(n1+1)*1000;W=(e1-e0+1)*TPX;Hh=(n1-n0+1)*TPX
    mos=np.full((Hh,W),np.nan,np.float32);nt=0
    for nk in range(n0,n1+1):
        for ek in range(e0,e1+1):
            d=load_one(nk,ek)
            if d is None:continue
            nt+=1;ox=int((ek-e0)*TPX);oy=int((n1-nk)*TPX);mos[oy:oy+TPX,ox:ox+TPX]=d[:TPX,:TPX]
    if nt==0:return [],0.0
    km2=float(np.isfinite(mos).sum())*CS*CS/1e6
    f=int(round(2.0/CS));hw=int(40/CS);step=int(STEP_M/CS)
    batch=[];pos=[]
    for py in range(hw,Hh-hw,step):
        for px in range(hw,W-hw,step):
            w=mos[py-hw:py+hw,px-hw:px+hw]
            if w.shape!=(2*hw,2*hw) or np.isnan(w).mean()>0.05:continue
            d2=downs(w,f);h=hs(d2,CS*f);lo,hi=np.percentile(h,2),np.percentile(h,98)
            if hi-lo<1e-6:continue
            batch.append(homog(np.asarray(Image.fromarray(np.clip((h-lo)/(hi-lo)*255,0,255).astype('uint8')).resize((128,128)),np.uint8)));pos.append((px,py))
    if not batch:return [],km2
    sc=[];X=torch.tensor(np.array(batch,dtype=np.uint8))
    with torch.no_grad():
        for k in range(0,len(X),512):sc.extend(torch.sigmoid(net(X[k:k+512].unsqueeze(1).float().to(dev)/255.)).cpu().numpy().tolist())
    sc=np.array(sc)
    # NMS 80m pe >=0.5
    order=np.argsort(-sc);kept=[]
    for k in order:
        if sc[k]<0.5:break
        px,py=pos[k]
        if any((px-q[0])**2+(py-q[1])**2<(80/CS)**2 for q in kept):continue
        kept.append((px,py,float(sc[k])))
    # pre-filtru score>=CANDTHR & coh<=0.70, apoi curbura doar pe subset
    subset=[]
    for px,py,s in kept:
        if s<CANDTHR:continue
        c=coh22(mos,px,py)
        if c>0.70:continue
        subset.append((px,py,s,c))
    del mos
    if not subset:return [],km2
    lls=trans([(xll0+px*CS,ytop0-py*CS) for px,py,s,c in subset],"EPSG:3844","EPSG:4326")
    fin=f"/tmp/_sweep_in_{TAG}.csv";fout=f"/tmp/_sweep_gate_{TAG}.csv"
    with open(fin,'w',newline='') as fo:
        w=csv.writer(fo);w.writerow(['lon','lat'])
        for lo,la in lls:w.writerow([f"{lo:.6f}",f"{la:.6f}"])
    subprocess.run([sys.executable,f"{H}/tools/curv_filter.py",fin,fout,f"{H}/curv_gate.json","0.70"],check=False)
    pg=[];
    try:
        for r in csv.DictReader(open(fout)):
            pg.append(float(r['pgate']) if r.get('pgate','NA') not in ('NA','') else 1.0)
    except:pg=[1.0]*len(subset)
    if len(pg)<len(subset):pg+=[1.0]*(len(subset)-len(pg))
    cands=[]
    for (px,py,s,c),(lo,la),p in zip(subset,lls,pg):
        if p>=0.70:cands.append([round(float(lo),6),round(float(la),6),round(float(s),3),round(float(c),3),round(float(p),3)])
    return cands,km2

def free_gb():
    return shutil.disk_usage(CACHE).free/2**30
def prune(next_wmin_e,next_nmin_n):
    if KEEP_DL:return 0
    rm=0
    for (nk,ek) in list(DOWNLOADED):
        if (ek+1)<=next_wmin_e and (nk+1)<=next_nmin_n:
            p=f"{CACHE}/{nk}_{ek}.npy"
            try:
                if os.path.exists(p):os.remove(p);rm+=1
            except:pass
            DOWNLOADED.discard((nk,ek))
    return rm

# ---- grila blocuri (randuri N asc, coloane E asc) ----
cols=list(range(E0,E1,STEP_TILES));rows=list(range(N0,N1,STEP_TILES))
blocks=[(n,e) for n in rows for e in cols]
BT=len(blocks)

# ---- stare / resume ----
st={"params":{"bbox":[E0,E1,N0,N1],"BLOCK_KM":BLOCK_KM,"STEP_TILES":STEP_TILES,"STEP_M":STEP_M,"CANDTHR":CANDTHR,"model":os.path.basename(MODEL)},
    "done":[],"cands":[],"km2":0.0,"t_scan":0.0,"nodata":[],"blocks_total":BT,"t_start":None}
if os.path.exists(STATE):
    try:
        st=json.load(open(STATE))
        for k in st.get("nodata",[]):NODATA.add(tuple(k))
        print(f"RESUME: {len(st['done'])}/{st.get('blocks_total',BT)} blocuri facute, {len(st['cands'])} candidati, {st['km2']:.0f}km2 scanat",flush=True)
    except Exception as ex:print("stare corupta, repornesc:",ex)
done=set(tuple(d) for d in st["done"])
if st.get("t_start") is None:st["t_start"]=time.time()

def save_state():
    st["nodata"]=[list(k) for k in NODATA]
    json.dump(st,open(STATE,'w'))
    with open(OUTCSV,'w',newline='') as fo:
        w=csv.writer(fo);w.writerow(['lon','lat','score','coh','pgate'])
        for c in st["cands"]:w.writerow(c)

print(f"SWEEP {TAG}: bbox E{E0}-{E1} N{N0}-{N1} km | {BT} blocuri ({BLOCK_KM}km pas {STEP_TILES}) | model {os.path.basename(MODEL)} | prag {CANDTHR} | disc liber {free_gb():.0f}GB",flush=True)
processed=0
for bi,(n,e) in enumerate(blocks):
    if (n,e) in done:continue
    if MAXBLOCKS and processed>=MAXBLOCKS:print(f"MAXBLOCKS={MAXBLOCKS} atins, opresc.",flush=True);break
    t0=time.time()
    # urmatorul bloc in rand (E) + urmatorul rand (N) pt prune
    row_cols=[c for c in cols if c>e]
    next_wmin_e=row_cols[0] if row_cols else 10**9
    nxt_rows=[r for r in rows if r>n]
    next_nmin_n=nxt_rows[0] if nxt_rows else 10**9
    try:
        cands,km2=scan_block(e,n)
    except Exception as ex:
        print(f"  [bloc {bi+1}/{BT} N{n} E{e}] EROARE: {ex}",flush=True);cands,km2=[],0.0
    dt=time.time()-t0
    st["cands"].extend(cands);st["km2"]+=km2;st["t_scan"]+=dt;done.add((n,e));st["done"].append([n,e])
    processed+=1
    rm=prune(next_wmin_e,next_nmin_n)
    # daca disc tot mic, prune agresiv (sterge TOT ce-i in urma frontului curent)
    if free_gb()<MINDISK_GB:rm+=prune(next_wmin_e,10**9 if not nxt_rows else next_nmin_n)
    save_state()
    el=st["t_scan"];nd=len(done);rate=(st["km2"]/el*60) if el>0 else 0
    rem=BT-nd;eta_h=(rem*(el/max(processed,1)))/3600
    print(f"  [{nd}/{BT}] N{n} E{e}: {km2:.0f}km2 {len(cands)}cand {dt:.0f}s | total {st['km2']:.0f}km2 {len(st['cands'])}cand | {rate:.1f}km2/min ETA~{eta_h:.1f}h | disc {free_gb():.0f}GB rm{rm}",flush=True)

save_state()
print(f"\n=== SWEEP {TAG} TERMINAT (sau MAXBLOCKS): {len(done)}/{BT} blocuri, {st['km2']:.0f}km2, {len(st['cands'])} candidati bruti -> {OUTCSV}",flush=True)
print(f"    timp scanare {st['t_scan']/3600:.1f}h | ruleaza finalize_sweep.py {TAG} pt dedup + descoperiri + harta",flush=True)
