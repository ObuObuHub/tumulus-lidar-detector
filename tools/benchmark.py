#!/usr/bin/env python3
# benchmark.py GT_CSV [MODEL] [MATCH_M] — BENCHMARK în formatul literaturii (Németh&Benedek): scanare COMPLETĂ
# pe bbox-ul ground-truth-ului (LAKI3 0.5m, single-scară = mod de operare), potrivește detecțiile la GT,
# raportează precizie/recall/F1 la prag de operare + AUPRC la PREVALENȚA REALĂ (PR la nivel de detecție) + FP/km².
# GT_CSV = coloane lon,lat (movile confirmate, held-out). -> stdout metrics. NU atinge nimic.
import os,sys,math,subprocess,csv
import numpy as np
from PIL import Image,ImageFilter
import torch,torch.nn as nn
H=os.path.dirname(os.path.dirname(os.path.abspath(__file__)));dev=torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
GT=sys.argv[1];MODEL=sys.argv[2] if len(sys.argv)>2 else f'{H}/combined_cnn.pt';MATCH=float(sys.argv[3]) if len(sys.argv)>3 else 50.0
CACHE="/tmp/laki3";CS=0.5;os.makedirs(CACHE,exist_ok=True)
import pyproj
_TF={}
def _tf(s,t):
    if (s,t) not in _TF:_TF[(s,t)]=pyproj.Transformer.from_crs(s,t,always_xy=True)
    return _TF[(s,t)]
def trans(pts,s,t):
    if not pts:return []
    tf=_tf(s,t);return [tuple(tf.transform(a,b)) for a,b in pts]
def dl(nk,ek):
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
    d=np.fromstring(' '.join(lines[i:]),sep=' ',dtype=np.float32)[:nc*nr].reshape(nr,nc);d[d==nd]=np.nan;np.save(p,d);return d
def hs(dem,cs,azs=(315,45,135,225,270,0),alt=35):
    gy,gx=np.gradient(dem,cs);sl=np.arctan(np.hypot(gx,gy));asp=np.arctan2(-gy,gx);o=np.zeros_like(dem);ar=math.radians(alt)
    for az in azs: azr=math.radians(360-az+90);o+=np.clip(np.sin(ar)*np.cos(sl)+np.cos(ar)*np.sin(sl)*np.cos(azr-asp),0,1)
    return o/len(azs)
def downs(a,fc):
    Hh,Ww=a.shape;return a[:Hh//fc*fc,:Ww//fc*fc].reshape(Hh//fc,fc,Ww//fc,fc).mean((1,3))
def homog(a):
    a2=np.asarray(Image.fromarray(a).filter(ImageFilter.GaussianBlur(0.8)),np.uint8);cdf=np.bincount(a2.ravel(),minlength=256).astype(np.float64).cumsum()
    return (cdf[a2]/cdf[-1]*255).astype(np.uint8) if cdf[-1]>0 else a2
rows=list(csv.DictReader(open(GT)));gll=[(float(r['lon']),float(r['lat'])) for r in rows]
gt=trans(gll,"EPSG:4326","EPSG:3844")  # (est,nord)
if not gt:print("ERROR: ground-truth CSV is empty (0 lon,lat rows) - nothing to benchmark");sys.exit(2)
es=[e for e,n in gt];ns=[n for e,n in gt];MARG=400
e0=int((min(es)-MARG)//1000);e1=int((max(es)+MARG)//1000);n0=int((min(ns)-MARG)//1000);n1=int((max(ns)+MARG)//1000)
xll0=e0*1000;ytop0=(n1+1)*1000;W=(e1-e0+1)*2000;Hh=(n1-n0+1)*2000;mos=np.full((Hh,W),np.nan,np.float32);nt=0
for nk in range(n0,n1+1):
    for ek in range(e0,e1+1):
        d=dl(nk,ek)
        if d is None: continue
        nt+=1;ox=int((ek*1000-xll0)/CS);oy=int((ytop0-(nk+1)*1000)/CS);mos[oy:oy+2000,ox:ox+2000]=d[:2000,:2000]
if nt==0:print("ERROR: no LAKI3 tiles over the GT bbox - area not covered (Romania only), OR the tile download failed (check network / geoportal.ancpi.ro / curl)");sys.exit(2)
area_km2=(np.isfinite(mos).sum())*(CS*CS)/1e6
print(f"BENCHMARK {os.path.basename(GT)} | {len(gt)} GT | {nt} tiles | ~{area_km2:.1f} km² scanned | model {os.path.basename(MODEL)}",flush=True)
f=int(round(2.0/CS));hw=int(40/CS)
class Net(nn.Module):
    def __init__(s):
        super().__init__();s.c=nn.Sequential(nn.Conv2d(1,16,3,2,1),nn.ReLU(),nn.Conv2d(16,32,3,2,1),nn.ReLU(),nn.Conv2d(32,64,3,2,1),nn.ReLU(),nn.AdaptiveAvgPool2d(1));s.f=nn.Linear(64,1)
    def forward(s,x): return s.f(s.c(x).flatten(1)).squeeze(1)
net=Net().to(dev);net.load_state_dict(torch.load(MODEL,map_location=dev,weights_only=True));net.eval()
step=int(12/CS);batch=[];pos=[]
for py in range(hw,Hh-hw,step):
    for px in range(hw,W-hw,step):
        w=mos[py-hw:py+hw,px-hw:px+hw]
        if w.shape!=(2*hw,2*hw) or np.isnan(w).mean()>0.05: continue
        d2=downs(w,f);h=hs(d2,CS*f);lo,hi=np.percentile(h,2),np.percentile(h,98)
        if hi-lo<1e-6: continue
        batch.append(homog(np.asarray(Image.fromarray(np.clip((h-lo)/(hi-lo)*255,0,255).astype('uint8')).resize((128,128)),np.uint8)));pos.append((px,py))
sc=[]
X=torch.tensor(np.array(batch,dtype=np.uint8))
with torch.no_grad():
    for k in range(0,len(X),512): sc.extend(torch.sigmoid(net(X[k:k+512].unsqueeze(1).float().to(dev)/255.)).cpu().numpy().tolist())
sc=np.array(sc)
# NMS -> detecții (peak-uri)
order=np.argsort(-sc);kept=[]
for k in order:
    if sc[k]<0.30: break   # colectez până la 0.30 pt curba PR
    px,py=pos[k]
    if any((px-q[0])**2+(py-q[1])**2<(MATCH/CS)**2 for q in kept): continue
    kept.append((px,py,float(sc[k])))
# potrivire la GT (fiecare GT o singură dată, greedy pe scor desc)
gtpx=[(int((e-xll0)/CS),int((ytop0-n)/CS)) for e,n in gt]
used=[False]*len(gtpx);dets=[]  # (score, is_TP)
for px,py,s in kept:
    hit=-1;bd=1e18
    for gi,(gx,gy) in enumerate(gtpx):
        if used[gi]: continue
        dd=(px-gx)**2+(py-gy)**2
        if dd<(MATCH/CS)**2 and dd<bd: bd=dd;hit=gi
    if hit>=0: used[hit]=True;dets.append((s,1))
    else: dets.append((s,0))
nGT=len(gtpx)
# DUMP detectii (lon,lat,score,istp) pt cascada filtru curbura — kept[i] <-> dets[i] (inainte de sort)
if os.environ.get('DUMP'):
    dump=os.environ['DUMP'];es_d=[px*CS+xll0 for px,py,s in kept];ns_d=[ytop0-py*CS for px,py,s in kept]
    ll=trans(list(zip(es_d,ns_d)),"EPSG:3844","EPSG:4326")
    with open(dump,'w') as fo:
        fo.write('lon,lat,score,istp\n')
        for (lo,la),(px,py,s),(ss,istp) in zip(ll,kept,dets): fo.write(f"{lo:.6f},{la:.6f},{s:.4f},{istp}\n")
    print(f"  DUMP {len(kept)} detections -> {dump}",flush=True)
# AUPRC la nivel detecție (prevalența reală: FP din scanare completă)
dets.sort(reverse=True);tp=0;fp=0;prev_rec=0.0;ap=0.0;pr_pts=[]
for s,istp in dets:
    if istp: tp+=1
    else: fp+=1
    prec=tp/(tp+fp);rec=tp/nGT;ap+=prec*(rec-prev_rec);prev_rec=rec;pr_pts.append((s,prec,rec))
print(f"  AUPRC (real prevalence, detections over {area_km2:.0f}km²): {ap:.3f}")
# operating points
for thr in (0.5,0.7,0.9):
    tp=sum(1 for s,t in dets if t==1 and s>=thr);fp=sum(1 for s,t in dets if t==0 and s>=thr)
    rec=tp/nGT;prec=tp/(tp+fp) if (tp+fp) else 0
    print(f"  @{thr}: recall {tp}/{nGT}={rec*100:.0f}% | precision {prec*100:.0f}% | FP {fp} ({fp/area_km2:.1f}/km²) | F1 {2*prec*rec/(prec+rec+1e-9):.2f}")
# prag care ține recall 100%
rec100=[s for s,t in dets if t==1]
if rec100:
    thr100=min(rec100);fp100=sum(1 for s,t in dets if t==0 and s>=thr100);tp100=sum(1 for s,t in dets if t==1 and s>=thr100)
    print(f"  @recall100 (threshold {thr100:.2f}): {tp100}/{nGT} | FP {fp100} ({fp100/area_km2:.1f}/km²) | precision {100*tp100/(tp100+fp100):.0f}%")
