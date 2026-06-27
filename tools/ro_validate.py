#!/usr/bin/env python3
# ro_validate.py — test RIGUROS pe ZONELE RO cu tumuli (analog ahn_score_points pt NL). Scor model + coerență
# la tumulii confirmați (gold_ran+Catane = nepărtinitor; +model_found+greens = broad) vs control random în
# aceleași tile-uri (>60m de orice tumul). AUROC baseline + cu filtru coh22>0.70. Cache LAKI3 local, rapid.
import os,sys,math,subprocess,csv,glob,random
import numpy as np
from PIL import Image,ImageFilter
import torch,torch.nn as nn
H=os.path.dirname(os.path.dirname(os.path.abspath(__file__)));dev=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
CACHE="/tmp/laki3";CS=0.5;TPX=2000
APP=os.environ.get("QGIS_APP","/Applications/QGIS-final-4_0_3.app/Contents")
ENV=dict(os.environ,DYLD_FRAMEWORK_PATH=f"{APP}/Frameworks",PROJ_DATA=f"{APP}/Resources/qgis/proj",PROJ_LIB=f"{APP}/Resources/qgis/proj",GDAL_DATA=f"{APP}/Resources/qgis/gdal")
GT=f"{APP}/MacOS/gdaltransform"
def trans_many(pts):
    inp="".join(f"{lo} {la}\n" for lo,la in pts)
    r=subprocess.run([GT,"-s_srs","EPSG:4326","-t_srs","EPSG:3844"],input=inp,capture_output=True,text=True,env=ENV)
    return [tuple(map(float,l.split()[:2])) for l in r.stdout.strip().split("\n")] if r.stdout.strip() else []
def hs(dem,cs,azs=(315,45,135,225,270,0),alt=35):
    gy,gx=np.gradient(dem,cs);sl=np.arctan(np.hypot(gx,gy));asp=np.arctan2(-gy,gx);o=np.zeros_like(dem);ar=math.radians(alt)
    for az in azs: azr=math.radians(360-az+90);o+=np.clip(np.sin(ar)*np.cos(sl)+np.cos(ar)*np.sin(sl)*np.cos(azr-asp),0,1)
    return o/len(azs)
def downs(a,fc):
    Hh,Ww=a.shape;return a[:Hh//fc*fc,:Ww//fc*fc].reshape(Hh//fc,fc,Ww//fc,fc).mean((1,3))
def homog(a):
    a2=np.asarray(Image.fromarray(a).filter(ImageFilter.GaussianBlur(0.8)),np.uint8);cdf=np.bincount(a2.ravel(),minlength=256).astype(np.float64).cumsum()
    return (cdf[a2]/cdf[-1]*255).astype(np.uint8) if cdf[-1]>0 else a2
_tc={}
def tile(nk,ek):
    k=(nk,ek)
    if k not in _tc:
        if len(_tc)>60: _tc.clear()
        p=f"{CACHE}/{nk}_{ek}.npy";_tc[k]=np.load(p) if os.path.exists(p) else None
    return _tc[k]
def window(est,nord,half):
    eks=sorted({int((est-half*CS)//1000),int((est+half*CS)//1000)});nks=sorted({int((nord-half*CS)//1000),int((nord+half*CS)//1000)})
    e0=min(eks);n1=max(nks);xll=e0*1000;ytop=(n1+1)*1000
    W=(max(eks)-e0+1)*TPX;Hh=(n1-min(nks)+1)*TPX;mos=np.full((Hh,W),np.nan,np.float32);got=False
    for nk in nks:
        for ek in eks:
            d=tile(nk,ek)
            if d is None: continue
            got=True;ox=int((ek*1000-xll)/CS);oy=int((ytop-(nk+1)*1000)/CS);mos[oy:oy+TPX,ox:ox+TPX]=d[:TPX,:TPX]
    if not got: return None
    px=int((est-xll)/CS);py=int((ytop-nord)/CS);return mos[py-half:py+half,px-half:px+half]
f=int(round(2.0/CS));hw=int(40/CS)
class Net(nn.Module):
    def __init__(s):
        super().__init__();s.c=nn.Sequential(nn.Conv2d(1,16,3,2,1),nn.ReLU(),nn.Conv2d(16,32,3,2,1),nn.ReLU(),nn.Conv2d(32,64,3,2,1),nn.ReLU(),nn.AdaptiveAvgPool2d(1));s.f=nn.Linear(64,1)
    def forward(s,x): return s.f(s.c(x).flatten(1)).squeeze(1)
net=Net().to(dev);net.load_state_dict(torch.load(f'{H}/combined_cnn.pt',weights_only=True));net.eval()
def score_coh(est,nord):
    w=window(est,nord,hw)
    if w is None or w.shape!=(2*hw,2*hw) or np.isnan(w).mean()>0.05: return None,None
    d2=downs(np.nan_to_num(w,nan=np.nanmedian(w)),f);h=hs(d2,CS*f);lo,hi=np.percentile(h,2),np.percentile(h,98)
    if hi-lo<1e-6: return None,None
    raw=np.asarray(Image.fromarray(np.clip((h-lo)/(hi-lo)*255,0,255).astype('uint8')).resize((128,128)),np.uint8)
    im=homog(raw)
    with torch.no_grad(): sc=float(torch.sigmoid(net(torch.tensor(im[None,None],dtype=torch.float32).to(dev)/255.)).item())
    # coerență rază 22m
    r=int(22/CS);cw=window(est,nord,r)
    coh=0.0
    if cw is not None and cw.shape==(2*r,2*r) and np.isnan(cw).mean()<0.1:
        cw=np.nan_to_num(cw,nan=np.nanmedian(cw));gy,gx=np.gradient(cw);Jxx=(gx*gx).mean();Jyy=(gy*gy).mean();Jxy=(gx*gy).mean();den=Jxx+Jyy
        coh=math.sqrt((Jxx-Jyy)**2+4*Jxy*Jxy)/den if den>1e-12 else 0.0
    return sc,coh
# pozitivi
def in_laki(lo,la): return 22.85<=lo<=24.05 and 43.78<=la<=44.55
TRUTH={11,17,18,30,43,45,50,55,57,64}
gold=[];broad=[]
em={int(r['idx']):(float(r['lon']),float(r['lat'])) for r in csv.DictReader(open('/tmp/eval_map.csv'))}
for i in TRUTH:
    if i in em: gold.append(em[i])
for r in csv.DictReader(open('labeled/labels.csv')):
    lo,la=float(r['lon']),float(r['lat'])
    if not in_laki(lo,la): continue
    if r['source']=='gold_ran': gold.append((lo,la))
    else: broad.append((lo,la))
for fn in glob.glob('labeled/batch*_greens.csv'):
    for r in csv.DictReader(open(fn)):
        lo,la=float(r['lon']),float(r['lat'])
        if in_laki(lo,la): broad.append((lo,la))
gold=list({(round(a,5),round(b,5)) for a,b in gold});broad=list({(round(a,5),round(b,5)) for a,b in broad})
print(f"pozitivi: GOLD(RAN+Catane)={len(gold)} | BROAD(+model/greens)={len(broad)}",flush=True)
allpos=gold+broad
ENp=trans_many(allpos);gold_en=ENp[:len(gold)];broad_en=ENp[len(gold):]
# control: random în jurul pozitivilor (aceleași zone), >60m de orice pozitiv
posset=np.array(ENp);random.seed(3);ctrl_en=[];tries=0
while len(ctrl_en)<300 and tries<20000:
    tries+=1;base=ENp[random.randrange(len(ENp))]
    e=base[0]+random.uniform(-700,700);n=base[1]+random.uniform(-700,700)
    if np.min(np.hypot(posset[:,0]-e,posset[:,1]-n))<60: continue
    ctrl_en.append((e,n))
def run(name,ens):
    out=[]
    for e,n in ens:
        sc,coh=score_coh(e,n)
        if sc is not None: out.append((sc,coh))
    return out
g=run('gold',gold_en);b=run('broad',broad_en);c=run('ctrl',ctrl_en)
print(f"scorate: gold={len(g)} broad={len(b)} ctrl={len(c)}",flush=True)
def auroc(P,N):
    a=np.array(P+N);rr=a.argsort().argsort()+1;return (rr[:len(P)].sum()-len(P)*(len(P)+1)/2)/(len(P)*len(N)) if P and N else float('nan')
def rep(tag,P,C):
    ps=[s for s,_ in P];cs=[s for s,_ in C]
    base=auroc(ps,cs)
    # filtru coh>0.70: scor->0 dacă direcțional
    pf=[0.0 if ch>0.70 else s for s,ch in P];cf=[0.0 if ch>0.70 else s for s,ch in C]
    af=auroc(pf,cf)
    rec=np.mean([s>=0.7 for s in ps])*100;fp=np.mean([s>=0.7 for s in cs])*100
    recf=np.mean([s>=0.7 for s in pf])*100;fpf=np.mean([s>=0.7 for s in cf])*100
    plost=np.mean([ch>0.70 for s,ch in P])*100;csup=np.mean([ch>0.70 for s,ch in C if s>=0.7])*100 if any(s>=0.7 for s,_ in C) else 0
    print(f"\n=== {tag} (n_poz={len(P)}, n_ctrl={len(C)}) ===")
    print(f"  mediană scor: tumuli {np.median(ps):.3f} vs control {np.median(cs):.3f}")
    print(f"  AUROC: baseline {base:.3f} -> cu filtru coh>0.70 {af:.3f}")
    print(f"  recall@0.7: {rec:.0f}% -> {recf:.0f}%  (tumuli tăiați de filtru: {plost:.0f}%)")
    print(f"  FP-control@0.7: {fp:.0f}% -> {fpf:.0f}%  (din FP@0.7, filtrul taie {csup:.0f}%)")
rep("GOLD (RAN+Catane, nepărtinitor)",g,c)
rep("BROAD (+model-found+greens)",b,c)
