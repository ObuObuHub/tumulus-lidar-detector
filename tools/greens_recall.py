#!/usr/bin/env python3
# greens_recall.py — held-out recall CORECT: pt fiecare movilă unică din /tmp/greens_qa_map.csv,
# găsește VÂRFUL detectorului (peak model) lângă coordonată, apoi scorează filtrul de curbură ACOLO
# (exact ca producția — detecția se fixează pe vârf). -> /tmp/greens_recall.csv + print recall@praguri.
import os,sys,math,subprocess,csv,json
import numpy as np
from PIL import Image,ImageFilter
import torch,torch.nn as nn
H=os.path.dirname(os.path.dirname(os.path.abspath(__file__)));dev=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
CACHE="/tmp/laki3";CS=0.5;TPX=2000
sys.path.insert(0,f"{H}/tools")
import curv_features2 as CF   # analyze(est,nord) -> features dict (citește din cache)
import pyproj
_TF={}
def _tf(s,t):
    if (s,t) not in _TF:_TF[(s,t)]=pyproj.Transformer.from_crs(s,t,always_xy=True)
    return _TF[(s,t)]
def trans(pts,s,t):
    if not pts:return []
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
class Net(nn.Module):
    def __init__(s):
        super().__init__();s.c=nn.Sequential(nn.Conv2d(1,16,3,2,1),nn.ReLU(),nn.Conv2d(16,32,3,2,1),nn.ReLU(),nn.Conv2d(32,64,3,2,1),nn.ReLU(),nn.AdaptiveAvgPool2d(1));s.f=nn.Linear(64,1)
    def forward(s,x): return s.f(s.c(x).flatten(1)).squeeze(1)
MODELP=sys.argv[1] if len(sys.argv)>1 else f'{H}/combined_cnn.pt'
net=Net().to(dev);net.load_state_dict(torch.load(MODELP,map_location=dev,weights_only=True));net.eval()
gate=json.load(open(f'{H}/curv_gate.json'));FEATS=gate['feats']
gw=np.array(gate['w']);gb=gate['b'];gmu=np.array(gate['mu']);gsd=np.array(gate['sd'])
def gpred(feat):
    try:
        v=np.array([feat[f] for f in FEATS])
        if np.any(~np.isfinite(v)):return None
        z=(v-gmu)/gsd;return float(1/(1+np.exp(-(z@gw+gb))))
    except:return None
def load_win(est,nord,half_m=120):
    half=int(half_m/CS)
    e0=int((est-half_m)//1000);e1=int((est+half_m)//1000);n0=int((nord-half_m)//1000);n1=int((nord+half_m)//1000)
    xll=e0*1000;ytop=(n1+1)*1000;W=(e1-e0+1)*TPX;Hh=(n1-n0+1)*TPX;mos=np.full((Hh,W),np.nan,np.float32);got=False
    for nk in range(n0,n1+1):
        for ek in range(e0,e1+1):
            p=f"{CACHE}/{nk}_{ek}.npy"
            if os.path.exists(p): d=np.load(p);got=True;ox=int((ek*1000-xll)/CS);oy=int((ytop-(nk+1)*1000)/CS);mos[oy:oy+TPX,ox:ox+TPX]=d[:TPX,:TPX]
    if not got:return None
    return mos,xll,ytop
f=int(round(2.0/CS));hw=int(40/CS)
def peak(est,nord,search_m=70):
    win=load_win(est,nord);
    if win is None:return None
    mos,xll,ytop=win;cx=int((est-xll)/CS);cy=int((ytop-nord)/CS);sr=int(search_m/CS);step=int(12/CS)
    best=(-1,est,nord)
    batch=[];posl=[]
    for py in range(cy-sr,cy+sr+1,step):
        for px in range(cx-sr,cx+sr+1,step):
            w=mos[py-hw:py+hw,px-hw:px+hw]
            if w.shape!=(2*hw,2*hw) or np.isnan(w).mean()>0.05:continue
            d2=downs(np.nan_to_num(w,nan=np.nanmedian(w)),f);h=hs(d2,CS*f);lo,hi=np.percentile(h,2),np.percentile(h,98)
            if hi-lo<1e-6:continue
            batch.append(homog(np.asarray(Image.fromarray(np.clip((h-lo)/(hi-lo)*255,0,255).astype('uint8')).resize((128,128)),np.uint8)));posl.append((px,py))
    if not batch:return None
    X=torch.tensor(np.array(batch),dtype=torch.float32).unsqueeze(1).to(dev)/255.
    with torch.no_grad():sc=torch.sigmoid(net(X)).cpu().numpy()
    k=int(np.argmax(sc));px,py=posl[k];pe=xll+px*CS;pn=ytop-py*CS
    return float(sc[k]),pe,pn
INP=sys.argv[2] if len(sys.argv)>2 else '/tmp/greens_qa_map.csv'
rows=list(csv.DictReader(open(INP)))
for r in rows:
    r.setdefault('montage_idx',r.get('idx',''));r.setdefault('gate','0')
en=trans([(float(r['lon']),float(r['lat'])) for r in rows],"EPSG:4326","EPSG:3844")
out=[]
for i,(r,(e,n)) in enumerate(zip(rows,en)):
    pk=peak(e,n)
    if pk is None: out.append((r['montage_idx'],r['gate'],None,None));continue
    psc,pe,pn=pk;feat=CF.analyze(pe,pn);pg=gpred(feat) if feat else None
    out.append((r['montage_idx'],r['gate'],round(psc,3),round(pg,3) if pg is not None else None))
    if (i+1)%10==0:print(f"  {i+1}/{len(rows)}",flush=True)
with open('/tmp/greens_recall.csv','w',newline='') as fo:
    w=csv.writer(fo);w.writerow(['idx','click_gate','peak_model','peak_gate'])
    for o in out:w.writerow(o)
pgv=[o[3] for o in out if o[3] is not None]
clk=[float(o[1]) for o in out if o[3] is not None]
import numpy as np
pgv=np.array(pgv);clk=np.array(clk)
print(f"\n{len(pgv)} movile unice scorate la VÂRF (producție-fidel):")
for t in [0.70,0.66,0.50,0.40]:
    print(f"  prag {t}: reține {100*(pgv>=t).mean():.0f}% la vârf  (vs {100*(clk>=t).mean():.0f}% la click-coord)")
print(f"  median gate la vârf {np.median(pgv):.3f} vs la click {np.median(clk):.3f}")
