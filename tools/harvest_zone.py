#!/usr/bin/env python3
# harvest_zone.py MARKED_IMG GEO_JSON [MODEL] [TH] [NEGPREFIX] — VALIDARE + HARVEST pe o zonă LAKI3 (non-Catane).
# 1) detectează cercurile verzi (tumulii lui Andrei) -> coords; 2) RECALL multi-scară la fiecare (max peste scale ±30m);
# 3) SWEEP multi-scară + NMS -> detecții; cele DEPARTE de cercuri = FP; 4) decupează FP ca NEGATIVE (recipe training
# 80m RAW, ca extract_marked) în dataset_neg_heatmap_<prefix>/ + manifest. Cercurile -> /tmp/<prefix>_pos.csv (pt
# adăugare ulterioară ca pozitivi). -> review/harvest_<prefix>.png (verde=prins, portocaliu=ratat, roșu=FP harvestat).
import os,sys,math,subprocess,json,csv
import numpy as np
from PIL import Image,ImageFilter,ImageDraw
import torch,torch.nn as nn
H=os.path.expanduser('~/lidar-match');dev=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
MARKED=sys.argv[1];GEO=json.load(open(sys.argv[2]));MODEL=sys.argv[3] if len(sys.argv)>3 else f'{H}/combined_cnn.pt'
TH=float(sys.argv[4]) if len(sys.argv)>4 else 0.7;PREFIX=sys.argv[5] if len(sys.argv)>5 else 'zone2'
SCALES=GEO.get("scales",[40,52,68,88,115]);CACHE="/tmp/laki3";CS=GEO["CS"]
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
def homog(a):
    a2=np.asarray(Image.fromarray(a).filter(ImageFilter.GaussianBlur(0.8)),np.uint8);cdf=np.bincount(a2.ravel(),minlength=256).astype(np.float64).cumsum()
    return (cdf[a2]/cdf[-1]*255).astype(np.uint8) if cdf[-1]>0 else a2
# --- cercuri verzi (union-find pe masca dilatată) ---
im=Image.open(MARKED).convert('RGB');MW,MH=im.size;a=np.asarray(im).astype(int)
green=(a[:,:,1]>110)&(a[:,:,1]-a[:,:,0]>45)&(a[:,:,1]-a[:,:,2]>45)
if green.sum()==0: print("EROARE: niciun cerc verde");sys.exit(1)
g=green.copy()
for _ in range(max(3,int(MW*0.008))): g=g|np.roll(g,1,0)|np.roll(g,-1,0)|np.roll(g,1,1)|np.roll(g,-1,1)
gy,gx=np.where(g);idx={(int(x),int(y)):i for i,(x,y) in enumerate(zip(gx,gy))};par=list(range(len(gx)))
def find(x):
    while par[x]!=x: par[x]=par[par[x]];x=par[x]
    return x
for i,(x,y) in enumerate(zip(gx.tolist(),gy.tolist())):
    for dx,dy in ((1,0),(0,1),(1,1),(-1,1)):
        j=idx.get((x+dx,y+dy))
        if j is not None: par[find(i)]=find(j)
from collections import defaultdict
grp=defaultdict(list)
for i in range(len(gx)): grp[find(i)].append(i)
cl=[(float(gx[v].mean()),float(gy[v].mean())) for v in grp.values() if len(v)>=MW*0.0015*MW*0.0015*100]
print(f"imagine {MW}x{MH} | {len(cl)} cercuri verzi | scări {SCALES}m",flush=True)
EL=GEO["east_left"];NT=GEO["north_top"];cW=GEO["crop_W"];cH=GEO["crop_H"]
marks=[(EL+(cx/MW)*cW*CS, NT-(cy/MH)*cH*CS) for cx,cy in cl]
# --- mosaic ---
est=EL+cW*CS/2;nord=NT-cH*CS/2;half=max(cW,cH)*CS/2+max(SCALES)+200
e0=int((est-half)//1000);e1=int((est+half)//1000);n0=int((nord-half)//1000);n1=int((nord+half)//1000)
TPX=2000;xll0=e0*1000;ytop0=(n1+1)*1000;W=(e1-e0+1)*TPX;Hh=(n1-n0+1)*TPX
mos=np.full((Hh,W),np.nan,np.float32)
for nk in range(n0,n1+1):
    for ek in range(e0,e1+1):
        d=dl(nk,ek)
        if d is None: continue
        ox=int((ek*1000-xll0)/CS);oy=int((ytop0-(nk+1)*1000)/CS);mos[oy:oy+TPX,ox:ox+TPX]=d[:TPX,:TPX]
f=int(round(2.0/CS))
class Net(nn.Module):
    def __init__(s):
        super().__init__();s.c=nn.Sequential(nn.Conv2d(1,16,3,2,1),nn.ReLU(),nn.Conv2d(16,32,3,2,1),nn.ReLU(),nn.Conv2d(32,64,3,2,1),nn.ReLU(),nn.AdaptiveAvgPool2d(1));s.f=nn.Linear(64,1)
    def forward(s,x): return s.f(s.c(x).flatten(1)).squeeze(1)
net=Net().to(dev);net.load_state_dict(torch.load(MODEL,map_location=dev,weights_only=True));net.eval()
def stamp_homog(px,py,M):  # pt scoring
    hw=int(M/2/CS);w=mos[py-hw:py+hw,px-hw:px+hw]
    if w.shape!=(2*hw,2*hw) or np.isnan(w).mean()>0.05: return None
    d2=downs(w,f);h=hs(d2,CS*f);lo,hi=np.percentile(h,2),np.percentile(h,98)
    if hi-lo<1e-6: return None
    return homog(np.asarray(Image.fromarray(np.clip((h-lo)/(hi-lo)*255,0,255).astype('uint8')).resize((128,128)),np.uint8))
def neg_stamp(px,py):  # pt SALVARE (recipe training 80m RAW, fără homog)
    wpx=int(80/CS);w=mos[py-wpx//2:py+wpx//2,px-wpx//2:px+wpx//2]
    if w.shape!=(wpx,wpx) or np.isnan(w).mean()>0.05: return None
    d2=downs(np.nan_to_num(w,nan=np.nanmedian(w)),f);h=hs(d2,CS*f);lo,hi=np.percentile(h,2),np.percentile(h,98)
    if hi-lo<1e-6: return None
    return np.asarray(Image.fromarray(np.clip((h-lo)/(hi-lo)*255,0,255).astype('uint8')).resize((128,128)),np.uint8)
def sb(stamps):
    X=torch.tensor(np.array(stamps,dtype=np.uint8)).unsqueeze(1).float().to(dev)/255.
    with torch.no_grad(): return torch.sigmoid(net(X)).cpu().numpy()
# RECALL multi-scară
def rec_ms(e,n):
    px0=int((e-xll0)/CS);py0=int((ytop0-n)/CS);sts=[]
    for M in SCALES:
        for de in range(-30,31,15):
            for dn in range(-30,31,15):
                s=stamp_homog(px0+int(de/CS),py0+int(dn/CS),M)
                if s is not None: sts.append(s)
    return float(sb(sts).max()) if sts else None
rec=[rec_ms(e,n) for e,n in marks]
hit=sum(1 for s in rec if s is not None and s>=TH)
print(f"\n=== RECALL multi-scară (tumulii marcați) ===")
for (e,n),s in zip(marks,rec):
    lo,la=trans((e,n),"EPSG:3844","EPSG:4326");print(f"  {la:.5f},{lo:.5f}  {('%.3f'%s) if s is not None else 'NA'}  {'PRINS' if (s and s>=TH) else 'ratat'}")
print(f"  RECALL: {hit}/{len(marks)} @>= {TH}")
# SWEEP multi-scară -> detecții
X0=int((EL-xll0)/CS);Y0=int((NT-ytop0)/-CS);step=int(12/CS)
gxs=range(X0,X0+cW,step);gys=range(Y0,Y0+cH,step)
best={}
for M in SCALES:
    sts=[];ps=[]
    for py in gys:
        for px in gxs:
            s=stamp_homog(px,py,M)
            if s is not None: sts.append(s);ps.append((px,py))
    sc=sb(sts)
    for (px,py),v in zip(ps,sc):
        if v>best.get((px,py),0): best[(px,py)]=float(v)
items=sorted(best.items(),key=lambda kv:-kv[1]);kept=[]
for (px,py),v in items:
    if v<TH: break
    if any((px-q[0])**2+(py-q[1])**2<(50/CS)**2 for q in kept): continue
    kept.append((px,py,v))
markpx=[(int((e-xll0)/CS),int((ytop0-n)/CS)) for e,n in marks]
fp=[(px,py,v) for px,py,v in kept if not any((px-mx)**2+(py-my)**2<(70/CS)**2 for mx,my in markpx)]
matched=len(kept)-len(fp)
print(f"\n=== SWEEP multi-scară (NMS >= {TH}) ===\n  detecții {len(kept)} | pe marcaje {matched} | FP (de harvestat) {len(fp)}")
# HARVEST FP -> negative
outdir=f'{H}/dataset_neg_heatmap_{PREFIX}';os.makedirs(outdir,exist_ok=True)
mf=open(f'{outdir}/manifest.csv','w');mw=csv.writer(mf);mw.writerow(['file','est','nord','score']);nw=0
for px,py,v in fp:
    st=neg_stamp(px,py)
    if st is None: continue
    e=xll0+px*CS;n=ytop0-py*CS;fn=f"hmneg_{PREFIX}_{nw:04d}.png";Image.fromarray(st).save(f"{outdir}/{fn}");mw.writerow([fn,f"{e:.1f}",f"{n:.1f}",f"{v:.3f}"]);nw+=1
mf.close()
print(f"  HARVEST: {nw} negative -> {outdir}/")
# salvează cercurile (pt adăugare pozitivi ulterior)
pf=open(f'/tmp/{PREFIX}_pos.csv','w');pw=csv.writer(pf);pw.writerow(['lon','lat','recall_score']);
for (e,n),s in zip(marks,rec): lo,la=trans((e,n),"EPSG:3844","EPSG:4326");pw.writerow([f"{lo:.6f}",f"{la:.6f}",('%.3f'%s) if s is not None else 'NA'])
pf.close();print(f"  cercuri salvate -> /tmp/{PREFIX}_pos.csv ({len(marks)} pozitivi candidați)")
# overlay
ov=Image.fromarray(np.clip(hs(np.nan_to_num(mos,nan=float(np.nanmin(mos))),CS)*255,0,255).astype('uint8')).convert('RGB');dr=ImageDraw.Draw(ov)
for (e,n),s in zip(marks,rec):
    mx=int((e-xll0)/CS);my=int((ytop0-n)/CS);col=(40,255,40) if (s and s>=TH) else (255,160,40)
    dr.ellipse([mx-40,my-40,mx+40,my+40],outline=col,width=4)
for px,py,v in fp: dr.ellipse([px-30,py-30,px+30,py+30],outline=(255,40,40),width=3)
crop=ov.crop((X0,Y0,X0+cW,Y0+cH));s2=min(1,1400/max(crop.size));crop.resize((int(crop.size[0]*s2),int(crop.size[1]*s2))).save(f'{H}/review/harvest_{PREFIX}.png')
print(f"\n-> review/harvest_{PREFIX}.png (verde=prins, portocaliu=ratat, roșu=FP harvestat)")
