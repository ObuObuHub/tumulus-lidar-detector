#!/usr/bin/env python3
# validate_marks_dtm.py MARKED_IMG GEO_JSON [MODEL] [TH] — VALIDARE OARBĂ.
# Detectează cercurile VERZI (tumulii marcați de Andrei) pe imaginea returnată, le mapează la lon/lat via geo-transform
# (fracție din imagine -> robust la resize Telegram), apoi scorează modelul INDEPENDENT:
#  • RECALL: scor max în fereastră ±30m la fiecare tumul marcat (protocol detecție).
#  • PRECIZIE: sweep dens + NMS -> detecții >=TH; cele care NU-s lângă un marcaj = FP candidate.
# -> review/validate_marks_dtm.png (verde=marcaj, albastru=prins, roșu=FP) + recall/precizie la stdout.
import os,sys,math,subprocess,json
import numpy as np
from PIL import Image,ImageFilter,ImageDraw,ImageFont
import torch,torch.nn as nn
H=os.path.dirname(os.path.dirname(os.path.abspath(__file__)));dev=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
MARKED=sys.argv[1];GEO=json.load(open(sys.argv[2] if len(sys.argv)>2 else '/tmp/hs_dtm_geo.json'))
MODEL=sys.argv[3] if len(sys.argv)>3 else f'{H}/combined_cnn.pt';TH=float(sys.argv[4]) if len(sys.argv)>4 else 0.7
CACHE="/tmp/laki3";CS=GEO["CS"]
APP=os.environ.get("QGIS_APP","/Applications/QGIS-final-4_0_3.app/Contents")
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
# --- detectează cercuri verzi -> centroizi (fracție din imagine) ---
im=Image.open(MARKED).convert('RGB');MW,MH=im.size;a=np.asarray(im).astype(int)
Rr,Gg,Bb=a[:,:,0],a[:,:,1],a[:,:,2];green=(Gg>110)&(Gg-Rr>45)&(Gg-Bb>45)
if green.sum()==0: print("EROARE: niciun cerc verde detectat");sys.exit(1)
# dilatare (închide inelul) via roll, apoi componente conexe via union-find pe pixeli verzi (fără scipy)
g=green.copy()
for _ in range(max(3,int(MW*0.008))):
    g=g|np.roll(g,1,0)|np.roll(g,-1,0)|np.roll(g,1,1)|np.roll(g,-1,1)
gy,gx=np.where(g);idx={(int(x),int(y)):i for i,(x,y) in enumerate(zip(gx,gy))}
par=list(range(len(gx)))
def find(a):
    while par[a]!=a: par[a]=par[par[a]];a=par[a]
    return a
for i,(x,y) in enumerate(zip(gx.tolist(),gy.tolist())):
    for dx,dy in ((1,0),(0,1),(1,1),(-1,1)):
        j=idx.get((x+dx,y+dy))
        if j is not None: par[find(i)]=find(j)
from collections import defaultdict
groups=defaultdict(list)
for i in range(len(gx)): groups[find(i)].append(i)
cl=[]
for g2 in groups.values():
    if len(g2)<MW*0.0015*MW*0.0015*100: continue   # ignoră zgârieturi mici
    cl.append((float(gx[g2].mean()),float(gy[g2].mean())))
print(f"imagine marcată {MW}x{MH} | {len(cl)} cercuri verzi detectate",flush=True)
# --- mosaic laki3 (din geo crop) ---
east_left=GEO["east_left"];north_top=GEO["north_top"];cW=GEO["crop_W"];cH=GEO["crop_H"]
def mark_to_ll(cx,cy):
    fx=cx/MW;fy=cy/MH;e=east_left+fx*cW*CS;n=north_top-fy*cH*CS;lo,la=trans((e,n),"EPSG:3844","EPSG:4326");return e,n,lo,la
marks=[mark_to_ll(cx,cy)+(cx,cy) for cx,cy in cl]
# construiește mosaic acoperind crop-ul
est=east_left+cW*CS/2;nord=north_top-cH*CS/2;half=max(cW,cH)*CS/2+200
e0=int((est-half)//1000);e1=int((est+half)//1000);n0=int((nord-half)//1000);n1=int((nord+half)//1000)
TPX=2000;xll0=e0*1000;ytop0=(n1+1)*1000;W=(e1-e0+1)*TPX;Hh=(n1-n0+1)*TPX
mos=np.full((Hh,W),np.nan,np.float32)
for nk in range(n0,n1+1):
    for ek in range(e0,e1+1):
        d=dl(nk,ek)
        if d is None: continue
        ox=int((ek*1000-xll0)/CS);oy=int((ytop0-(nk+1)*1000)/CS);mos[oy:oy+TPX,ox:ox+TPX]=d[:TPX,:TPX]
f=int(round(2.0/CS));hw=int(40/CS)
class Net(nn.Module):
    def __init__(s):
        super().__init__();s.c=nn.Sequential(nn.Conv2d(1,16,3,2,1),nn.ReLU(),nn.Conv2d(16,32,3,2,1),nn.ReLU(),nn.Conv2d(32,64,3,2,1),nn.ReLU(),nn.AdaptiveAvgPool2d(1));s.f=nn.Linear(64,1)
    def forward(s,x): return s.f(s.c(x).flatten(1)).squeeze(1)
net=Net().to(dev);net.load_state_dict(torch.load(MODEL,map_location=dev,weights_only=True));net.eval()
def stamp_en(e,n):
    px=int((e-xll0)/CS);py=int((ytop0-n)/CS);w=mos[py-hw:py+hw,px-hw:px+hw]
    if w.shape!=(2*hw,2*hw) or np.isnan(w).mean()>0.05: return None
    d2=downs(w,f);h=hs(d2,CS*f);lo,hi=np.percentile(h,2),np.percentile(h,98)
    if hi-lo<1e-6: return None
    return homog(np.asarray(Image.fromarray(np.clip((h-lo)/(hi-lo)*255,0,255).astype('uint8')).resize((128,128)),np.uint8))
def score_batch(stamps):
    X=torch.tensor(np.array(stamps,dtype=np.uint8)).unsqueeze(1).float().to(dev)/255.
    with torch.no_grad(): return torch.sigmoid(net(X)).cpu().numpy()
# RECALL: scor max ±30m la fiecare marcaj
def recall_score(e,n):
    best=0.0
    sts=[];offs=[]
    for de in range(-30,31,15):
        for dn in range(-30,31,15):
            s=stamp_en(e+de,n+dn)
            if s is not None: sts.append(s);offs.append((de,dn))
    if not sts: return None
    return float(score_batch(sts).max())
rec=[]
for e,n,lo,la,cx,cy in marks:
    sc=recall_score(e,n);rec.append(sc)
hit=sum(1 for s in rec if s is not None and s>=TH)
print(f"\n=== RECALL (tumuli marcați de Andrei) ===")
for (e,n,lo,la,cx,cy),s in zip(marks,rec):
    tag="PRINS" if (s is not None and s>=TH) else "ratat"
    print(f"  {la:.5f},{lo:.5f}  scor {s if s is None else round(s,3)}  {tag}")
print(f"  RECALL: {hit}/{len(marks)} @>= {TH}")
# PRECIZIE: sweep dens + NMS pe crop
step=int(15/CS)
gpx=range(int((east_left-xll0)/CS)+hw,int((east_left+cW*CS-xll0)/CS)-hw,step)
gpy=range(int((ytop0-north_top)/CS)+hw,int((ytop0-(north_top-cH*CS))/CS)-hw,step)
sts=[];posn=[]
for py in gpy:
    for px in gpx:
        w=mos[py-hw:py+hw,px-hw:px+hw]
        if w.shape!=(2*hw,2*hw) or np.isnan(w).mean()>0.05: continue
        d2=downs(w,f);h=hs(d2,CS*f);lo2,hi2=np.percentile(h,2),np.percentile(h,98)
        if hi2-lo2<1e-6: continue
        sts.append(homog(np.asarray(Image.fromarray(np.clip((h-lo2)/(hi2-lo2)*255,0,255).astype('uint8')).resize((128,128)),np.uint8)));posn.append((px,py))
allsc=score_batch(sts);order=np.argsort(-allsc);kept=[]
for k in order:
    if allsc[k]<TH: break
    px,py=posn[k]
    if any((px-q[0])**2+(py-q[1])**2<(50/CS)**2 for q in kept): continue
    kept.append((px,py,float(allsc[k])))
# clasifică detecții: lângă un marcaj = matched, altfel FP
markpx=[(int((e-xll0)/CS),int((ytop0-n)/CS)) for e,n,lo,la,cx,cy in marks]
fp=[];matched=[]
for px,py,s in kept:
    near=any((px-mx)**2+(py-my)**2<(60/CS)**2 for mx,my in markpx)
    (matched if near else fp).append((px,py,s))
print(f"\n=== PRECIZIE (sweep oarb, NMS >= {TH}) ===")
print(f"  detecții total {len(kept)} | pe marcaje {len(matched)} | FP (departe de marcaje) {len(fp)}")
print(f"  precizie brută {len(matched)}/{len(kept)} = {100*len(matched)//max(1,len(kept))}%  (⚠ unele 'FP' în Catane pot fi movile nemarcate)")
# overlay
ov=Image.fromarray(np.clip((hs(np.nan_to_num(mos,nan=float(np.nanmin(mos))),CS)*255),0,255).astype('uint8')).convert('RGB')
dr=ImageDraw.Draw(ov)
for (e,n,lo,la,cx,cy),s in zip(marks,rec):
    mx=int((e-xll0)/CS);my=int((ytop0-n)/CS);col=(40,255,40) if (s is not None and s>=TH) else (255,160,40)
    dr.ellipse([mx-45,my-45,mx+45,my+45],outline=col,width=4)
for px,py,s in fp: dr.ellipse([px-35,py-35,px+35,py+35],outline=(255,40,40),width=3)
crop=ov.crop((int((east_left-xll0)/CS),int((ytop0-north_top)/CS),int((east_left+cW*CS-xll0)/CS),int((ytop0-(north_top-cH*CS))/CS)))
mxs=max(crop.size);sc2=min(1,1400/mxs);crop.resize((int(crop.size[0]*sc2),int(crop.size[1]*sc2))).save(f'{H}/review/validate_marks_dtm.png')
print(f"\n-> review/validate_marks_dtm.png (verde=prins, portocaliu=ratat, roșu=FP)")
