#!/usr/bin/env python3
# ran_pass.py CLON CLAT KM [MODEL] [SCALES_m] — PASS al modelului peste o zonă LAKI3 (Oltenia 0.5m), cu
# descărcare automată tile-uri + heatmap multi-scară + marcaje RAN (env RAN_PTS="lon,lat,label;...").
# Raportează: scor la fiecare movilă RAN, statistici câmp, top detecții (NMS). -> review/ran_pass.png
import os,sys,math,subprocess,zipfile
import numpy as np
from PIL import Image,ImageFilter,ImageDraw,ImageFont
import torch,torch.nn as nn
H=os.path.dirname(os.path.dirname(os.path.abspath(__file__)));dev=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
CLON=float(sys.argv[1]);CLAT=float(sys.argv[2]);KM=float(sys.argv[3]) if len(sys.argv)>3 else 1.6
MODEL=sys.argv[4] if len(sys.argv)>4 else f'{H}/combined_cnn.pt'
SCALES=[float(x) for x in (sys.argv[5].split(',') if len(sys.argv)>5 else "48,64,80".split(','))]
CACHE="/tmp/laki3";CS=0.5;TPX=2000;os.makedirs(CACHE,exist_ok=True)
APP=os.environ.get("QGIS_APP","/Applications/QGIS-final-4_0_3.app/Contents")
ENV=dict(os.environ,DYLD_FRAMEWORK_PATH=f"{APP}/Frameworks",PROJ_DATA=f"{APP}/Resources/qgis/proj",PROJ_LIB=f"{APP}/Resources/qgis/proj",GDAL_DATA=f"{APP}/Resources/qgis/gdal")
GT=f"{APP}/MacOS/gdaltransform"
def trans(lon,lat):
    r=subprocess.run([GT,"-s_srs","EPSG:4326","-t_srs","EPSG:3844"],input=f"{lon} {lat}\n",capture_output=True,text=True,env=ENV);a,b,_=r.stdout.split();return float(a),float(b)
def load_one(nk,ek):  # download+extract LAKI3 tile (ca dtm_slrm)
    npy=f"{CACHE}/{nk}_{ek}.npy"
    if os.path.exists(npy): return np.load(npy)
    z=f"{CACHE}/{nk}_{ek}.zip"
    if not os.path.exists(z) or os.path.getsize(z)<1000:
        subprocess.run(["curl","-s","--max-time","90","-o",z,f"https://geoportal.ancpi.ro/laki3_mnt/zip/{nk}_{ek}.zip"],check=False)
    try: zf=zipfile.ZipFile(z)
    except: return None
    asc=[n for n in zf.namelist() if n.lower().endswith('.asc')]
    if not asc: return None
    raw=zf.read(asc[0]).decode('latin-1').replace(',','.');lines=raw.split('\n');hdr={};i=0
    while i<len(lines):
        p=lines[i].split()
        if len(p)>=2 and p[0].lower() in ('ncols','nrows','xllcorner','yllcorner','cellsize','nodata_value'): hdr[p[0].lower()]=float(p[1]);i+=1
        else: break
    nc=int(hdr['ncols']);nr=int(hdr['nrows']);nd=hdr.get('nodata_value',-9999)
    data=np.fromstring(' '.join(lines[i:]),sep=' ',dtype=np.float32)[:nc*nr].reshape(nr,nc);data[data==nd]=np.nan;np.save(npy,data);return data
def hs(dem,cs,azs=(315,45,135,225,270,0),alt=35):
    gy,gx=np.gradient(dem,cs);sl=np.arctan(np.hypot(gx,gy));asp=np.arctan2(-gy,gx);o=np.zeros_like(dem);ar=math.radians(alt)
    for az in azs: azr=math.radians(360-az+90);o+=np.clip(np.sin(ar)*np.cos(sl)+np.cos(ar)*np.sin(sl)*np.cos(azr-asp),0,1)
    return o/len(azs)
def downs(a,fc):
    Hh,Ww=a.shape;return a[:Hh//fc*fc,:Ww//fc*fc].reshape(Hh//fc,fc,Ww//fc,fc).mean((1,3))
def homog(a):
    a2=np.asarray(Image.fromarray(a).filter(ImageFilter.GaussianBlur(0.8)),np.uint8);cdf=np.bincount(a2.ravel(),minlength=256).astype(np.float64).cumsum()
    return (cdf[a2]/cdf[-1]*255).astype(np.uint8) if cdf[-1]>0 else a2
est,nord=trans(CLON,CLAT);half=KM*1000/2
e0=int((est-half)//1000);e1=int((est+half)//1000);n0=int((nord-half)//1000);n1=int((nord+half)//1000)
xll0=e0*1000;ytop0=(n1+1)*1000;W=(e1-e0+1)*TPX;Hh=(n1-n0+1)*TPX
mos=np.full((Hh,W),np.nan,np.float32);nt=0
for nk in range(n0,n1+1):
    for ek in range(e0,e1+1):
        d=load_one(nk,ek)
        if d is None: continue
        nt+=1;ox=int((ek*1000-xll0)/CS);oy=int((ytop0-(nk+1)*1000)/CS);mos[oy:oy+TPX,ox:ox+TPX]=d[:TPX,:TPX]
if nt==0: print(f"EROARE: niciun tile LAKI3 pt {CLON},{CLAT} (nk {n0}-{n1}, ek {e0}-{e1}) — probabil neacoperit");sys.exit(2)
print(f"mozaic {W}x{Hh}px ({KM}km, 0.5m), {nt} tile, model {os.path.basename(MODEL)}",flush=True)
def llpx(lon,lat):
    e,nn_=trans(lon,lat);return (e-xll0)/CS,(ytop0-nn_)/CS
f=int(round(2.0/CS));hw=int(40/CS)
class Net(nn.Module):
    def __init__(s):
        super().__init__();s.c=nn.Sequential(nn.Conv2d(1,16,3,2,1),nn.ReLU(),nn.Conv2d(16,32,3,2,1),nn.ReLU(),nn.Conv2d(32,64,3,2,1),nn.ReLU(),nn.AdaptiveAvgPool2d(1));s.f=nn.Linear(64,1)
    def forward(s,x): return s.f(s.c(x).flatten(1)).squeeze(1)
net=Net().to(dev);net.load_state_dict(torch.load(MODEL,map_location=dev,weights_only=True));net.eval()
def stamp(px,py,meters):
    h=int(meters/2/CS);w=mos[py-h:py+h,px-h:px+h]
    if w.shape!=(2*h,2*h) or np.isnan(w).mean()>0.05: return None
    d2=downs(w,f);sh=hs(d2,CS*f);lo,hi=np.percentile(sh,2),np.percentile(sh,98)
    if hi-lo<1e-6: return None
    return homog(np.asarray(Image.fromarray(np.clip((sh-lo)/(hi-lo)*255,0,255).astype('uint8')).resize((128,128)),np.uint8))
def score(stamps):
    if not stamps: return np.array([])
    X=torch.tensor(np.array(stamps,dtype=np.uint8));out=[]
    with torch.no_grad():
        for k in range(0,len(X),512): out.extend(torch.sigmoid(net(X[k:k+512].unsqueeze(1).float().to(dev)/255.)).cpu().numpy().tolist())
    return np.array(out)
STEP=int(10/CS)
gxs=list(range(hw,W-hw,STEP));gys=list(range(hw,Hh-hw,STEP))
best=np.zeros((len(gys),len(gxs)),np.float32)
for M in SCALES:
    batch=[];pos=[]
    for iy,py in enumerate(gys):
        for ix,px in enumerate(gxs):
            s=stamp(px,py,M)
            if s is not None: batch.append(s);pos.append((iy,ix))
    sc=score(batch);g=np.zeros_like(best)
    for (iy,ix),v in zip(pos,sc): g[iy,ix]=v
    best=np.maximum(best,g)
    if sc.size: print(f"  scară {M:.0f}m: %>=0.7 {(sc>=0.7).mean()*100:.1f}% | %>=0.9 {(sc>=0.9).mean()*100:.1f}%",flush=True)
v=best[best>0]
print(f"COMBINAT: mediană {np.median(v):.3f} | %>=0.7 {(v>=0.7).mean()*100:.1f}% | %>=0.9 {(v>=0.9).mean()*100:.1f}%",flush=True)
# detecții NMS din grid combinat
det=[]
flat=[(best[iy,ix],gxs[ix],gys[iy]) for iy in range(len(gys)) for ix in range(len(gxs)) if best[iy,ix]>=0.85]
flat.sort(reverse=True);kept0=[]
for s,px,py in flat:
    if all((px-q[1])**2+(py-q[2])**2>(80/CS)**2 for q in kept0): kept0.append((s,px,py))
# filtru coerență direcțională (tunat 24.06, coh22>0.70): taie arătură/șanț, Catane 91→100%, 0 movile pierdute
def coh22(px,py):
    r=int(22/CS);w=mos[py-r:py+r,px-r:px+r]
    if w.shape!=(2*r,2*r) or np.isnan(w).mean()>0.1: return 0.0
    w=np.nan_to_num(w,nan=np.nanmedian(w));gy2,gx2=np.gradient(w)
    Jxx=(gx2*gx2).mean();Jyy=(gy2*gy2).mean();Jxy=(gx2*gy2).mean();den=Jxx+Jyy
    return math.sqrt((Jxx-Jyy)**2+4*Jxy*Jxy)/den if den>1e-12 else 0.0
NOCOH=os.environ.get('NOCOH')  # NOCOH=1 dezactivează filtrul (debug)
kept=[(s,px,py) for s,px,py in kept0 if NOCOH or coh22(px,py)<=0.70]
print(f"detecții >=0.85 (NMS 80m): {len(kept0)} -> {len(kept)} după filtru coerență (-{len(kept0)-len(kept)} direcționale)",flush=True)
# filtru CURBURĂ (24.06): taie mușuroaiele compacte/aspre, păstrează domurile netede. Catane held-out 47->16 FP @recall100.
# CURVGATE=prag (ex 0.70) activează; folosește curv_gate.json + curv_features2 prin curv_filter.py (citește din cache).
CURVGATE=os.environ.get('CURVGATE')
if CURVGATE and kept:
    import csv as _csvg
    pts="".join(f"{xll0+px*CS} {ytop0-py*CS}\n" for s,px,py in kept)
    rr=subprocess.run([GT,"-s_srs","EPSG:3844","-t_srs","EPSG:4326"],input=pts,capture_output=True,text=True,env=ENV)
    lls=[l.split()[:2] for l in rr.stdout.strip().split("\n")]
    with open('/tmp/_curvg_in.csv','w',newline='') as f:
        w=_csvg.writer(f);w.writerow(['lon','lat'])
        for ll in lls: w.writerow([ll[0],ll[1]])
    subprocess.run([sys.executable,f"{H}/tools/curv_filter.py","/tmp/_curvg_in.csv","/tmp/_curvg_out.csv",f"{H}/curv_gate.json",CURVGATE],check=False)
    keepf=[r.get('keep')=='1' for r in _csvg.DictReader(open('/tmp/_curvg_out.csv'))]
    if len(keepf)==len(kept):
        nk=len(kept);kept=[k for k,ok in zip(kept,keepf) if ok]
        print(f"  -> {len(kept)} după filtru curbură (-{nk-len(kept)} mușuroaie/aspre, prag {CURVGATE})",flush=True)
# render
def jet(s):
    r=np.clip(1.5-np.abs(4*s-3),0,1);g=np.clip(1.5-np.abs(4*s-2),0,1);b=np.clip(1.5-np.abs(4*s-1),0,1)
    return np.stack([r,g,b],-1)
field=np.asarray(Image.fromarray((best*255).astype('uint8')).resize((W,Hh),Image.BICUBIC),np.float32)/255.
bgs=hs(np.nan_to_num(mos,nan=float(np.nanmin(mos))),CS);bg=np.clip((bgs-np.percentile(bgs,2))/(np.percentile(bgs,98)-np.percentile(bgs,2)+1e-6),0,1)
rgb=(np.stack([bg,bg,bg],-1)*255).astype(np.float32);col=jet(field)*255;alpha=(np.clip((field-0.5)/0.5,0,1)*0.75)[...,None]
out=(rgb*(1-alpha)+col*alpha).astype(np.uint8);img=Image.fromarray(out);dr=ImageDraw.Draw(img)
try: fnt=ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf",max(18,W//60))
except: fnt=ImageFont.load_default()
print("--- scor model la movilele RAN (max pe scări) ---")
for item in os.environ.get('RAN_PTS','').split(';'):
    if not item.strip(): continue
    lo,la,lab=item.split(',');lo=float(lo);la=float(la);px,py=llpx(lo,la)
    sc=max([score([stamp(int(px),int(py),M)])[0] if stamp(int(px),int(py),M) is not None else 0.0 for M in SCALES]) if 0<=px<W and 0<=py<Hh else float('nan')
    print(f"  {lab:22s} ({lo:.4f},{la:.4f}) -> {sc:.3f}")
    if 0<=px<W and 0<=py<Hh:
        r=int(50/CS);dr.ellipse([px-r,py-r,px+r,py+r],outline=(255,255,255),width=max(3,W//400));dr.text((px+r,py-r),lab,fill=(255,255,255),font=fnt)
os.makedirs(f"{H}/review",exist_ok=True);img.save(f"{H}/review/ran_pass.png");print(f"-> {H}/review/ran_pass.png")
if os.environ.get('DUMP'):
    import csv as _csv
    top=kept[:48]
    pts="".join(f"{xll0+px*CS} {ytop0-py*CS}\n" for s,px,py in top)
    rr=subprocess.run([GT,"-s_srs","EPSG:3844","-t_srs","EPSG:4326"],input=pts,capture_output=True,text=True,env=ENV)
    lls=[l.split()[:2] for l in rr.stdout.strip().split("\n")]
    with open('/tmp/ran_dets.csv','w',newline='') as fcsv, open('/tmp/lin_coords.csv','w',newline='') as flin:
        w=_csv.writer(fcsv);w.writerow(['idx','lon','lat','score']);wl=_csv.writer(flin);wl.writerow(['label','idx','lon','lat'])
        for i,((s,px,py),ll) in enumerate(zip(top,lls),1): w.writerow([i,ll[0],ll[1],round(s,3)]);wl.writerow(['G',i,ll[0],ll[1]])
    def hcrop(px,py,meters=160,out=150):
        h=int(meters/2/CS);w=mos[py-h:py+h,px-h:px+h]
        if w.shape!=(2*h,2*h) or np.isnan(w).mean()>0.1: return None
        sh=hs(np.nan_to_num(w,nan=float(np.nanmin(w))),CS);lo,hi=np.percentile(sh,2),np.percentile(sh,98)
        return np.asarray(Image.fromarray(np.clip((sh-lo)/(hi-lo+1e-6)*255,0,255).astype('uint8')).resize((out,out)),np.uint8)
    cols=8;rows=(len(top)+cols-1)//cols;c2=150;cv=Image.new('RGB',(cols*c2,rows*(c2+18)),(15,15,15));d2=ImageDraw.Draw(cv)
    try: f2=ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf",16)
    except: f2=ImageFont.load_default()
    for i,(s,px,py) in enumerate(top):
        cr=hcrop(px,py);x=(i%cols)*c2;y=(i//cols)*(c2+18)
        if cr is not None: cv.paste(Image.fromarray(cr),(x,y+18))
        d2.text((x+3,y+1),f"{i+1}",fill=(120,255,120),font=f2)
    cv.save(f"{H}/review/ran_board.png");print(f"-> {H}/review/ran_board.png ({len(top)} detecții, /tmp/ran_dets.csv)")
