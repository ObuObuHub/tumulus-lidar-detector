#!/usr/bin/env python3
# ahn_score_points.py CLON CLAT KM [MODEL] — TEST RIGUROS punct-cu-punct pe AHN.
# Fetch AHN DTM 0.5m + hillshade NATIV 0.5m. Încarcă movilele OSM (/tmp/nl_barrows.json), găsește cele din
# tile, scorează modelul FIX pe fiecare (stamp multi-scară 28/32m, max) = POZITIVI. Scorează N puncte random
# >60m de orice movilă = CONTROL NEGATIV. Raport: distribuții + AUROC + %≥0.7 pozitivi vs control.
import sys,os,math,subprocess,json,random
import numpy as np
from PIL import Image,ImageFilter
H=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import torch,torch.nn as nn
dev=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
CLON=float(sys.argv[1]);CLAT=float(sys.argv[2]);KM=float(sys.argv[3]) if len(sys.argv)>3 else 1.0
MODEL=sys.argv[4] if len(sys.argv)>4 else f'{H}/combined_cnn.pt'
CS=0.5;SCALES=[28,32];MPP=0.5
APP=os.environ.get("QGIS_APP","/Applications/QGIS-final-4_0_3.app/Contents")
ENV=dict(os.environ,DYLD_FRAMEWORK_PATH=f"{APP}/Frameworks",PROJ_DATA=f"{APP}/Resources/qgis/proj",PROJ_LIB=f"{APP}/Resources/qgis/proj",GDAL_DATA=f"{APP}/Resources/qgis/gdal")
GT=f"{APP}/MacOS/gdaltransform";GTR=f"{APP}/MacOS/gdal_translate"
def trans_many(pts,s="EPSG:4326",t="EPSG:28992"):
    inp="".join(f"{lo} {la}\n" for la,lo in pts)
    r=subprocess.run([GT,"-s_srs",s,"-t_srs",t],input=inp,capture_output=True,text=True,env=ENV)
    return [tuple(map(float,l.split()[:2])) for l in r.stdout.strip().split("\n")] if r.stdout.strip() else []
est,nord=trans_many([(CLAT,CLON)])[0];half=KM*1000/2
x0,x1=est-half,est+half;y0,y1=nord-half,nord+half
url=(f"https://service.pdok.nl/rws/ahn/wcs/v1_0?service=WCS&version=2.0.1&request=GetCoverage"
     f"&coverageId=dtm_05m&subset=x({x0:.1f},{x1:.1f})&subset=y({y0:.1f},{y1:.1f})&format=image/tiff")
tif="/tmp/ahn_score.tif";asc="/tmp/ahn_score.asc"
print(f"fetch AHN {KM}km @ {CLON},{CLAT}...",flush=True)
subprocess.run(["curl","-s","--max-time","180","-o",tif,url],check=False)
if not os.path.exists(tif) or os.path.getsize(tif)<10000: sys.exit("EROARE fetch AHN")
if os.path.exists(asc): os.remove(asc)
subprocess.run([GTR,"-of","AAIGrid",tif,asc],capture_output=True,env=ENV)
L=open(asc).read().split('\n');hdr={};i=0
while i<len(L) and L[i].split() and L[i].split()[0].lower() in('ncols','nrows','xllcorner','yllcorner','cellsize','nodata_value'):
    k,v=L[i].split()[:2];hdr[k.lower()]=float(v);i+=1
nc,nr=int(hdr['ncols']),int(hdr['nrows']);xll=hdr['xllcorner'];yll=hdr['yllcorner'];ce=hdr['cellsize']
dem=np.fromstring(' '.join(L[i:]),sep=' ',dtype=np.float32)[:nc*nr].reshape(nr,nc)
dem[dem>1e30]=np.nan;dem[dem==hdr.get('nodata_value',-9999)]=np.nan
ytop=yll+nr*ce
dem=np.nan_to_num(dem,nan=float(np.nanmedian(dem)))
def hs(d,cs,azs=(315,45,135,225,270,0),alt=35):
    gy,gx=np.gradient(d,cs);sl=np.arctan(np.hypot(gx,gy));asp=np.arctan2(-gy,gx);o=np.zeros_like(d);ar=math.radians(alt)
    for az in azs: azr=math.radians(360-az+90);o+=np.clip(np.sin(ar)*np.cos(sl)+np.cos(ar)*np.sin(sl)*np.cos(azr-asp),0,1)
    return o/len(azs)
sh=hs(dem,ce);lo,hi=np.percentile(sh,2),np.percentile(sh,98)
A=np.clip((sh-lo)/(hi-lo+1e-9)*255,0,255).astype(np.uint8)  # hillshade nativ, imagine
Hh,Ww=A.shape
def homog(a):
    a2=np.asarray(Image.fromarray(a).filter(ImageFilter.GaussianBlur(0.8)),np.uint8);cdf=np.bincount(a2.ravel(),minlength=256).astype(np.float64).cumsum()
    return (cdf[a2]/cdf[-1]*255).astype(np.uint8) if cdf[-1]>0 else a2
def stamp(cx,cy,M):
    win=int(round(M/MPP));content=max(8,int(round(M/2.0)));h=win//2
    if cx-h<0 or cy-h<0 or cx+h>Ww or cy+h>Hh: return None
    w=A[cy-h:cy+h,cx-h:cx+h]
    if w.shape!=(2*h,2*h) or w.std()<3: return None
    plo,phi=np.percentile(w,2),np.percentile(w,98);a=np.clip((w-plo)/(phi-plo+1e-6),0,1)
    a2=np.asarray(Image.fromarray((a*255).astype('uint8')).resize((content,content)),np.uint8)
    return homog(np.asarray(Image.fromarray(a2).resize((128,128)),np.uint8))
class Net(nn.Module):
    def __init__(s):
        super().__init__();s.c=nn.Sequential(nn.Conv2d(1,16,3,2,1),nn.ReLU(),nn.Conv2d(16,32,3,2,1),nn.ReLU(),nn.Conv2d(32,64,3,2,1),nn.ReLU(),nn.AdaptiveAvgPool2d(1));s.f=nn.Linear(64,1)
    def forward(s,x): return s.f(s.c(x).flatten(1)).squeeze(1)
net=Net().to(dev);net.load_state_dict(torch.load(MODEL,map_location=dev,weights_only=True));net.eval()
def score_pt(px,py):
    best=0.0
    for M in SCALES:
        s=stamp(px,py,M)
        if s is None: continue
        with torch.no_grad():
            v=float(torch.sigmoid(net(torch.tensor(s[None,None],dtype=torch.float32).to(dev)/255.)).item())
        best=max(best,v)
    return best
# movile OSM din tile
barrows=json.load(open('/tmp/nl_barrows.json'))  # list [la,lo,nm]
inb=[(la,lo) for la,lo,_ in barrows if abs(la-CLAT)<KM/111.0 and abs(lo-CLON)<KM/(111.0*math.cos(math.radians(CLAT)))]
if not inb: print("NICIO movilă OSM în tile");
EN=trans_many(inb) if inb else []
bxy=[]
for (la,lo),(e,n) in zip(inb,EN):
    px=int((e-xll)/ce);py=int((ytop-n)/ce)
    if 0<=px<Ww and 0<=py<Hh: bxy.append((px,py))
pos=[score_pt(px,py) for px,py in bxy]
# control: puncte random >60m de orice movilă
bset=np.array(bxy) if bxy else np.zeros((0,2))
md=int(60/ce);ctrl=[];ctrl_xy=[];tries=0
random.seed(len(bxy)+nc)
while len(ctrl)<max(60,len(bxy)) and tries<5000:
    tries+=1;px=random.randint(80,Ww-80);py=random.randint(80,Hh-80)
    if len(bset) and np.min(np.hypot(bset[:,0]-px,bset[:,1]-py))<md: continue
    v=score_pt(px,py)
    if v is not None: ctrl.append(v);ctrl_xy.append((px,py))
# --- LINIARITATE (ideea Andrei: taie hiturile liniare din control) — replică linearity_test.py pe DEM AHN ---
def boxblur(a,r):
    ii=np.zeros((a.shape[0]+1,a.shape[1]+1));ii[1:,1:]=np.cumsum(np.cumsum(a,0),1)
    H2,W2=a.shape;ys=np.arange(H2);xs=np.arange(W2)
    y0=np.clip(ys-r,0,H2);y1=np.clip(ys+r+1,0,H2);x0=np.clip(xs-r,0,W2);x1=np.clip(xs+r+1,0,W2)
    return (ii[y1][:,x1]-ii[y0][:,x1]-ii[y1][:,x0]+ii[y0][:,x0])/((y1-y0)[:,None]*(x1-x0)[None,:])
LH=int(160/ce)
def cc_seed(mask,sy,sx):
    if not mask[sy,sx]:
        best=None
        for dy in range(-24,25):
            for dx in range(-24,25):
                y,x=sy+dy,sx+dx
                if 0<=y<mask.shape[0] and 0<=x<mask.shape[1] and mask[y,x]:
                    dd=dy*dy+dx*dx
                    if best is None or dd<best[0]: best=(dd,y,x)
        if best is None: return None
        sy,sx=best[1],best[2]
    region=np.zeros_like(mask);region[sy,sx]=True
    while True:
        d=region.copy();d[1:,:]|=region[:-1,:];d[:-1,:]|=region[1:,:];d[:,1:]|=region[:,:-1];d[:,:-1]|=region[:,1:];d&=mask
        if d.sum()==region.sum(): break
        region=d
    return region
def lin_ratio(px,py):
    if px-LH<0 or py-LH<0 or px+LH>Ww or py+LH>Hh: return 1.0
    w=dem[py-LH:py+LH,px-LH:px+LH]
    slrm=w-boxblur(w,int(200/ce));thr=slrm.mean()+1.0*slrm.std();mask=slrm>thr
    region=cc_seed(mask,LH,LH)
    if region is None or region.sum()<20: return 1.0
    ys,xs=np.nonzero(region);mx,my=xs.mean(),ys.mean()
    mxx=((xs-mx)**2).mean();myy=((ys-my)**2).mean();mxy=((xs-mx)*(ys-my)).mean()
    tr=mxx+myy;s=math.sqrt(max(0,tr*tr/4-(mxx*myy-mxy*mxy)));l1=tr/2+s;l2=tr/2-s
    return math.sqrt(l1/max(l2,1e-6))
# COERENȚĂ DIRECȚIONALĂ (structure tensor) — prinde arătură/șanț (gradienți paraleli) ≠ movilă (radial izotrop)
def coherence(px,py,rad_m=22):
    r=int(rad_m/ce)
    if px-r<0 or py-r<0 or px+r>Ww or py+r>Hh: return 0.0
    w=dem[py-r:py+r,px-r:px+r]
    gy,gx=np.gradient(w)
    Jxx=(gx*gx).mean();Jyy=(gy*gy).mean();Jxy=(gx*gy).mean()
    den=Jxx+Jyy
    if den<1e-12: return 0.0
    return math.sqrt((Jxx-Jyy)**2+4*Jxy*Jxy)/den  # [0,1]: 1=perfect direcțional, 0=izotrop
pos_lin=np.array([lin_ratio(px,py) for px,py in bxy])
ctrl_lin=np.array([lin_ratio(px,py) for px,py in ctrl_xy])
pos_coh=np.array([coherence(px,py) for px,py in bxy])
ctrl_coh=np.array([coherence(px,py) for px,py in ctrl_xy])
# DUMP features multi-rază pt tuning (env DUMP=1, append /tmp/nl_feat.csv)
if os.environ.get('DUMP'):
    import csv as _csv
    fn='/tmp/nl_feat.csv';new=not os.path.exists(fn)
    fo=open(fn,'a');wj=_csv.writer(fo)
    if new: wj.writerow(['site','label','score','coh15','coh22','coh30','coh40','lin'])
    SITE=os.environ.get('SITE','nl')
    for (px,py),sc_ in zip(bxy,[score_pt(px,py) for px,py in bxy]):
        wj.writerow([SITE,'pos',f'{sc_:.3f}',f'{coherence(px,py,15):.3f}',f'{coherence(px,py,22):.3f}',f'{coherence(px,py,30):.3f}',f'{coherence(px,py,40):.3f}',f'{lin_ratio(px,py):.2f}'])
    for (px,py),sc_ in zip(ctrl_xy,ctrl):
        wj.writerow([SITE,'neg',f'{sc_:.3f}',f'{coherence(px,py,15):.3f}',f'{coherence(px,py,22):.3f}',f'{coherence(px,py,30):.3f}',f'{coherence(px,py,40):.3f}',f'{lin_ratio(px,py):.2f}'])
    fo.close();print(f"  DUMP -> {fn} ({SITE})")
pos=np.array([p for p in pos if p is not None]);ctrl=np.array(ctrl)
def rep(name,a):
    if not len(a): print(f"  {name}: 0 puncte");return
    print(f"  {name}: n={len(a)} | mediană {np.median(a):.3f} | medie {a.mean():.3f} | %≥0.7 {(a>=0.7).mean()*100:.0f}% | %≥0.9 {(a>=0.9).mean()*100:.0f}%")
print(f"=== {os.path.basename(MODEL)} @ {CLON},{CLAT} ({KM}km, hillshade nativ {nc}px) ===")
rep("MOVILE OSM (pozitivi)",pos);rep("CONTROL random (negativi)",ctrl)
# AUROC simplu (Mann-Whitney)
if len(pos) and len(ctrl):
    allv=np.concatenate([pos,ctrl]);r=allv.argsort().argsort()+1
    auc=(r[:len(pos)].sum()-len(pos)*(len(pos)+1)/2)/(len(pos)*len(ctrl))
    print(f"  >>> AUROC movile-vs-control: {auc:.3f}  (0.5=la întâmplare, 1.0=separare perfectă)")
    print(f"  >>> separare medii: movile {pos.mean():.3f} vs control {ctrl.mean():.3f}  (Δ={pos.mean()-ctrl.mean():+.3f})")
    # --- GATING LINIARITATE: scor->0 dacă feature liniar (ratio>THR). Vede dacă scoate FP din control fără să piardă movile ---
    def auroc(p,c):
        a=np.concatenate([p,c]);rr=a.argsort().argsort()+1
        return (rr[:len(p)].sum()-len(p)*(len(p)+1)/2)/(len(p)*len(c))
    hp=(pos>=0.7);hc=(ctrl>=0.7)
    print(f"  --- liniaritate la hiturile ≥0.7: movile lin median {np.median(pos_lin[hp]) if hp.any() else float('nan'):.2f} (n={hp.sum()}) | control lin median {np.median(ctrl_lin[hc]) if hc.any() else float('nan'):.2f} (n={hc.sum()})")
    for THR in (3.0,4.0,6.0):
        pg=np.where(pos_lin>THR,0.0,pos);cg=np.where(ctrl_lin>THR,0.0,ctrl)
        rec=(pg>=0.7).mean()*100;fp=(cg>=0.7).mean()*100;ag=auroc(pg,cg)
        cut_p=(pos_lin>THR).mean()*100;cut_c=(ctrl_lin>THR).mean()*100
        print(f"  gating lin>{THR:.0f}: AUROC {ag:.3f} | recall@0.7 {rec:.0f}% | FP-control@0.7 {fp:.0f}% | tăiate movile {cut_p:.0f}% / control {cut_c:.0f}%")
    # --- COERENȚĂ DIRECȚIONALĂ (arătură/șanț): movilă=izotrop(coh mic), arătură/șanț=direcțional(coh mare) ---
    print(f"  --- coerență la hiturile ≥0.7: movile coh median {np.median(pos_coh[hp]) if hp.any() else float('nan'):.2f} | control coh median {np.median(ctrl_coh[hc]) if hc.any() else float('nan'):.2f}")
    for THR in (0.5,0.6,0.7):
        pg=np.where(pos_coh>THR,0.0,pos);cg=np.where(ctrl_coh>THR,0.0,ctrl)
        rec=(pg>=0.7).mean()*100;fp=(cg>=0.7).mean()*100;ag=auroc(pg,cg)
        cut_p=(pos_coh>THR).mean()*100;cut_c=(ctrl_coh>THR).mean()*100
        print(f"  gating coh>{THR:.1f}: AUROC {ag:.3f} | recall@0.7 {rec:.0f}% | FP-control@0.7 {fp:.0f}% | tăiate movile {cut_p:.0f}% / control {cut_c:.0f}%")
