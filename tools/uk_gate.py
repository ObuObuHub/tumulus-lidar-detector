#!/usr/bin/env python3
# uk_gate.py CLON CLAT KM [MODEL] — test GENERALIZARE pe UK (țară nouă, LiDAR 1m EA). Model + gate scale-adaptiv pe
# round barrows OSM (/tmp/uk_barrows.json) vs control. EA WCS DTM 1m (EPSG:27700), transform OSGB pur-python (Helmert+TM).
import sys,os,math,subprocess,json,random
import numpy as np
from PIL import Image,ImageFilter
import torch,torch.nn as nn
H=os.path.dirname(os.path.dirname(os.path.abspath(__file__)));dev=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
CLON=float(sys.argv[1]);CLAT=float(sys.argv[2]);KM=float(sys.argv[3]) if len(sys.argv)>3 else 2.0
MODEL=sys.argv[4] if len(sys.argv)>4 else f'{H}/combined_cnn.pt'
MPP=1.0;SCALES=[80];CID="13787b9a-26a4-4775-8523-806d13af58fc__Lidar_Composite_Elevation_DTM_1m"  # 80m = fereastra FIXĂ a antrenamentului (corecție Andrei 25.06): single-scale, NU multi-scală
APP=os.environ.get("QGIS_APP","/Applications/QGIS-final-4_0_3.app/Contents")
ENV=dict(os.environ,DYLD_FRAMEWORK_PATH=f"{APP}/Frameworks",PROJ_DATA=f"{APP}/Resources/qgis/proj",PROJ_LIB=f"{APP}/Resources/qgis/proj",GDAL_DATA=f"{APP}/Resources/qgis/gdal")
GTR=f"{APP}/MacOS/gdal_translate"
def osgb(lat_deg,lon_deg):
    aw=6378137.0;bw=6356752.3142;e2w=1-bw*bw/(aw*aw);lat=math.radians(lat_deg);lon=math.radians(lon_deg)
    nu=aw/math.sqrt(1-e2w*math.sin(lat)**2);x=nu*math.cos(lat)*math.cos(lon);y=nu*math.cos(lat)*math.sin(lon);z=(1-e2w)*nu*math.sin(lat)
    tx,ty,tz=-446.448,125.157,-542.060;s=-20.4894e-6
    rx=math.radians(-0.1502/3600);ry=math.radians(-0.2470/3600);rz=math.radians(-0.8421/3600)
    x2=tx+(1+s)*(x-rz*y+ry*z);y2=ty+(1+s)*(rz*x+y-rx*z);z2=tz+(1+s)*(-ry*x+rx*y+z)
    a=6377563.396;b=6356256.909;e2=1-b*b/(a*a);p=math.sqrt(x2*x2+y2*y2);lat2=math.atan2(z2,p*(1-e2))
    for _ in range(12):
        nu2=a/math.sqrt(1-e2*math.sin(lat2)**2);lat2=math.atan2(z2+e2*nu2*math.sin(lat2),p)
    lon2=math.atan2(y2,x2)
    F0=0.9996012717;lat0=math.radians(49);lon0=math.radians(-2);E0=400000;N0=-100000;n=(a-b)/(a+b)
    nu2=a*F0/math.sqrt(1-e2*math.sin(lat2)**2);rho=a*F0*(1-e2)/(1-e2*math.sin(lat2)**2)**1.5;eta2=nu2/rho-1
    M=b*F0*((1+n+1.25*n*n+1.25*n**3)*(lat2-lat0)-(3*n+3*n*n+2.625*n**3)*math.sin(lat2-lat0)*math.cos(lat2+lat0)+(1.875*n*n+1.875*n**3)*math.sin(2*(lat2-lat0))*math.cos(2*(lat2+lat0))-(35/24*n**3)*math.sin(3*(lat2-lat0))*math.cos(3*(lat2+lat0)))
    sl=math.sin(lat2);cl=math.cos(lat2);tl=math.tan(lat2)
    I=M+N0;II=nu2/2*sl*cl;III=nu2/24*sl*cl**3*(5-tl**2+9*eta2);IIIA=nu2/720*sl*cl**5*(61-58*tl**2+tl**4)
    IV=nu2*cl;Vv=nu2/6*cl**3*(nu2/rho-tl**2);VI=nu2/120*cl**5*(5-18*tl**2+tl**4+14*eta2-58*tl**2*eta2)
    dl=lon2-lon0;return E0+IV*dl+Vv*dl**3+VI*dl**5, I+II*dl**2+III*dl**4+IIIA*dl**6
cE,cN=osgb(CLAT,CLON);half=KM*1000/2
x0,x1=cE-half,cE+half;y0,y1=cN-half,cN+half
url=(f"https://environment.data.gov.uk/spatialdata/lidar-composite-digital-terrain-model-dtm-1m/wcs?service=WCS&version=2.0.1"
     f"&request=GetCoverage&coverageId={CID}&subset=E({x0:.0f},{x1:.0f})&subset=N({y0:.0f},{y1:.0f})&format=image/tiff")
tif="/tmp/uk_g.tif";asc="/tmp/uk_g.asc"
print(f"fetch UK EA DTM 1m {KM}km @ {CLON},{CLAT} (E{cE:.0f} N{cN:.0f})...",flush=True)
subprocess.run(["curl","-s","--max-time","180","-o",tif,url],check=False)
if not os.path.exists(tif) or os.path.getsize(tif)<10000: sys.exit(f"EROARE fetch UK DTM ({os.path.getsize(tif) if os.path.exists(tif) else 0}b)")
if os.path.exists(asc): os.remove(asc)
subprocess.run([GTR,"-q","-of","AAIGrid",tif,asc],env=ENV,check=False)
L=open(asc).read().split('\n');hdr={};i=0
while i<len(L) and L[i].split() and L[i].split()[0].lower() in('ncols','nrows','xllcorner','yllcorner','cellsize','nodata_value'):
    k,v=L[i].split()[:2];hdr[k.lower()]=float(v);i+=1
nc,nr=int(hdr['ncols']),int(hdr['nrows']);xll=hdr['xllcorner'];yll=hdr['yllcorner'];ce=hdr['cellsize'];ytop=yll+nr*ce
DEM=np.fromstring(' '.join(L[i:]),sep=' ',dtype=np.float32)[:nc*nr].reshape(nr,nc)
DEM[DEM<-1000]=np.nan;DEM[DEM>1e30]=np.nan;DEM=np.nan_to_num(DEM,nan=float(np.nanmedian(DEM)))
print(f"DTM {nc}x{nr} ce={ce}m",flush=True)
def hs(d,cs,azs=(315,45,135,225,270,0),alt=35):
    gy,gx=np.gradient(d,cs);sl=np.arctan(np.hypot(gx,gy));asp=np.arctan2(-gy,gx);o=np.zeros_like(d);ar=math.radians(alt)
    for az in azs: azr=math.radians(360-az+90);o+=np.clip(np.sin(ar)*np.cos(sl)+np.cos(ar)*np.sin(sl)*np.cos(azr-asp),0,1)
    return o/len(azs)
def downs(a,fc):
    Hh,Ww=a.shape;return a[:Hh//fc*fc,:Ww//fc*fc].reshape(Hh//fc,fc,Ww//fc,fc).mean((1,3))
sh=hs(DEM,ce);lo,hi=np.percentile(sh,2),np.percentile(sh,98);A=np.clip((sh-lo)/(hi-lo+1e-9)*255,0,255).astype(np.uint8);Hh,Ww=A.shape
def homog(a):
    a2=np.asarray(Image.fromarray(a).filter(ImageFilter.GaussianBlur(0.8)),np.uint8);cdf=np.bincount(a2.ravel(),minlength=256).astype(np.float64).cumsum()
    return (cdf[a2]/cdf[-1]*255).astype(np.uint8) if cdf[-1]>0 else a2
def stamp(cx,cy,M):
    win=int(round(M/MPP));h=win//2
    if cx-h<0 or cy-h<0 or cx+h>Ww or cy+h>Hh: return None
    w=DEM[cy-h:cy+h,cx-h:cx+h]
    if w.shape!=(2*h,2*h): return None
    # MATCH EXACT ANTRENAMENT (neg_stamp): DEM 80m -> downs la 2m efectiv -> hillshade@2m -> norm -> 128 -> homog.
    fu=max(1,int(round(2.0/ce)));d2=downs(np.nan_to_num(w,nan=float(np.nanmedian(w))),fu);h2=hs(d2,ce*fu)
    lo2,hi2=np.percentile(h2,2),np.percentile(h2,98)
    if hi2-lo2<1e-6: return None
    return homog(np.asarray(Image.fromarray(np.clip((h2-lo2)/(hi2-lo2)*255,0,255).astype('uint8')).resize((128,128)),np.uint8))
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
        with torch.no_grad(): v=float(torch.sigmoid(net(torch.tensor(s[None,None],dtype=torch.float32).to(dev)/255.)).item())
        best=max(best,v)
    return best
gate=json.load(open(f'{H}/curv_gate.json'));FEATS=gate['feats'];gw=np.array(gate['w']);gb=gate['b'];gmu=np.array(gate['mu']);gsd=np.array(gate['sd'])
def boxblur1(a,r):
    Hh2,Ww2=a.shape;ii=np.zeros((Hh2+1,Ww2+1),np.float64);ii[1:,1:]=np.cumsum(np.cumsum(a,0),1)
    ys=np.arange(Hh2);xs=np.arange(Ww2);yA=np.clip(ys-r,0,Hh2);yB=np.clip(ys+r+1,0,Hh2);xA=np.clip(xs-r,0,Ww2);xB=np.clip(xs+r+1,0,Ww2)
    S=ii[yB][:,xB]-ii[yA][:,xB]-ii[yB][:,xA]+ii[yA][:,xA];cnt=((yB-yA)[:,None]*(xB-xA)[None,:]).astype(np.float64);return (S/cnt).astype(np.float32)
def _fit(rc,prof):
    best=(1e9,8.0)
    for s in np.linspace(3,40,38):
        g=np.exp(-(rc**2)/(2*s*s));Mm=np.c_[g,np.ones(len(g))];sol,_,_,_=np.linalg.lstsq(Mm,prof,rcond=None);ss=float(((prof-Mm@sol)**2).sum())
        if ss<best[0]:best=(ss,s)
    return best
def _rad(res,rad,cap,rmax):
    e=np.arange(0,rmax+2,2);pr=[];rc=[]
    for k in range(len(e)-1):
        m=(rad>=e[k])&(rad<e[k+1])&cap
        if m.sum()>=3:pr.append(float(res[m].mean()));rc.append((e[k]+e[k+1])/2)
    return np.array(rc),np.array(pr)
def gate_feat(px,py):
    half=int(110/ce)
    if px-half<0 or py-half<0 or px+half>Ww or py+half>Hh: return None
    w=DEM[py-half:py+half,px-half:px+half]
    if w.shape!=(2*half,2*half): return None
    f=int(round(2.0/ce));Hc=w.shape[0]//f*f;z=w[:Hc,:Hc].reshape(Hc//f,f,Hc//f,f).mean((1,3));cs2=ce*f
    c0=z.shape[0]//2;rr=int(16/cs2);sub=z[c0-rr:c0+rr,c0-rr:c0+rr];off=np.unravel_index(np.argmax(sub),sub.shape);ay=c0-rr+off[0];ax=c0-rr+off[1]
    ys,xs=np.mgrid[0:z.shape[0],0:z.shape[1]].astype(np.float64);rad=np.hypot(ys-ay,xs-ax)*cs2
    bg0=(rad>=30)&(rad<=52)
    if bg0.sum()<20: return None
    A0=np.c_[xs[bg0].ravel(),ys[bg0].ravel(),np.ones(bg0.sum())];c0f,_,_,_=np.linalg.lstsq(A0,z[bg0].ravel(),rcond=None)
    res0=z-(c0f[0]*xs+c0f[1]*ys+c0f[2]);rc0,p0=_rad(res0,rad,rad<=45,45)
    if len(p0)<4: return None
    _,sigma=_fit(rc0,p0);s=float(np.clip(sigma,3,35));R=float(np.clip(2.2*s,10,48));Rin=float(np.clip(0.9*s,4,22))
    bgi=float(np.clip(R+6,R+6,52));bgo=float(np.clip(R+24,bgi+6,54));bg=(rad>=bgi)&(rad<=bgo)
    if bg.sum()<15: bg=bg0
    M=np.c_[xs[bg].ravel(),ys[bg].ravel(),np.ones(bg.sum())];co,_,_,_=np.linalg.lstsq(M,z[bg].ravel(),rcond=None);res=z-(co[0]*xs+co[1]*ys+co[2])
    cap=rad<=R;capin=rad<=Rin;prom=max(float(res[int(ay),int(ax)]),0.05);relief=float(z[cap].max()-np.percentile(z[cap],10))
    rc,prof=_rad(res,rad,cap,R)
    if len(prof)<4: return None
    ss,sg2=_fit(rc,prof);sstot=float(((prof-prof.mean())**2).sum())+1e-9;gfit=1-ss/sstot;fwhm=float(2.3548*sg2)
    e=np.arange(0,R+2,2);cvs=[]
    for k in range(len(e)-1):
        m=(rad>=e[k])&(rad<e[k+1])&cap
        if m.sum()>=8:
            v=res[m];mu=v.mean()
            if abs(mu)>1e-3:cvs.append(v.std()/abs(mu))
    sym=float(1.0/(1.0+np.mean(cvs))) if cvs else 0.0
    lap=(np.roll(res,1,0)+np.roll(res,-1,0)+np.roll(res,1,1)+np.roll(res,-1,1)-4*res)/(cs2*cs2)
    convex=float(-lap[capin].mean()/prom);rough=float(lap[cap].std()/prom)
    def slpk(rm): bgb=boxblur1(boxblur1(z,int(max(2,rm/cs2))),int(max(2,rm/cs2)));return float((z-bgb)[capin].mean()/prom)
    sl15=slpk(max(8,1.2*s));sl45=slpk(max(16,2.6*s));mono=float(np.mean(np.diff(prof)<=0)) if len(prof)>2 else 0.0
    d=dict(prom=prom,gfit=gfit,fwhm_m=fwhm,sym=sym,convex=convex,rough=rough,slrm15=sl15,slrm45=sl45,mono=mono,relief_m=relief)
    try:
        v=np.array([d[k] for k in FEATS])
        if np.any(~np.isfinite(v)): return None
        zz=(v-gmu)/gsd;return float(1/(1+np.exp(-(zz@gw+gb))))
    except: return None
# barrows
bar=json.load(open('/tmp/uk_barrows.json'))['elements']
pts=[]
for e in bar:
    if e['type']=='node': pts.append((e['lat'],e['lon']))
    elif 'center' in e: pts.append((e['center']['lat'],e['center']['lon']))
bxy=[]
for la,lo in pts:
    E,N=osgb(la,lo);px=int((E-xll)/ce);py=int((ytop-N)/ce)
    if 0<=px<Ww and 0<=py<Hh: bxy.append((px,py))
def auc(p,n):
    p=np.array([x for x in p if x is not None]);n=np.array([x for x in n if x is not None])
    if not len(p) or not len(n): return float('nan')
    allv=np.concatenate([p,n]);_,inv,cnt=np.unique(allv,return_inverse=True,return_counts=True);cs=np.cumsum(cnt);st=cs-cnt;avg=(st+cs+1)/2.0;ranks=avg[inv]
    return float((ranks[:len(p)].sum()-len(p)*(len(p)+1)/2)/(len(p)*len(n)))
pos_m=[score_pt(*b) for b in bxy];pos_g=[gate_feat(*b) for b in bxy]
bset=np.array(bxy) if bxy else np.zeros((0,2));md=int(60/ce);random.seed(len(bxy)+nc);cm=[];cg=[];tr=0
while len(cm)<max(60,len(bxy)) and tr<6000:
    tr+=1;px=random.randint(120,Ww-120);py=random.randint(120,Hh-120)
    if len(bset) and np.min(np.hypot(bset[:,0]-px,bset[:,1]-py))<md: continue
    cm.append(score_pt(px,py));cg.append(gate_feat(px,py))
pg=[g for g in pos_g if g is not None];cgv=[g for g in cg if g is not None]
print(f"=== UK {CLON},{CLAT} ({KM}km, EA 1m) | barrows OSM n={len(bxy)} | control n={len(cm)} ===")
print(f"  MODEL: AUROC {auc(pos_m,cm):.3f} | barrows med {np.median(pos_m):.2f} vs control {np.median(cm):.2f}")
print(f"  GATE: AUROC {auc(pg,cgv):.3f} | barrows med {np.median(pg):.2f} vs control {np.median(cgv):.2f} (n {len(pg)}/{len(cgv)})")
print(f"  gate retenție@0.70: barrows {100*np.mean(np.array(pg)>=0.70):.0f}% | control taie {100*np.mean(np.array(cgv)<0.70):.0f}%")
