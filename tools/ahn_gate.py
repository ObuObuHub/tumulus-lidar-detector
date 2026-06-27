#!/usr/bin/env python3
# ahn_gate.py CLON CLAT KM [MODEL] — test CROSS-ȚARĂ al modelului ȘI al filtrului de curbură pe movile NL (AHN 0.5m).
# Fetch AHN DTM -> scor model (stamp multi-scară) + scor filtru curbură (trasături v2 din DEM AHN) la fiecare movilă
# OSM vs control random. Raport: AUROC model, AUROC gate, retenție gate@0.70 movile vs control.
import sys,os,math,subprocess,json,random
import numpy as np
from PIL import Image,ImageFilter
import torch,torch.nn as nn
H=os.path.dirname(os.path.dirname(os.path.abspath(__file__)));dev=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
CLON=float(sys.argv[1]);CLAT=float(sys.argv[2]);KM=float(sys.argv[3]) if len(sys.argv)>3 else 2.0
MODEL=sys.argv[4] if len(sys.argv)>4 else f'{H}/combined_cnn.pt'
SCALES=[28,32];MPP=0.5
APP=os.environ.get("QGIS_APP","/Applications/QGIS-final-4_0_3.app/Contents")
ENV=dict(os.environ,DYLD_FRAMEWORK_PATH=f"{APP}/Frameworks",PROJ_DATA=f"{APP}/Resources/qgis/proj",PROJ_LIB=f"{APP}/Resources/qgis/proj",GDAL_DATA=f"{APP}/Resources/qgis/gdal")
GT=f"{APP}/MacOS/gdaltransform";GTR=f"{APP}/MacOS/gdal_translate"
def trans_many(pts,s="EPSG:4326",t="EPSG:28992"):
    inp="".join(f"{lo} {la}\n" for la,lo in pts);r=subprocess.run([GT,"-s_srs",s,"-t_srs",t],input=inp,capture_output=True,text=True,env=ENV)
    return [tuple(map(float,l.split()[:2])) for l in r.stdout.strip().split("\n")] if r.stdout.strip() else []
est,nord=trans_many([(CLAT,CLON)])[0];half=KM*1000/2
x0,x1=est-half,est+half;y0,y1=nord-half,nord+half
url=(f"https://service.pdok.nl/rws/ahn/wcs/v1_0?service=WCS&version=2.0.1&request=GetCoverage"
     f"&coverageId=dtm_05m&subset=x({x0:.1f},{x1:.1f})&subset=y({y0:.1f},{y1:.1f})&format=image/tiff")
tif="/tmp/ahn_g.tif";asc="/tmp/ahn_g.asc"
print(f"fetch AHN {KM}km @ {CLON},{CLAT}...",flush=True)
subprocess.run(["curl","-s","--max-time","180","-o",tif,url],check=False)
if not os.path.exists(tif) or os.path.getsize(tif)<10000: sys.exit("EROARE fetch AHN")
if os.path.exists(asc): os.remove(asc)
subprocess.run([GTR,"-of","AAIGrid",tif,asc],capture_output=True,env=ENV)
L=open(asc).read().split('\n');hdr={};i=0
while i<len(L) and L[i].split() and L[i].split()[0].lower() in('ncols','nrows','xllcorner','yllcorner','cellsize','nodata_value'):
    k,v=L[i].split()[:2];hdr[k.lower()]=float(v);i+=1
nc,nr=int(hdr['ncols']),int(hdr['nrows']);xll=hdr['xllcorner'];yll=hdr['yllcorner'];ce=hdr['cellsize']
DEM=np.fromstring(' '.join(L[i:]),sep=' ',dtype=np.float32)[:nc*nr].reshape(nr,nc)
DEM[DEM>1e30]=np.nan;DEM[DEM==hdr.get('nodata_value',-9999)]=np.nan;ytop=yll+nr*ce
DEM=np.nan_to_num(DEM,nan=float(np.nanmedian(DEM)))
def hs(d,cs,azs=(315,45,135,225,270,0),alt=35):
    gy,gx=np.gradient(d,cs);sl=np.arctan(np.hypot(gx,gy));asp=np.arctan2(-gy,gx);o=np.zeros_like(d);ar=math.radians(alt)
    for az in azs: azr=math.radians(360-az+90);o+=np.clip(np.sin(ar)*np.cos(sl)+np.cos(ar)*np.sin(sl)*np.cos(azr-asp),0,1)
    return o/len(azs)
sh=hs(DEM,ce);lo,hi=np.percentile(sh,2),np.percentile(sh,98);A=np.clip((sh-lo)/(hi-lo+1e-9)*255,0,255).astype(np.uint8);Hh,Ww=A.shape
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
        with torch.no_grad(): v=float(torch.sigmoid(net(torch.tensor(s[None,None],dtype=torch.float32).to(dev)/255.)).item())
        best=max(best,v)
    return best
# ---- filtru curbură: trasături v2 din DEM AHN (port curv_features2.analyze, fereastră 100m @2m) ----
import json as _json
GATEJ=os.environ.get('GATE',f'{H}/curv_gate.json')  # GATE=...gate_v3.json pt scale-adaptiv
gate=_json.load(open(GATEJ));FEATS=gate['feats'];SCALEADAPT=os.environ.get('V3','') !=''
gw=np.array(gate['w']);gb=gate['b'];gmu=np.array(gate['mu']);gsd=np.array(gate['sd'])
def boxblur1(a,r):
    Hh2,Ww2=a.shape;ii=np.zeros((Hh2+1,Ww2+1),np.float64);ii[1:,1:]=np.cumsum(np.cumsum(a,0),1)
    ys=np.arange(Hh2);xs=np.arange(Ww2);yA=np.clip(ys-r,0,Hh2);yB=np.clip(ys+r+1,0,Hh2);xA=np.clip(xs-r,0,Ww2);xB=np.clip(xs+r+1,0,Ww2)
    S=ii[yB][:,xB]-ii[yA][:,xB]-ii[yB][:,xA]+ii[yA][:,xA];cnt=((yB-yA)[:,None]*(xB-xA)[None,:]).astype(np.float64)
    return (S/cnt).astype(np.float32)
def _fit_sigma(rc,prof,smin=4,smax=40,ns=19):
    best=(1e9,8.0,prof.max() if len(prof) else 0.1,0.0)
    for s in np.linspace(smin,smax,ns):
        g=np.exp(-(rc**2)/(2*s*s));Mm=np.c_[g,np.ones(len(g))];sol,_,_,_=np.linalg.lstsq(Mm,prof,rcond=None);ss=float(((prof-Mm@sol)**2).sum())
        if ss<best[0]:best=(ss,s,sol[0],sol[1])
    return best
def _radial(res,rad,cap,rmax,step=2):
    edges=np.arange(0,rmax+step,step);prof=[];rc=[]
    for k in range(len(edges)-1):
        m=(rad>=edges[k])&(rad<edges[k+1])&cap
        if m.sum()>=3:prof.append(float(res[m].mean()));rc.append((edges[k]+edges[k+1])/2)
    return np.array(rc),np.array(prof)
def gate_feat(px,py):
    half=int(110/ce)
    if px-half<0 or py-half<0 or px+half>Ww or py+half>Hh: return None
    w=DEM[py-half:py+half,px-half:px+half]
    if w.shape!=(2*half,2*half): return None
    f=int(round(2.0/ce));Hc=w.shape[0]//f*f;z=w[:Hc,:Hc].reshape(Hc//f,f,Hc//f,f).mean((1,3));cs2=ce*f
    c0=z.shape[0]//2;rr=int(16/cs2);sub=z[c0-rr:c0+rr,c0-rr:c0+rr]
    off=np.unravel_index(np.argmax(sub),sub.shape);ay=c0-rr+off[0];ax=c0-rr+off[1]
    ys,xs=np.mgrid[0:z.shape[0],0:z.shape[1]].astype(np.float64);rad=np.hypot(ys-ay,xs-ax)*cs2
    if SCALEADAPT:
        bg0=(rad>=30)&(rad<=52)
        if bg0.sum()<20: return None
        A0=np.c_[xs[bg0].ravel(),ys[bg0].ravel(),np.ones(bg0.sum())];c0f,_,_,_=np.linalg.lstsq(A0,z[bg0].ravel(),rcond=None)
        res0=z-(c0f[0]*xs+c0f[1]*ys+c0f[2]);rc0,prof0=_radial(res0,rad,rad<=45,45)
        if len(prof0)<4: return None
        _,sigma,_,_=_fit_sigma(rc0,prof0,3,40,38);s=float(np.clip(sigma,3,35))
        R=float(np.clip(2.2*s,10,48));Rin=float(np.clip(0.9*s,4,22));bgi=float(np.clip(R+6,R+6,52));bgo=float(np.clip(R+24,bgi+6,54))
        bg=(rad>=bgi)&(rad<=bgo)
        if bg.sum()<15: bg=bg0
        sl_a=max(8,1.2*s);sl_b=max(16,2.6*s);sfit=(3,40,38)
    else:
        R=25.0;Rin=12.0;bg=(rad>=30)&(rad<=50);sl_a=15;sl_b=45;sfit=(4,40,19)
        if bg.sum()<20: return None
    M=np.c_[xs[bg].ravel(),ys[bg].ravel(),np.ones(bg.sum())];coef,_,_,_=np.linalg.lstsq(M,z[bg].ravel(),rcond=None)
    res=z-(coef[0]*xs+coef[1]*ys+coef[2])
    cap=rad<=R;capin=rad<=Rin;prom=max(float(res[int(ay),int(ax)]),0.05);relief=float(z[cap].max()-np.percentile(z[cap],10))
    rc,prof=_radial(res,rad,cap,R)
    if len(prof)<4: return None
    ss,sg2,_,_=_fit_sigma(rc,prof,*sfit);sstot=float(((prof-prof.mean())**2).sum())+1e-9;gfit=1-ss/sstot;fwhm_m=float(2.3548*sg2)
    edges=np.arange(0,R+2,2);cvs=[]
    for k in range(len(edges)-1):
        m=(rad>=edges[k])&(rad<edges[k+1])&cap
        if m.sum()>=8:
            v=res[m];mu=v.mean()
            if abs(mu)>1e-3: cvs.append(v.std()/abs(mu))
    sym=float(1.0/(1.0+np.mean(cvs))) if cvs else 0.0
    lap=(np.roll(res,1,0)+np.roll(res,-1,0)+np.roll(res,1,1)+np.roll(res,-1,1)-4*res)/(cs2*cs2)
    convex=float(-lap[capin].mean()/prom);rough=float(lap[cap].std()/prom)
    def slpk(rm): bgb=boxblur1(boxblur1(z,int(max(2,rm/cs2))),int(max(2,rm/cs2)));return float((z-bgb)[capin].mean()/prom)
    slrm15=slpk(sl_a);slrm45=slpk(sl_b);mono=float(np.mean(np.diff(prof)<=0)) if len(prof)>2 else 0.0
    d=dict(prom=prom,gfit=gfit,fwhm_m=fwhm_m,sym=sym,convex=convex,rough=rough,slrm15=slrm15,slrm45=slrm45,mono=mono,relief_m=relief)
    try:
        v=np.array([d[k] for k in FEATS])
        if np.any(~np.isfinite(v)): return None
        zz=(v-gmu)/gsd;return float(1/(1+np.exp(-(zz@gw+gb))))
    except: return None
# movile + control
barrows=json.load(open('/tmp/nl_barrows.json'))
inb=[(la,lo) for la,lo,_ in barrows if abs(la-CLAT)<KM/111.0 and abs(lo-CLON)<KM/(111.0*math.cos(math.radians(CLAT)))]
EN=trans_many(inb) if inb else []
bxy=[]
for (la,lo),(e,n) in zip(inb,EN):
    px=int((e-xll)/ce);py=int((ytop-n)/ce)
    if 0<=px<Ww and 0<=py<Hh: bxy.append((px,py))
def auc(p,n):
    p=np.array([x for x in p if x is not None]);n=np.array([x for x in n if x is not None])
    if len(p)==0 or len(n)==0: return float('nan')
    allv=np.concatenate([p,n]);_,inv,cnt=np.unique(allv,return_inverse=True,return_counts=True);cs=np.cumsum(cnt);st=cs-cnt;avg=(st+cs+1)/2.0;ranks=avg[inv]
    return float((ranks[:len(p)].sum()-len(p)*(len(p)+1)/2)/(len(p)*len(n)))
pos_m=[score_pt(*b) for b in bxy];pos_g=[gate_feat(*b) for b in bxy]
bset=np.array(bxy) if bxy else np.zeros((0,2));md=int(60/ce);random.seed(len(bxy)+nc);ctrl_m=[];ctrl_g=[];tries=0
while len(ctrl_m)<max(60,len(bxy)) and tries<6000:
    tries+=1;px=random.randint(120,Ww-120);py=random.randint(120,Hh-120)
    if len(bset) and np.min(np.hypot(bset[:,0]-px,bset[:,1]-py))<md: continue
    ctrl_m.append(score_pt(px,py));ctrl_g.append(gate_feat(px,py))
pg=[g for g in pos_g if g is not None];cg=[g for g in ctrl_g if g is not None]
print(f"=== AHN {CLON},{CLAT} ({KM}km) | movile OSM n={len(bxy)} | control n={len(ctrl_m)} ===")
print(f"  MODEL: AUROC {auc(pos_m,ctrl_m):.3f} | movile med {np.median(pos_m):.2f} vs control {np.median(ctrl_m):.2f}")
print(f"  GATE curbură: AUROC {auc(pg,cg):.3f} | movile med {np.median(pg):.2f} vs control {np.median(cg):.2f} (n_g {len(pg)}/{len(cg)})")
print(f"  gate retenție@0.70: movile {100*np.mean(np.array(pg)>=0.70):.0f}% | control păstrat {100*np.mean(np.array(cg)>=0.70):.0f}% (taie {100*np.mean(np.array(cg)<0.70):.0f}% control)")
if os.environ.get('SAMPLES'):
    from PIL import ImageDraw,ImageFont
    def F(s,b=True):
        try:return ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial Bold.ttf' if b else '/System/Library/Fonts/Supplemental/Arial.ttf',s)
        except:return ImageFont.load_default()
    items=sorted(zip(bxy,pos_m,pos_g),key=lambda t:-(t[1] or 0))[:30]
    half=int(40/ce)  # 80m window (movile NL mici)
    cols=6;CW=150;CHH=340;rn=(len(items)+cols-1)//cols
    img=Image.new('RGB',(cols*CW,rn*CHH+50),(16,16,18));dr=ImageDraw.Draw(img)
    dr.text((10,8),f"Movile OLANDA (AHN 0.5m) @ {CLON},{CLAT}. Sus=hillshade, jos=SLRM. m=scor model, g=scor filtru.",fill=(255,230,90),font=F(16))
    dr.text((10,30),"Movile mici (~10-15m) — modelul le prinde (m mare), filtrul RO le scorează jos (g mic, scara RO).",fill=(220,220,220),font=F(13,False))
    for k,((px,py),m,g) in enumerate(items):
        cx=(k%cols)*CW;cy=(k//cols)*CHH+50
        if px-half>=0 and py-half>=0 and px+half<Ww and py+half<Hh:
            wsh=A[py-half:py+half,px-half:px+half]
            img.paste(Image.fromarray(wsh).resize((140,140)).convert('RGB'),(cx+5,cy+22))
            wz=DEM[py-half:py+half,px-half:px+half];sl=wz-boxblur1(boxblur1(wz,int(30/ce)),int(30/ce))
            lo2,hi2=np.percentile(sl,2),np.percentile(sl,98);sli=np.clip((sl-lo2)/(hi2-lo2+1e-9)*255,0,255).astype('uint8')
            img.paste(Image.fromarray(sli).resize((140,140)).convert('RGB'),(cx+5,cy+166))
        dr.text((cx+5,cy+4),f"m{m:.2f} g{(g if g is not None else -1):.2f}",fill=(120,235,120) if (m or 0)>=0.7 else (235,170,60),font=F(13))
    img.save(f'{H}/review/ahn_samples.jpg',quality=88);print(f"-> review/ahn_samples.jpg {img.size} ({len(items)} movile)")
