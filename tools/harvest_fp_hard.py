#!/usr/bin/env python3
# harvest_fp_hard.py [N] [THR] [MAXPERTILE] [SEED] — HARD-NEGATIVE MINING pe cache laki3 RO.
# Păstrează DOAR celulele pe care modelul CURENT le scorează >=THR = fals-pozitivele lui REALE.
# Exclude dome-veto (formă de movilă = posibil tumul, principiul de formă) + vecinătate movile cunoscute (<120m).
# Clasifică plough/ditch/stream/anthro pt verificare. Recipe = neg_stamp training (80m DEM->2m->hs@2m->128 RAW).
# -> dataset_neg_ro_fp5k/ (SUPRASCRIE valul easy) + manifest.csv + boards review/fphard_<class>.jpg
import os,sys,math,subprocess,csv,glob,random,collections
import numpy as np
from PIL import Image,ImageDraw,ImageFont,ImageFilter
import torch,torch.nn as nn
H=os.path.dirname(os.path.dirname(os.path.abspath(__file__)));dev=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
CACHE="/tmp/laki3";CS=0.5;TPX=2000;f=int(round(2.0/CS));WPX=int(80/CS)
N=int(sys.argv[1]) if len(sys.argv)>1 else 5000
THR=float(sys.argv[2]) if len(sys.argv)>2 else 0.5
MAXPERTILE=int(sys.argv[3]) if len(sys.argv)>3 else 40
SEED=int(sys.argv[4]) if len(sys.argv)>4 else 7
STEP_M=30;MODEL=os.environ.get('MODEL',f'{H}/combined_cnn.pt');OUT=os.environ.get('OUTDIR',f"{H}/dataset_neg_ro_fp5k")
random.seed(SEED);np.random.seed(SEED)
os.makedirs(OUT,exist_ok=True)
APP=os.environ.get("QGIS_APP","/Applications/QGIS-final-4_0_3.app/Contents")
ENV=dict(os.environ,DYLD_FRAMEWORK_PATH=f"{APP}/Frameworks",PROJ_DATA=f"{APP}/Resources/qgis/proj",PROJ_LIB=f"{APP}/Resources/qgis/proj",GDAL_DATA=f"{APP}/Resources/qgis/gdal")
GT=f"{APP}/MacOS/gdaltransform"
def trans(pts,s,t):
    if not pts: return []
    inp="\n".join(f"{a} {b}" for a,b in pts)+"\n"
    r=subprocess.run([GT,"-s_srs",s,"-t_srs",t],input=inp,capture_output=True,text=True,env=ENV)
    return [(float(l.split()[0]),float(l.split()[1])) for l in r.stdout.strip().split("\n") if l.split()]
def hs(dem,cs,azs=(315,45,135,225,270,0),alt=35):
    gy,gx=np.gradient(dem,cs);sl=np.arctan(np.hypot(gx,gy));asp=np.arctan2(-gy,gx);o=np.zeros_like(dem);ar=math.radians(alt)
    for az in azs: azr=math.radians(360-az+90);o+=np.clip(np.sin(ar)*np.cos(sl)+np.cos(ar)*np.sin(sl)*np.cos(azr-asp),0,1)
    return o/len(azs)
def downs(a,fc):
    Hh,Ww=a.shape;return a[:Hh//fc*fc,:Ww//fc*fc].reshape(Hh//fc,fc,Ww//fc,fc).mean((1,3))
def _histeq(a):
    h=np.bincount(a.ravel(),minlength=256).astype(np.float64);cdf=h.cumsum();return a if cdf[-1]==0 else (cdf[a]/cdf[-1]*255).astype(np.uint8)
def homog(a): return _histeq(np.asarray(Image.fromarray(a).filter(ImageFilter.GaussianBlur(0.8)),np.uint8))
def neg_stamp(w):  # RAW 128 (recipe neg_stamp, fără homog — homog se aplică la scoring/load)
    if w.shape!=(WPX,WPX) or np.isnan(w).mean()>0.05: return None
    d2=downs(np.nan_to_num(w,nan=np.nanmedian(w)),f);h=hs(d2,CS*f);lo,hi=np.percentile(h,2),np.percentile(h,98)
    if hi-lo<1e-6: return None
    return np.asarray(Image.fromarray(np.clip((h-lo)/(hi-lo)*255,0,255).astype('uint8')).resize((128,128)),np.uint8)
def feats(w):
    z=downs(np.nan_to_num(w,nan=np.nanmedian(w)),f);n=z.shape[0]
    ys,xs=np.mgrid[0:n,0:n].astype(float);A=np.c_[xs.ravel(),ys.ravel(),np.ones(n*n)];co,_,_,_=np.linalg.lstsq(A,z.ravel(),rcond=None)
    r=z-(co[0]*xs+co[1]*ys+co[2]);cx=cy=n/2.0;rad=np.hypot(xs-cx,ys-cy)*CS*f
    inner=rad<=14;ring=(rad>=22)&(rad<=36)
    if inner.sum()<6 or ring.sum()<6: return None
    prom=float(r[inner].mean());relief=float(np.percentile(r,98)-np.percentile(r,2));rough=float(r.std());valley=float(r[inner].mean()-r[ring].mean())
    gy,gx=np.gradient(z,CS*f);Sxx=float((gx*gx).mean());Syy=float((gy*gy).mean());Sxy=float((gx*gy).mean())
    tr=Sxx+Syy;dsc=max((tr/2)**2-(Sxx*Syy-Sxy*Sxy),0.0);l1=tr/2+math.sqrt(dsc);l2=tr/2-math.sqrt(dsc);coh=float((l1-l2)/(l1+l2+1e-9))
    mid=rad<=24  # COERENȚĂ CENTRALĂ (doar miez, ignoră arătura din jur): rotund=jos, alungit=sus
    Sxm=float((gx*gx)[mid].mean());Sym=float((gy*gy)[mid].mean());Sxym=float((gx*gy)[mid].mean())
    trm=Sxm+Sym;dscm=max((trm/2)**2-(Sxm*Sym-Sxym*Sxym),0.0);ccoh=float((math.sqrt(dscm)*2)/(trm+1e-9))
    cvs=[]
    for k in range(0,24,4):
        m=(rad>=k)&(rad<k+4)
        if m.sum()>=6:
            v=r[m];mu=v.mean()
            if abs(mu)>1e-3: cvs.append(v.std()/abs(mu))
    sym=float(1.0/(1.0+np.mean(cvs))) if cvs else 0.0
    prof=[]
    for k in range(0,28,4):
        m=(rad>=k)&(rad<k+4)
        if m.sum()>=4: prof.append(r[m].mean())
    mono=float(np.mean(np.diff(np.array(prof))<=0)) if len(prof)>=4 else 0.0
    return dict(prom=prom,relief=relief,rough=rough,valley=valley,coh=coh,mono=mono,ccoh=ccoh,sym=sym)
def classify_hard(F):
    p,relief,rough,valley,coh,mono=F['prom'],F['relief'],F['rough'],F['valley'],F['coh'],F['mono']
    ccoh=F.get('ccoh',coh);sym=F.get('sym',0.0)
    # DOME-VETO CENTRU (posibil tumul, ex. movilă pe câmp arat): centru rotund(ccoh jos)+convex+simetric+radial -> EXCLUDE
    if p>0.08 and ccoh<0.45 and sym>=0.55 and mono>=0.55: return 'EXCL_dome'
    # DOME-VETO fereastră (dom rotund pe teren plat, nedirectional)
    if p>0.20 and mono>=0.70 and coh<0.50: return 'EXCL_dome'
    if coh>=0.50 and abs(p)<0.25: return 'plough'
    if coh>=0.45 and p<=-0.12: return 'ditch'
    if valley<=-0.22 and p<0.10: return 'stream'
    if rough>=0.45 or relief>=0.8: return 'anthro'
    return 'other'
class Net(nn.Module):
    def __init__(s):
        super().__init__();s.c=nn.Sequential(nn.Conv2d(1,16,3,2,1),nn.ReLU(),nn.Conv2d(16,32,3,2,1),nn.ReLU(),nn.Conv2d(32,64,3,2,1),nn.ReLU(),nn.AdaptiveAvgPool2d(1));s.f=nn.Linear(64,1)
    def forward(s,x): return s.f(s.c(x).flatten(1)).squeeze(1)
net=Net().to(dev);net.load_state_dict(torch.load(MODEL,map_location=dev,weights_only=True));net.eval()
def score_batch(raws):
    xb=torch.tensor(np.array([homog(r) for r in raws],dtype=np.uint8)).unsqueeze(1).float().to(dev)/255.
    with torch.no_grad(): return torch.sigmoid(net(xb)).cpu().numpy()
# movile cunoscute -> excludere <120m
kll=[]
for pth,lonc,latc in [(f'{H}/labeled/labels.csv','lon','lat'),(f'{H}/labeled/confirmed_new_positives_20260625.csv','lon','lat'),(f'{H}/labeled/possible_mounds_20260626.csv','lon','lat'),('/tmp/catane_gt_full.csv','lon','lat')]:
    if os.path.exists(pth):
        for r in csv.DictReader(open(pth)):
            if 'verdict' in r and r['verdict']!='mound': continue
            try: kll.append((float(r[lonc]),float(r[latc])))
            except: pass
known=np.array(trans(kll,"EPSG:4326","EPSG:3844")) if kll else np.empty((0,2))
print(f"movile cunoscute pt excludere: {len(kll)}",flush=True)
def near_mound(e,n,d=120):
    if known.shape[0]==0: return False
    return bool(np.min((known[:,0]-e)**2+(known[:,1]-n)**2)<d*d)
# scan
tiles=glob.glob(f"{CACHE}/*.npy");random.shuffle(tiles)
buckets=collections.defaultdict(list);excl=[];ntiles=0;nscored=0
step=int(STEP_M/CS);hw=WPX//2;CLS=['plough','ditch','stream','anthro','other']
for tp in tiles:
    if sum(len(buckets[c]) for c in CLS)>=N: break
    try: nk,ek=os.path.basename(tp)[:-4].split('_');nk=int(nk);ek=int(ek);T=np.load(tp)
    except: continue
    if T.shape[0]<WPX or T.shape[1]<WPX: continue
    ntiles+=1;cand=[]
    ys=list(range(hw,T.shape[0]-hw,step));xs=list(range(hw,T.shape[1]-hw,step))
    for py in ys:
        for px in xs:
            w=T[py-hw:py+hw,px-hw:px+hw]
            if w.shape!=(WPX,WPX) or np.isnan(w).mean()>0.05: continue
            st=neg_stamp(w)
            if st is None: continue
            cand.append((px,py,st))
    if not cand: continue
    scs=[]
    for i in range(0,len(cand),1024): scs.extend(score_batch([c[2] for c in cand[i:i+1024]]))
    nscored+=len(cand);kept=0
    order=sorted(range(len(cand)),key=lambda i:-scs[i])
    for i in order:
        if kept>=MAXPERTILE: break
        if scs[i]<THR: break
        px,py,st=cand[i];w=T[py-hw:py+hw,px-hw:px+hw];F=feats(w)
        if F is None: continue
        lab=classify_hard(F);est=ek*1000+px*CS;nord=(nk+1)*1000-py*CS
        if lab=='EXCL_dome':
            if len(excl)<400: excl.append((st,est,nord,float(scs[i]),F))
            continue
        if near_mound(est,nord): continue
        buckets[lab].append((st,est,nord,float(scs[i]),F));kept+=1
    if ntiles%50==0: print(f"  {ntiles} tiles ({nscored} scored) | "+" ".join(f"{c} {len(buckets[c])}" for c in CLS)+f" | excl_dome {len(excl)}",flush=True)
tot=sum(len(buckets[c]) for c in CLS)
print(f"SCAN gata: {ntiles} tiles, {nscored} scored | "+" ".join(f"{c} {len(buckets[c])}" for c in CLS)+f" | tot {tot} | excl_dome {len(excl)}",flush=True)
# manifest + save (suprascrie dir)
for old in glob.glob(f"{OUT}/*.png"): os.remove(old)
allpts=[(c,i,e,nn) for c in CLS for i,(st,e,nn,sc,F) in enumerate(buckets[c])]
inp="".join(f"{e} {nn}\n" for _,_,e,nn in allpts)
r=subprocess.run([GT,"-s_srs","EPSG:3844","-t_srs","EPSG:4326"],input=inp,capture_output=True,text=True,env=ENV)
ll=[(float(x.split()[0]),float(x.split()[1])) for x in r.stdout.strip().split("\n")] if r.stdout.strip() else [(0,0)]*len(allpts)
llmap={(c,i):(lo,la) for (c,i,_,_),(lo,la) in zip(allpts,ll)}
mf=open(f"{OUT}/manifest.csv","w");mw=csv.writer(mf);mw.writerow(['file','class','model_score','lon','lat','prom','coh','relief','valley','rough','mono']);ntot=0
for c in CLS:
    for i,(st,e,nn,sc,F) in enumerate(buckets[c]):
        fn=f"{c}_{i:05d}.png";Image.fromarray(st).save(f"{OUT}/{fn}")
        lo,la=llmap.get((c,i),(0,0));mw.writerow([fn,c,f"{sc:.3f}",f"{lo:.6f}",f"{la:.6f}",f"{F['prom']:.3f}",f"{F['coh']:.3f}",f"{F['relief']:.3f}",f"{F['valley']:.3f}",f"{F['rough']:.3f}",f"{F['mono']:.2f}"]);ntot+=1
mf.close();print(f"SALVAT: {ntot} stampe HARD -> {OUT}/",flush=True)
def board(items,title,path,cols=20,rows=15,thumb=72):
    sel=items[:cols*rows];W=cols*thumb;Hh=rows*thumb+30;im=Image.new('RGB',(W,Hh),(15,15,18));d=ImageDraw.Draw(im)
    try: ft=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial Bold.ttf',16)
    except: ft=ImageFont.load_default()
    for k,it in enumerate(sel):
        st=it[0];r_,c_=divmod(k,cols);x=c_*thumb;y=30+r_*thumb;im.paste(Image.fromarray(st).resize((thumb-2,thumb-2)),(x+1,y+1))
    d.rectangle([0,0,W,28],fill=(12,12,14));d.text((6,5),title,fill=(255,255,255),font=ft);im.save(path,quality=88);print(f"-> {path} ({len(sel)}/{len(items)})")
for c in CLS:
    if buckets[c]: random.shuffle(buckets[c]);board(buckets[c],f"HARD-NEG {c.upper()} (model aprinde fals, >=0.5, non-dom) — {len(buckets[c])}",f"{H}/review/fphard_{c}.jpg")
if excl: board(excl,f"EXCLUSE dome (model aprinde DAR formă movilă -> posibil tumul, NU negativ) — {len(excl)}",f"{H}/review/fphard_excluded_dome.jpg")
