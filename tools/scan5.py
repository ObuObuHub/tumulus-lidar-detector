#!/usr/bin/env python3
# scan5.py [N=5] [KM=1.2] — scanează N zone RANDOM din SV-RO (LAKI III Oltenia 0.5m), extrage candidați
# COMPACT pozitivi (filtrează liniarele=FP evident), board cu perechi CLEAN|HEATMAP numerotate + map CSV.
import os,sys,math,subprocess,zipfile,random,csv
import numpy as np
from PIL import Image,ImageFilter,ImageDraw,ImageFont
import torch,torch.nn as nn
H=os.path.expanduser('~/lidar-match');dev=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
N=int(sys.argv[1]) if len(sys.argv)>1 else 5
KM=float(sys.argv[2]) if len(sys.argv)>2 else 1.2
SCALES=[48.0,80.0];CACHE="/tmp/laki3";CS=0.5;TPX=2000;os.makedirs(CACHE,exist_ok=True)
BOX=tuple(float(x) for x in os.environ.get('BOX','22.95,43.80,23.95,44.05').split(','))  # default=Dolj SUD (LAKI III); rejection scoate lunca/apa
APP="/Applications/QGIS-final-4_0_3.app/Contents"
ENV=dict(os.environ,DYLD_FRAMEWORK_PATH=f"{APP}/Frameworks",PROJ_DATA=f"{APP}/Resources/qgis/proj",PROJ_LIB=f"{APP}/Resources/qgis/proj",GDAL_DATA=f"{APP}/Resources/qgis/gdal")
GT=f"{APP}/MacOS/gdaltransform"
def trans(lon,lat):
    r=subprocess.run([GT,"-s_srs","EPSG:4326","-t_srs","EPSG:3844"],input=f"{lon} {lat}\n",capture_output=True,text=True,env=ENV);a,b,_=r.stdout.split();return float(a),float(b)
def head200(nk,ek):
    r=subprocess.run(["curl","-sI","--max-time","8",f"https://geoportal.ancpi.ro/laki3_mnt/zip/{nk}_{ek}.zip"],capture_output=True,text=True)
    return "200" in r.stdout.split("\n")[0]
def load_one(nk,ek):
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
class Net(nn.Module):
    def __init__(s):
        super().__init__();s.c=nn.Sequential(nn.Conv2d(1,16,3,2,1),nn.ReLU(),nn.Conv2d(16,32,3,2,1),nn.ReLU(),nn.Conv2d(32,64,3,2,1),nn.ReLU(),nn.AdaptiveAvgPool2d(1));s.f=nn.Linear(64,1)
    def forward(s,x): return s.f(s.c(x).flatten(1)).squeeze(1)
net=Net().to(dev);net.load_state_dict(torch.load(f'{H}/combined_cnn.pt',map_location=dev,weights_only=True));net.eval()
def jet(s):
    r=np.clip(1.5-np.abs(4*s-3),0,1);g=np.clip(1.5-np.abs(4*s-2),0,1);b=np.clip(1.5-np.abs(4*s-1),0,1);return np.stack([r,g,b],-1)
# compactness: linie lungă (major axis) din relief local -> liniar=FP
def compactness(w):
    d2=downs(np.nan_to_num(w,nan=float(np.nanmedian(w))),4);cs2=CS*4
    rng=float(np.ptp(d2))+1e-6
    blur=np.asarray(Image.fromarray(((d2-d2.min())/rng*255).astype('uint8')).filter(ImageFilter.GaussianBlur(15)),np.float32)/255*rng
    lr=d2-blur
    cy,cx=lr.shape[0]//2,lr.shape[1]//2
    thr=np.percentile(lr,88);mask=lr>=thr
    # flood fill BFS din centru (vecinătate 8)
    if not mask[cy,cx]:
        # caută cel mai apropiat pixel mask de centru într-o rază mică
        ys,xs=np.where(mask)
        if len(ys)==0: return 0.0,9999
        di=(ys-cy)**2+(xs-cx)**2;k=di.argmin()
        if di[k]>(8)**2: return 0.0,9999
        cy,cx=ys[k],xs[k]
    from collections import deque
    seen=np.zeros_like(mask);q=deque([(cy,cx)]);seen[cy,cx]=1;comp=[]
    while q:
        y,x=q.popleft();comp.append((y,x))
        for dy in (-1,0,1):
            for dx in (-1,0,1):
                ny,nx=y+dy,x+dx
                if 0<=ny<mask.shape[0] and 0<=nx<mask.shape[1] and mask[ny,nx] and not seen[ny,nx]: seen[ny,nx]=1;q.append((ny,nx))
    comp=np.array(comp)
    if len(comp)<4: return 0.0,0
    yy=comp[:,0]-comp[:,0].mean();xx=comp[:,1]-comp[:,1].mean();cov=np.cov(np.vstack([xx,yy]));ev=np.linalg.eigvalsh(cov)
    ev=np.clip(ev,1e-6,None);ratio=math.sqrt(ev[1]/ev[0]);major_m=4*math.sqrt(ev[1])*cs2*2
    return ratio,major_m
# === pick zones ===
zones=[];tries=0
while len(zones)<N and tries<N*12+60:
    tries+=1;lon=random.uniform(BOX[0],BOX[2]);lat=random.uniform(BOX[1],BOX[3])
    if any((lon-z[0])**2+(lat-z[1])**2<(0.03)**2 for z in zones): continue
    est,nord=trans(lon,lat);nk=int(nord//1000);ek=int(est//1000)
    if not head200(nk,ek): continue
    d=load_one(nk,ek)
    if d is None or np.nanstd(d)<0.6 or np.isnan(d).mean()>0.4: continue  # plat/apă
    zones.append((round(lon,5),round(lat,5),nk,ek));print(f"zonă {len(zones)}: {round(lon,5)},{round(lat,5)} (tile {nk}_{ek})",flush=True)
if len(zones)<N: print(f"DOAR {len(zones)} zone acoperite găsite din {tries} încercări",flush=True)
# === scan each zone ===
allc=[]  # candidați compacți: dict
for zi,(clon,clat,_,_) in enumerate(zones,1):
    est,nord=trans(clon,clat);half=KM*1000/2
    e0=int((est-half)//1000);e1=int((est+half)//1000);n0=int((nord-half)//1000);n1=int((nord+half)//1000)
    xll0=e0*1000;ytop0=(n1+1)*1000;W=(e1-e0+1)*TPX;Hh=(n1-n0+1)*TPX
    mos=np.full((Hh,W),np.nan,np.float32);nt=0
    for nk in range(n0,n1+1):
        for ek in range(e0,e1+1):
            d=load_one(nk,ek)
            if d is None: continue
            nt+=1;ox=int((ek*1000-xll0)/CS);oy=int((ytop0-(nk+1)*1000)/CS);mos[oy:oy+TPX,ox:ox+TPX]=d[:TPX,:TPX]
    f=int(round(2.0/CS));hw=int(40/CS)
    def stamp(px,py,m):
        h=int(m/2/CS);w=mos[py-h:py+h,px-h:px+h]
        if w.shape!=(2*h,2*h) or np.isnan(w).mean()>0.05: return None
        d2=downs(w,f);sh=hs(d2,CS*f);lo,hi=np.percentile(sh,2),np.percentile(sh,98)
        if hi-lo<1e-6: return None
        return homog(np.asarray(Image.fromarray(np.clip((sh-lo)/(hi-lo)*255,0,255).astype('uint8')).resize((128,128)),np.uint8))
    STEP=int(10/CS);gxs=list(range(hw,W-hw,STEP));gys=list(range(hw,Hh-hw,STEP))
    best=np.zeros((len(gys),len(gxs)),np.float32)
    for M in SCALES:
        batch=[];pos=[]
        for iy,py in enumerate(gys):
            for ix,px in enumerate(gxs):
                s=stamp(px,py,M)
                if s is not None: batch.append(s);pos.append((iy,ix))
        if not batch: continue
        X=torch.tensor(np.array(batch,dtype=np.uint8));sc=[]
        with torch.no_grad():
            for k in range(0,len(X),512): sc.extend(torch.sigmoid(net(X[k:k+512].unsqueeze(1).float().to(dev)/255.)).cpu().numpy().tolist())
        g=np.zeros_like(best)
        for (iy,ix),v in zip(pos,sc): g[iy,ix]=v
        best=np.maximum(best,g)
    # render heatmap + clean
    field=np.asarray(Image.fromarray((best*255).astype('uint8')).resize((W,Hh),Image.BICUBIC),np.float32)/255.
    bgs=hs(np.nan_to_num(mos,nan=float(np.nanmin(mos))),CS);bg=np.clip((bgs-np.percentile(bgs,2))/(np.percentile(bgs,98)-np.percentile(bgs,2)+1e-6),0,1)
    clean=(bg*255).astype(np.uint8)
    rgb=(np.stack([bg,bg,bg],-1)*255).astype(np.float32);col=jet(field)*255;alpha=(np.clip((field-0.5)/0.5,0,1)*0.75)[...,None]
    heat=(rgb*(1-alpha)+col*alpha).astype(np.uint8)
    Image.fromarray(clean).save(f"/tmp/scan_{zi}_clean.png");Image.fromarray(heat).save(f"/tmp/scan_{zi}_heat.png")
    # NMS detections >=0.9
    flat=[(best[iy,ix],gxs[ix],gys[iy]) for iy in range(len(gys)) for ix in range(len(gxs)) if best[iy,ix]>=0.90]
    flat.sort(reverse=True);kept=[]
    for s,px,py in flat:
        if all((px-q[1])**2+(py-q[2])**2>(80/CS)**2 for q in kept): kept.append((s,px,py))
    ncomp=0
    for s,px,py in kept:
        h=int(160/2/CS);w=mos[py-h:py+h,px-h:px+h]
        if w.shape!=(2*h,2*h) or np.isnan(w).mean()>0.1: continue
        ratio,major=compactness(w)
        if major>=150 or ratio>=4.0: continue  # liniar -> FP, sar
        # filtru coerență direcțională (tunat 24.06): taie arătură/șanț (coh22>0.70), Catane 91→100%, 0 movile pierdute
        r22=int(22/CS);cw=np.nan_to_num(mos[py-r22:py+r22,px-r22:px+r22],nan=float(np.nanmedian(mos[py-r22:py+r22,px-r22:px+r22])))
        if cw.shape==(2*r22,2*r22):
            gy2,gx2=np.gradient(cw);Jxx=(gx2*gx2).mean();Jyy=(gy2*gy2).mean();Jxy=(gx2*gy2).mean();den=Jxx+Jyy
            if den>1e-12 and math.sqrt((Jxx-Jyy)**2+4*Jxy*Jxy)/den>0.70: continue  # direcțional -> FP, sar
        e=xll0+px*CS;nn_=ytop0-py*CS
        allc.append({'zi':zi,'px':px,'py':py,'score':s,'est':e,'nord':nn_,'W':W,'Hh':Hh});ncomp+=1
    print(f"  zona {zi}: {nt} tile, {len(kept)} detecții>=0.9, {ncomp} COMPACTE (candidați)",flush=True)
# coords pt candidați (batch)
if allc:
    pts="".join(f"{c['est']} {c['nord']}\n" for c in allc)
    rr=subprocess.run([GT,"-s_srs","EPSG:3844","-t_srs","EPSG:4326"],input=pts,capture_output=True,text=True,env=ENV)
    lls=[l.split()[:2] for l in rr.stdout.strip().split("\n")]
    for c,ll in zip(allc,lls): c['lon']=float(ll[0]);c['lat']=float(ll[1])
# board-uri PAGINATE: perechi clean|heatmap, sortate pe scor desc (TOATE, fără cap)
allc.sort(key=lambda c:-c['score'])
OUTPREF=os.environ.get('OUTPREF','scan5');NP=24;cw=190;pair=cw*2+6;cols=4;band=20;hh=int(160/2/CS)
opens={}
def getimg(p):
    if p not in opens: opens[p]=Image.open(p).convert('RGB')
    return opens[p]
nb=max(1,(len(allc)+NP-1)//NP)
try: fnt=ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf",15)
except: fnt=ImageFont.load_default()
with open(f'/tmp/{OUTPREF}_map.csv','w',newline='') as f:
    w=csv.writer(f);w.writerow(['idx','board','zona','lon','lat','score'])
    for b in range(nb):
        chunk=allc[b*NP:(b+1)*NP];rows=max(1,(len(chunk)+cols-1)//cols)
        cv=Image.new('RGB',(cols*pair+(cols+1)*6,rows*(cw+band)+6),(12,12,12));d2=ImageDraw.Draw(cv)
        for j,c in enumerate(chunk):
            gi=b*NP+j+1
            cl=getimg(f"/tmp/scan_{c['zi']}_clean.png");ht=getimg(f"/tmp/scan_{c['zi']}_heat.png")
            px,py=c['px'],c['py'];bb=(max(0,px-hh),max(0,py-hh),min(c['W'],px+hh),min(c['Hh'],py+hh))
            ci=cl.crop(bb).resize((cw,cw));hi=ht.crop(bb).resize((cw,cw))
            col=(j%cols)*pair+(j%cols+1)*6;row=(j//cols)*(cw+band)+band
            cv.paste(ci,(col,row));cv.paste(hi,(col+cw+6,row))
            d2.text((col+3,row-18),f"#{gi} z{c['zi']} {c['score']:.2f}",fill=(120,255,120),font=fnt)
            w.writerow([gi,b+1,c['zi'],round(c['lon'],5),round(c['lat'],5),round(c['score'],3)])
        cv.save(f"{H}/review/{OUTPREF}_board_{b+1:02d}.png");print(f"board {b+1}: {len(chunk)} cand -> review/{OUTPREF}_board_{b+1:02d}.png",flush=True)
print(f"\nTOTAL candidați: {len(allc)} în {nb} board-uri -> /tmp/{OUTPREF}_map.csv");print("zone:",[(z[0],z[1]) for z in zones])
