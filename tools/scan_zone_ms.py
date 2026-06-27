#!/usr/bin/env python3
# scan_zone.py CLON CLAT KM [MODEL] — survey productie pe o zona: scaneaza (model hillshade single-scale 80m),
# NMS, filtru coerenta (coh22>0.70) + filtru CURBURA (gate 0.70), apoi:
#   /tmp/zone_dets.csv (lon,lat,score,coh,pgate,keep)
#   review/zone_view.jpg  (hillshade + heatmap cald + marcaje: VERDE=candidat pastrat, ×gri=suprimat)
#   review/zone_board.jpg (crop-uri hillshade ale candidatilor PASTRATI, numerotate)
import os,sys,math,subprocess,csv
import numpy as np
from PIL import Image,ImageFilter,ImageDraw,ImageFont
import torch,torch.nn as nn
H=os.path.expanduser('~/lidar-match');dev=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
CLON=float(sys.argv[1]);CLAT=float(sys.argv[2]);KM=float(sys.argv[3]) if len(sys.argv)>3 else 4.0
MODEL=sys.argv[4] if len(sys.argv)>4 else f'{H}/combined_cnn.pt'
CACHE="/tmp/laki3";CS=0.5;TPX=2000;os.makedirs(CACHE,exist_ok=True)
APP="/Applications/QGIS-final-4_0_3.app/Contents"
ENV=dict(os.environ,DYLD_FRAMEWORK_PATH=f"{APP}/Frameworks",PROJ_DATA=f"{APP}/Resources/qgis/proj",PROJ_LIB=f"{APP}/Resources/qgis/proj",GDAL_DATA=f"{APP}/Resources/qgis/gdal")
GTb=f"{APP}/MacOS/gdaltransform"
def trans(pts,s,t):
    inp="\n".join(f"{a} {b}" for a,b in pts)+"\n";r=subprocess.run([GTb,"-s_srs",s,"-t_srs",t],input=inp,capture_output=True,text=True,env=ENV)
    return [tuple(map(float,l.split()[:2])) for l in r.stdout.strip().split("\n") if l.split()]
def load_one(nk,ek):
    p=f"{CACHE}/{nk}_{ek}.npy"
    if os.path.exists(p):
        try:return np.load(p)
        except:pass
    z=f"{CACHE}/{nk}_{ek}.zip"
    if not os.path.exists(z):subprocess.run(["curl","-s","--max-time","120","-o",z,f"https://geoportal.ancpi.ro/laki3_mnt/zip/{nk}_{ek}.zip"],check=False)
    try:import zipfile;zf=zipfile.ZipFile(z)
    except:
        if os.path.exists(z):os.remove(z)
        return None
    asc=[n for n in zf.namelist() if n.lower().endswith('.asc')]
    if not asc:return None
    raw=zf.read(asc[0]).decode('latin-1').replace(',','.');lines=raw.split('\n');hdr={};i=0
    while i<len(lines):
        pp=lines[i].split()
        if len(pp)>=2 and pp[0].lower() in ('ncols','nrows','xllcorner','yllcorner','cellsize','nodata_value'):hdr[pp[0].lower()]=float(pp[1]);i+=1
        else:break
    nc=int(hdr['ncols']);nr=int(hdr['nrows']);nd=hdr.get('nodata_value',-9999)
    d=np.fromstring(' '.join(lines[i:]),sep=' ',dtype=np.float32)[:nc*nr].reshape(nr,nc);d[d==nd]=np.nan;np.save(p,d);return d
def hs(dem,cs,azs=(315,45,135,225,270,0),alt=35):
    gy,gx=np.gradient(dem,cs);sl=np.arctan(np.hypot(gx,gy));asp=np.arctan2(-gy,gx);o=np.zeros_like(dem);ar=math.radians(alt)
    for az in azs: azr=math.radians(360-az+90);o+=np.clip(np.sin(ar)*np.cos(sl)+np.cos(ar)*np.sin(sl)*np.cos(azr-asp),0,1)
    return o/len(azs)
def downs(a,fc):
    Hh,Ww=a.shape;return a[:Hh//fc*fc,:Ww//fc*fc].reshape(Hh//fc,fc,Ww//fc,fc).mean((1,3))
def homog(a):
    a2=np.asarray(Image.fromarray(a).filter(ImageFilter.GaussianBlur(0.8)),np.uint8);cdf=np.bincount(a2.ravel(),minlength=256).astype(np.float64).cumsum()
    return (cdf[a2]/cdf[-1]*255).astype(np.uint8) if cdf[-1]>0 else a2
est,nord=trans([(CLON,CLAT)],"EPSG:4326","EPSG:3844")[0];half=KM*1000/2
e0=int((est-half)//1000);e1=int((est+half)//1000);n0=int((nord-half)//1000);n1=int((nord+half)//1000)
xll0=e0*1000;ytop0=(n1+1)*1000;W=(e1-e0+1)*TPX;Hh=(n1-n0+1)*TPX;mos=np.full((Hh,W),np.nan,np.float32);nt=0
for nk in range(n0,n1+1):
    for ek in range(e0,e1+1):
        d=load_one(nk,ek)
        if d is None:continue
        nt+=1;ox=int((ek*1000-xll0)/CS);oy=int((ytop0-(nk+1)*1000)/CS);mos[oy:oy+TPX,ox:ox+TPX]=d[:TPX,:TPX]
if nt==0:print("EROARE: nicio dala LAKI3 (zona neacoperita?)");sys.exit(2)
area=np.isfinite(mos).sum()*CS*CS/1e6
print(f"mozaic {W}x{Hh} ({KM}km, {nt} dale, ~{area:.1f}km²) model {os.path.basename(MODEL)}",flush=True)
f=int(round(2.0/CS));hw=int(40/CS)
class Net(nn.Module):
    def __init__(s):
        super().__init__();s.c=nn.Sequential(nn.Conv2d(1,16,3,2,1),nn.ReLU(),nn.Conv2d(16,32,3,2,1),nn.ReLU(),nn.Conv2d(32,64,3,2,1),nn.ReLU(),nn.AdaptiveAvgPool2d(1));s.f=nn.Linear(64,1)
    def forward(s,x): return s.f(s.c(x).flatten(1)).squeeze(1)
net=Net().to(dev);net.load_state_dict(torch.load(MODEL,map_location=dev,weights_only=True));net.eval()
SCALES=[float(x) for x in os.environ.get('SCALES','40,52,68,80').split(',')]
step=int(12/CS)
gpos=[(px,py) for py in range(hw,Hh-hw,step) for px in range(hw,W-hw,step)]
print(f"{len(gpos)} poziții × {len(SCALES)} scări {SCALES}; scorez multi-scară (max-pool)...",flush=True)
scmax=np.zeros(len(gpos),np.float32)
for S in SCALES:
    hwS=int(S/2/CS);batch=[];idxs=[]
    for gi,(px,py) in enumerate(gpos):
        w=mos[py-hwS:py+hwS,px-hwS:px+hwS]
        if w.shape!=(2*hwS,2*hwS) or np.isnan(w).mean()>0.05:continue
        d2=downs(w,f);h=hs(d2,CS*f);lo,hi=np.percentile(h,2),np.percentile(h,98)
        if hi-lo<1e-6:continue
        batch.append(homog(np.asarray(Image.fromarray(np.clip((h-lo)/(hi-lo)*255,0,255).astype('uint8')).resize((128,128)),np.uint8)));idxs.append(gi)
    if batch:
        X=torch.tensor(np.array(batch,dtype=np.uint8));vals=[]
        with torch.no_grad():
            for k in range(0,len(X),512):vals.extend(torch.sigmoid(net(X[k:k+512].unsqueeze(1).float().to(dev)/255.)).cpu().numpy().tolist())
        for gi,v in zip(idxs,vals): scmax[gi]=max(scmax[gi],v)
    print(f"  scara {S:.0f}m: {len(batch)} ferestre",flush=True)
pos=[gpos[i] for i in range(len(gpos)) if scmax[i]>0]
sc=np.array([scmax[i] for i in range(len(gpos)) if scmax[i]>0])
# NMS 80m, colectez detectii >=0.5 (pt heatmap) si candidati >=0.85
order=np.argsort(-sc);kept=[]
for k in order:
    if sc[k]<0.5:break
    px,py=pos[k]
    if any((px-q[0])**2+(py-q[1])**2<(80/CS)**2 for q in kept):continue
    kept.append((px,py,float(sc[k])))
print(f"{len(kept)} detectii (NMS 80m, >=0.5)",flush=True)
# coerenta
def coh22(px,py,r_m=22):
    r=int(r_m/CS);w=mos[py-r:py+r,px-r:px+r]
    if w.shape!=(2*r,2*r) or np.isnan(w).mean()>0.1:return 0.0
    w=np.nan_to_num(w,nan=np.nanmedian(w));gy,gx=np.gradient(w);Jxx=(gx*gx).mean();Jyy=(gy*gy).mean();Jxy=(gx*gy).mean();den=Jxx+Jyy
    return math.sqrt((Jxx-Jyy)**2+4*Jxy*Jxy)/den if den>1e-12 else 0.0
# lon/lat pt fiecare detectie
lls=trans([(xll0+px*CS,ytop0-py*CS) for px,py,s in kept],"EPSG:3844","EPSG:4326")
# gate curbura: scriu candidatii, chem curv_filter
with open('/tmp/_zone_in.csv','w',newline='') as fo:
    w=csv.writer(fo);w.writerow(['lon','lat'])
    for lo,la in lls:w.writerow([lo,la])
subprocess.run([f"{H}/venv/bin/python",f"{H}/tools/curv_filter.py","/tmp/_zone_in.csv","/tmp/_zone_gate.csv",f"{H}/curv_gate.json","0.70"],check=False)
gate=list(csv.DictReader(open('/tmp/_zone_gate.csv')))
rows=[]
for (px,py,s),(lo,la),g in zip(kept,lls,gate):
    coh=coh22(px,py);pg=float(g['pgate']) if g['pgate']!='NA' else 1.0
    keep = (s>=float(os.environ.get('CANDTHR','0.85'))) and (coh<=0.70) and (pg>=0.70)
    rows.append(dict(px=px,py=py,score=s,lon=lo,lat=la,coh=coh,pgate=pg,keep=keep))
with open('/tmp/zone_dets.csv','w',newline='') as fo:
    w=csv.writer(fo);w.writerow(['lon','lat','score','coh','pgate','keep'])
    for r in rows:w.writerow([f"{r['lon']:.6f}",f"{r['lat']:.6f}",f"{r['score']:.3f}",f"{r['coh']:.3f}",f"{r['pgate']:.3f}",int(r['keep'])])
cand=[r for r in rows if r['score']>=0.85]
keptc=[r for r in rows if r['keep']]
print(f"candidati >=0.85: {len(cand)} -> dupa filtre (coerenta+curbura): {len(keptc)} pastrati",flush=True)
# ===== HEATMAP wide =====
DW=1600;fac=max(1,W//DW);dw=W//fac;dh=Hh//fac
dem=mos[:dh*fac,:dw*fac].reshape(dh,fac,dw,fac).mean((1,3));dem=np.nan_to_num(dem,nan=np.nanmedian(dem))
shd=hs(dem,CS*fac);lo,hi=np.percentile(shd,2),np.percentile(shd,98);shg=np.clip((shd-lo)/(hi-lo)*255,0,255).astype(np.uint8)
basef=np.stack([shg]*3,-1).astype(np.float32)
acc=np.zeros((dh,dw),np.float32)
for px,py,s in kept:
    if s<0.7:continue
    x=px//fac;y=py//fac
    if 0<=x<dw and 0<=y<dh:acc[y,x]=max(acc[y,x],s)
accb=np.asarray(Image.fromarray((acc*255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(4)),np.float32)/255.;accb=accb/(accb.max()+1e-9)
warm=np.zeros((dh,dw,3),np.float32);warm[...,0]=np.clip(accb*3,0,1);warm[...,1]=np.clip(accb*1.6-0.3,0,1);al=np.clip(accb*1.4,0,1)[...,None]
disp=(basef*(1-al*0.5)+warm*255*al*0.5).astype(np.uint8)
img=Image.fromarray(disp).convert('RGB');dr=ImageDraw.Draw(img)
try:ft=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial Bold.ttf',16);ftb=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial Bold.ttf',22)
except:ft=ftb=ImageFont.load_default()
for r in cand:
    x=r['px']//fac;y=r['py']//fac
    if r['keep']:dr.ellipse([x-11,y-11,x+11,y+11],outline=(0,255,0),width=4)
    else:dr.line([x-7,y-7,x+7,y+7],fill=(140,140,150),width=2);dr.line([x-7,y+7,x+7,y-7],fill=(140,140,150),width=2)
for i,r in enumerate([c for c in cand if c['keep']],1):
    x=r['px']//fac;y=r['py']//fac;dr.text((x+12,y-9),f"{i}",fill=(150,255,150),font=ft)
hd=Image.new('RGB',(img.size[0],52),(10,10,12));hdr_d=ImageDraw.Draw(hd)
hdr_d.text((8,4),f"SCAN {CLAT},{CLON} ~{area:.0f}km² — model producție + filtre coerență&curbură",fill=(255,255,255),font=ftb)
hdr_d.text((8,30),f"VERDE = {len(keptc)} candidați păstrați (movile probabile)   ×gri = {len(cand)-len(keptc)} suprimate (FP: linii/mușuroaie aspre)",fill=(150,255,150),font=ft)
out=Image.new('RGB',(img.size[0],img.size[1]+52),(10,10,12));out.paste(hd,(0,0));out.paste(img,(0,52))
out.save(f'{H}/review/zone_view.jpg',quality=84)
print(f"-> review/zone_view.jpg {out.size} ({os.path.getsize(H+'/review/zone_view.jpg')//1024}KB)")
# ===== BOARD candidati pastrati =====
keptc_sorted=sorted(keptc,key=lambda r:-r['score'])[:48]
def crop(px,py,m=160,o=150):
    h=int(m/2/CS);w=mos[py-h:py+h,px-h:px+h]
    if w.shape!=(2*h,2*h) or np.isnan(w).mean()>0.1:return None
    sh2=hs(np.nan_to_num(w,nan=float(np.nanmin(w))),CS);l2,h2=np.percentile(sh2,2),np.percentile(sh2,98)
    return np.asarray(Image.fromarray(np.clip((sh2-l2)/(h2-l2+1e-6)*255,0,255).astype('uint8')).resize((o,o)),np.uint8)
if keptc_sorted:
    cols=8;rows_n=(len(keptc_sorted)+cols-1)//cols;c2=150;bv=Image.new('RGB',(cols*c2,rows_n*(c2+20)+24),(15,15,15));bd=ImageDraw.Draw(bv)
    bd.text((6,4),f"Candidați păstrați (movile probabile) — {CLAT},{CLON}",fill=(150,255,150),font=ft)
    for i,r in enumerate(keptc_sorted):
        cr=crop(r['px'],r['py']);x=(i%cols)*c2;y=(i//cols)*(c2+20)+24
        if cr is not None:bv.paste(Image.fromarray(cr),(x,y+16))
        bd.text((x+3,y+1),f"{i+1}  s{r['score']:.2f}",fill=(150,255,150),font=ft)
    bv.save(f'{H}/review/zone_board.jpg',quality=85);print(f"-> review/zone_board.jpg ({len(keptc_sorted)} candidați)")
else:
    print("(0 candidați păstrați — zonă curată / fără movile evidente)")
