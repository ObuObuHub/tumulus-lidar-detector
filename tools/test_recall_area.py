#!/usr/bin/env python3
# test_recall_area.py CLON CLAT KM — TEST ONEST de recall+FP pe o zonă RO cu movile cunoscute (DTM 0.5m).
# 1) scorează fiecare movilă RAN/etichetată din zonă (recall, separat held-out vs train)
# 2) sweep peste zonă -> celule scor-mare departe de movile cunoscute = fals-pozitive (sau movile noi)
# 3) overlay: hillshade + verde=movile cunoscute (cu scor) + roșu=FP. Rată reală, nu AUROC umflat.
import os,sys,math,subprocess,zipfile,json,csv
import numpy as np
from PIL import Image,ImageDraw,ImageFont,ImageFilter
import torch,torch.nn as nn
H=os.path.dirname(os.path.dirname(os.path.abspath(__file__)));dev=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
CLON=float(sys.argv[1]);CLAT=float(sys.argv[2]);KM=float(sys.argv[3]) if len(sys.argv)>3 else 6.0
CACHE="/tmp/laki3";CS=0.5;TPX=2000;os.makedirs(CACHE,exist_ok=True)
APP=os.environ.get("QGIS_APP","/Applications/QGIS-final-4_0_3.app/Contents")
ENV=dict(os.environ,DYLD_FRAMEWORK_PATH=f"{APP}/Frameworks",PROJ_DATA=f"{APP}/Resources/qgis/proj",PROJ_LIB=f"{APP}/Resources/qgis/proj",GDAL_DATA=f"{APP}/Resources/qgis/gdal")
GT=f"{APP}/MacOS/gdaltransform"
def trans(pts,s,t):
    if not pts: return []
    inp="\n".join(f"{a} {b}" for a,b in pts)+"\n"
    r=subprocess.run([GT,"-s_srs",s,"-t_srs",t],input=inp,capture_output=True,text=True,env=ENV)
    return [(float(l.split()[0]),float(l.split()[1])) for l in r.stdout.strip().split("\n") if l.split()]
def to_st(lo,la): return trans([(lo,la)],"EPSG:4326","EPSG:3844")[0]
def dl(nk,ek):
    p=f"{CACHE}/{nk}_{ek}.npy"
    if os.path.exists(p): return np.load(p)
    z=f"{CACHE}/{nk}_{ek}.zip"
    if not os.path.exists(z): subprocess.run(["curl","-s","--max-time","60","-o",z,f"https://geoportal.ancpi.ro/laki3_mnt/zip/{nk}_{ek}.zip"],check=False)
    try: zf=zipfile.ZipFile(z);asc=[n for n in zf.namelist() if n.lower().endswith('.asc')]
    except: return None
    if not asc: return None
    raw=zf.read(asc[0]).decode('latin-1').replace(',','.');L=raw.split('\n');i=0
    while i<len(L) and L[i].split() and L[i].split()[0].lower() in('ncols','nrows','xllcorner','yllcorner','cellsize','nodata_value'):i+=1
    d=np.fromstring(' '.join(L[i:]),sep=' ',dtype=np.float32)[:TPX*TPX].reshape(TPX,TPX);d[d==-9999]=np.nan;np.save(p,d);return d
est,nord=to_st(CLON,CLAT);half=KM*1000/2
e0=int((est-half)//1000);e1=int((est+half)//1000);n0=int((nord-half)//1000);n1=int((nord+half)//1000)
xll0=e0*1000;ytop0=(n1+1)*1000;W=(e1-e0+1)*TPX;Hh=(n1-n0+1)*TPX
mos=np.full((Hh,W),np.nan,np.float32);got=0
for nk in range(n0,n1+1):
    for ek in range(e0,e1+1):
        d=dl(nk,ek)
        if d is None: continue
        got+=1;ox=int((ek*1000-xll0)/CS);oy=int((ytop0-(nk+1)*1000)/CS);mos[oy:oy+TPX,ox:ox+TPX]=d[:TPX,:TPX]
print(f"mozaic {W}x{Hh}px ({KM}km), {got} dale, goluri {np.isnan(mos).mean()*100:.0f}%",flush=True)
def hs(dem,cs,azs=(315,45,135,225,270,0),alt=35):
    gy,gx=np.gradient(dem,cs);sl=np.arctan(np.hypot(gx,gy));asp=np.arctan2(-gy,gx);o=np.zeros_like(dem);ar=math.radians(alt)
    for az in azs:
        azr=math.radians(360-az+90);o+=np.clip(np.sin(ar)*np.cos(sl)+np.cos(ar)*np.sin(sl)*np.cos(azr-asp),0,1)
    return o/len(azs)
def downs(a,fac):
    Hh2,Ww=a.shape;return a[:Hh2//fac*fac,:Ww//fac*fac].reshape(Hh2//fac,fac,Ww//fac,fac).mean((1,3))
f=int(round(2.0/CS));wpx=int(80/CS)
def stamp(px,py):
    w=mos[py-wpx//2:py+wpx//2,px-wpx//2:px+wpx//2]
    if w.shape!=(wpx,wpx) or np.isnan(w).mean()>0.05: return None
    d2=downs(w,f);h=hs(d2,CS*f);lo,hi=np.percentile(h,2),np.percentile(h,98)
    if hi-lo<1e-6: return None
    im=Image.fromarray(np.clip((h-lo)/(hi-lo)*255,0,255).astype('uint8')).resize((128,128)).filter(ImageFilter.GaussianBlur(0.8))
    a=np.asarray(im,np.uint8);cdf=np.bincount(a.ravel(),minlength=256).astype(np.float64).cumsum()  # OMOGENIZARE = la fel ca train
    if cdf[-1]>0: a=(cdf[a]/cdf[-1]*255).astype(np.uint8)
    return a.astype(np.float32)/255.
class Net(nn.Module):
    def __init__(s):
        super().__init__();s.c=nn.Sequential(nn.Conv2d(1,16,3,2,1),nn.ReLU(),nn.Conv2d(16,32,3,2,1),nn.ReLU(),nn.Conv2d(32,64,3,2,1),nn.ReLU(),nn.AdaptiveAvgPool2d(1));s.f=nn.Linear(64,1)
    def forward(s,x): return s.f(s.c(x).flatten(1)).squeeze(1)
net=Net().to(dev);net.load_state_dict(torch.load(f'{H}/combined_cnn.pt',weights_only=True));net.eval()
def score_batch(ims):
    if not ims: return []
    xb=torch.tensor(np.array(ims)).unsqueeze(1).float().to(dev)
    with torch.no_grad(): return list(torch.sigmoid(net(xb)).cpu().numpy())
# === movile cunoscute în zonă (RAN Dolj + labels), cu held-out flag ===
plan=json.load(open('/tmp/dolj_sweep_plan.json'))
known=[]  # (lon,lat,heldout,nume)
for p in plan['all']:
    lo,la,ho,loc,nume=p
    if abs(lo-CLON)<half/85000 and abs(la-CLAT)<half/111000: known.append((lo,la,ho,nume))
sts=trans([(k[0],k[1]) for k in known],"EPSG:4326","EPSG:3844")
# RECALL ONEST = detectare în vecinătate: scor MAX peste ferestre la <=50m (centroidul RAN nu cade
# mereu pe vârful movilei; măturând cu pasul, modelul oricum trece prin ferestre lângă movilă).
def best_score(e,n,rad=50,step=15):
    ims=[]
    for de in range(-rad,rad+1,step):
        for dn in range(-rad,rad+1,step):
            im=stamp(int((e+de-xll0)/CS),int((ytop0-(n+dn))/CS))
            if im is not None: ims.append(im)
    if not ims: return None
    return float(max(score_batch(ims)))
ho_hit=ho_tot=tr_hit=tr_tot=0;mound_marks=[]
for (lo,la,ho,nume),(e,n) in zip(known,sts):
    sc=best_score(e,n)
    if sc is None: continue
    mound_marks.append((e,n,sc,ho)); meta=mound_marks  # for printing below
    if ho: ho_tot+=1; ho_hit+=sc>=0.5
    else: tr_tot+=1; tr_hit+=sc>=0.5
printable=[]
for (lo,la,ho,nume),(e,n) in zip(known,sts):
    sc=next((m[2] for m in mound_marks if abs(m[0]-e)<1 and abs(m[1]-n)<1),None)
    if sc is not None: printable.append((ho,sc,nume))
print(f"\n=== RECALL pe movile cunoscute (max în vecinătate <=50m, prag 0.5) ===")
print(f"  HELD-OUT (netrainuite): {ho_hit}/{ho_tot}"+(f" = {100*ho_hit/ho_tot:.0f}%" if ho_tot else ""))
print(f"  train (sanity):         {tr_hit}/{tr_tot}"+(f" = {100*tr_hit/tr_tot:.0f}%" if tr_tot else ""))
for ho,sc,nume in sorted(printable,key=lambda z:-z[1]):
    print(f"   {'HELD-OUT' if ho else 'train   '} scor {sc:.2f}  {nume}")
# === sweep -> FP ===
stride=int(40/CS);ys=list(range(wpx//2,Hh-wpx//2,stride));xs=list(range(wpx//2,W-wpx//2,stride))
hi_cells=[];batch=[];pos=[]
def flush():
    global batch,pos
    if not batch: return
    for sc,(e,n) in zip(score_batch(batch),pos):
        if sc>=0.6: hi_cells.append((float(sc),e,n))
    batch=[];pos=[]
for yy in ys:
    for xx in xs:
        im=stamp(xx,yy)
        if im is None: continue
        e=xll0+xx*CS;n=ytop0-yy*CS;batch.append(im);pos.append((e,n))
        if len(batch)>=512: flush()
flush()
# dedup hi cells 80m + classify near known mound (<=120m) vs FP
hi_cells.sort(reverse=True);dedup=[]
for sc,e,n in hi_cells:
    if any((e-d[1])**2+(n-d[2])**2<80*80 for d in dedup): continue
    dedup.append((sc,e,n))
knownEN=[(m[0],m[1]) for m in mound_marks]
fp=[c for c in dedup if not any((c[1]-ke)**2+(c[2]-kn)**2<120*120 for ke,kn in knownEN)]
area=KM*KM
print(f"\n=== FALS-POZITIVE (sweep) ===")
print(f"  celule >0.6 dedup: {len(dedup)} | lângă movile cunoscute: {len(dedup)-len(fp)} | FP (departe): {len(fp)}")
print(f"  FP/km²: {len(fp)/area:.1f}")
# === EXPORT TOT ce a cotat modelul (dedup >0.6) pt evaluare manuală Andrei ===
det_ll=trans([(e,n) for sc,e,n in dedup],"EPSG:3844","EPSG:4326")
knownEN_set=knownEN
det=[]
for (sc,e,n),(lo,la) in zip(dedup,det_ll):
    neark=any((e-ke)**2+(n-kn)**2<120*120 for ke,kn in knownEN_set)
    det.append((sc,e,n,lo,la,neark))
import csv as _csv
with open('/tmp/catane_detections.csv','w',newline='') as fcsv:
    wcsv=_csv.writer(fcsv);wcsv.writerow(['idx','score','lon','lat','langa_movila_cunoscuta','gmaps'])
    for i,(sc,e,n,lo,la,nk) in enumerate(det,1):
        wcsv.writerow([i,f"{sc:.3f}",f"{lo:.5f}",f"{la:.5f}",int(nk),f"https://www.google.com/maps/@{la},{lo},400m/data=!3m1!1e3"])
print(f"  -> /tmp/catane_detections.csv ({len(det)} detecții, idx+coord+gmaps)")
# montaje numerotate cu stampa 80m (ce a văzut modelul) + context 160m
def stamp_ctx(px,py,meters):
    hw=int((meters/2)/CS);w=mos[py-hw:py+hw,px-hw:px+hw]
    if w.shape!=(2*hw,2*hw) or np.isnan(w).mean()>0.1: return None
    d2=downs(w,f);h=hs(d2,CS*f);lo2,hi2=np.percentile(h,2),np.percentile(h,98)
    if hi2-lo2<1e-6: return None
    return Image.fromarray(np.clip((h-lo2)/(hi2-lo2)*255,0,255).astype('uint8')).resize((120,120))
cols=8;per=64
for page in range((len(det)+per-1)//per):
    chunk=det[page*per:(page+1)*per];rows_=(len(chunk)+cols-1)//cols
    MM=Image.new('RGB',(cols*126,rows_*140),(15,15,15));d_=ImageDraw.Draw(MM)
    try: ft=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial Bold.ttf',12)
    except: ft=ImageFont.load_default()
    for j,(sc,e,n,lo,la,nk) in enumerate(chunk):
        idx=page*per+j+1;px=int((e-xll0)/CS);py=int((ytop0-n)/CS);im=stamp_ctx(px,py,160)
        x=(j%cols)*126;y=(j//cols)*140
        if im: MM.paste(im.convert('RGB'),(x+3,y+20))
        col=(0,200,255) if nk else (255,210,0)
        d_.text((x+3,y+4),f"#{idx} {sc:.2f}",fill=col,font=ft)
    MM.save(f'{H}/review/catane_detections_{page+1}.png');print(f"  -> review/catane_detections_{page+1}.png")
# === overlay render ===
view=900;sc_f=view/max(W,Hh)
base=np.where(np.isnan(mos),0,mos)
hh=hs(downs(base,f),CS*f);lo,hi=np.percentile(hh,2),np.percentile(hh,98)
img=Image.fromarray(np.clip((hh-lo)/(hi-lo)*255,0,255).astype('uint8')).resize((int(W*sc_f),int(Hh*sc_f))).convert('RGB')
dr=ImageDraw.Draw(img)
def topx(e,n): return (e-xll0)/CS*sc_f,(ytop0-n)/CS*sc_f
for e,n,sc,ho in mound_marks:
    x,y=topx(e,n);col=(0,255,0) if ho else (0,180,255)
    r=9;dr.ellipse([x-r,y-r,x+r,y+r],outline=col,width=3)
    dr.text((x+r,y-r),f"{sc:.2f}",fill=col)
for sc,e,n in fp:
    x,y=topx(e,n);dr.ellipse([x-5,y-5,x+5,y+5],outline=(255,40,40),width=2)
img.save(f'{H}/review/dolj_recall.png')
print("\nverde=movile held-out, albastru=movile train, roșu=fals-pozitive -> review/dolj_recall.png")
json.dump({'ho':[ho_hit,ho_tot],'tr':[tr_hit,tr_tot],'fp':len(fp),'fp_km2':len(fp)/area},open('/tmp/dolj_recall.json','w'))
