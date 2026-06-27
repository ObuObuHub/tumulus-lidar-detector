#!/usr/bin/env python3
# tune_feat_ro.py — dump features (coerență la raze multiple + liniaritate) pt punctele de test RO din cache LAKI3.
# POZITIVI: 10 movile Catane (truth) + sample dataset_pos (tumuli confirmați RO). NEGATIVI: sample batch hard-neg
# (fals-pozitive marcate de Andrei, toate detectate ≥0.9). -> /tmp/ro_feat.csv (label,kind,lon,lat,coh15,coh22,coh30,coh40,lin)
import os,sys,math,subprocess,csv,glob,random
import numpy as np
CACHE="/tmp/laki3";CS=0.5;TPX=2000
APP=os.environ.get("QGIS_APP","/Applications/QGIS-final-4_0_3.app/Contents")
ENV=dict(os.environ,DYLD_FRAMEWORK_PATH=f"{APP}/Frameworks",PROJ_DATA=f"{APP}/Resources/qgis/proj",PROJ_LIB=f"{APP}/Resources/qgis/proj",GDAL_DATA=f"{APP}/Resources/qgis/gdal")
GT=f"{APP}/MacOS/gdaltransform"
def trans_many(pts):
    inp="".join(f"{lo} {la}\n" for lo,la in pts)
    r=subprocess.run([GT,"-s_srs","EPSG:4326","-t_srs","EPSG:3844"],input=inp,capture_output=True,text=True,env=ENV)
    return [tuple(map(float,l.split()[:2])) for l in r.stdout.strip().split("\n")] if r.stdout.strip() else []
def boxblur(a,r):
    ii=np.zeros((a.shape[0]+1,a.shape[1]+1));ii[1:,1:]=np.cumsum(np.cumsum(a,0),1)
    H2,W2=a.shape;ys=np.arange(H2);xs=np.arange(W2)
    y0=np.clip(ys-r,0,H2);y1=np.clip(ys+r+1,0,H2);x0=np.clip(xs-r,0,W2);x1=np.clip(xs+r+1,0,W2)
    return (ii[y1][:,x1]-ii[y0][:,x1]-ii[y1][:,x0]+ii[y0][:,x0])/((y1-y0)[:,None]*(x1-x0)[None,:])
WINm=200.0;HALF=int(WINm/CS)
def local_window(est,nord,half):
    e0=int((est-WINm)//1000);e1=int((est+WINm)//1000);n0=int((nord-WINm)//1000);n1=int((nord+WINm)//1000)
    xll0=e0*1000;ytop0=(n1+1)*1000;W=(e1-e0+1)*TPX;Hh=(n1-n0+1)*TPX
    mos=np.full((Hh,W),np.nan,np.float32);got=False
    for nk in range(n0,n1+1):
        for ek in range(e0,e1+1):
            p=f"{CACHE}/{nk}_{ek}.npy"
            if not os.path.exists(p):continue
            d=np.load(p);ox=int((ek*1000-xll0)/CS);oy=int((ytop0-(nk+1)*1000)/CS);mos[oy:oy+TPX,ox:ox+TPX]=d[:TPX,:TPX];got=True
    if not got: return None
    px=int((est-xll0)/CS);py=int((ytop0-nord)/CS)
    return mos[py-half:py+half,px-half:px+half]
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
def lin_ratio(w):  # w = fereastră centrată, full 2*HALF
    slrm=w-boxblur(w,int(200/CS));thr=slrm.mean()+1.0*slrm.std();mask=slrm>thr
    region=cc_seed(mask,w.shape[0]//2,w.shape[1]//2)
    if region is None or region.sum()<20: return 1.0
    ys,xs=np.nonzero(region);mx,my=xs.mean(),ys.mean()
    mxx=((xs-mx)**2).mean();myy=((ys-my)**2).mean();mxy=((xs-mx)*(ys-my)).mean()
    tr=mxx+myy;s=math.sqrt(max(0,tr*tr/4-(mxx*myy-mxy*mxy)));l1=tr/2+s;l2=tr/2-s
    return math.sqrt(l1/max(l2,1e-6))
def coh(w,rad_m):
    c=w.shape[0]//2;r=int(rad_m/CS);ww=w[c-r:c+r,c-r:c+r]
    gy,gx=np.gradient(ww);Jxx=(gx*gx).mean();Jyy=(gy*gy).mean();Jxy=(gx*gy).mean();den=Jxx+Jyy
    return math.sqrt((Jxx-Jyy)**2+4*Jxy*Jxy)/den if den>1e-12 else 0.0
# coords
TRUTH={11,17,18,30,43,45,50,55,57,64}
pos=[];neg=[]
em={int(r['idx']):(float(r['lon']),float(r['lat'])) for r in csv.DictReader(open('/tmp/eval_map.csv'))}
for i in TRUTH:
    if i in em: pos.append((em[i][0],em[i][1],f'catane{i}'))
# Catane FP: punctele non-truth cu scor mare (model le aprinde) = negativi „catfp" pt precizia Catane directă
sc_eval={int(r['idx']):float(r['score']) for r in csv.DictReader(open('/tmp/eval_rescore.csv')) if r['score'] not in ('NA','')}
for i,(lo,la) in em.items():
    if i not in TRUTH and sc_eval.get(i,0)>=0.5: neg.append((lo,la,f'catfp{i}'))
# pozitivi confirmați RO (labeled/labels.csv: gold_ran + expert_validated), DOAR footprint LAKI3 (Oltenia)
def in_laki(lo,la): return 22.85<=lo<=24.05 and 43.78<=la<=44.55
random.seed(7)
dp=[(float(r['lon']),float(r['lat'])) for r in csv.DictReader(open('labeled/labels.csv'))
    if r.get('source') in ('gold_ran','expert_validated') and in_laki(float(r['lon']),float(r['lat']))]
random.shuffle(dp)
for lo,la in dp[:50]: pos.append((lo,la,'ropos'))
# batch hard-neg (FP marcate)
hn=[]
for f in sorted(glob.glob('labeled/batch*_hardneg.csv')):
    for r in csv.DictReader(open(f)): hn.append((float(r['lon']),float(r['lat'])))
random.shuffle(hn)
for lo,la in hn[:450]: neg.append((lo,la,'batchfp'))
allp=[(lo,la,k,'pos') for lo,la,k in pos]+[(lo,la,k,'neg') for lo,la,k in neg]
EN=trans_many([(lo,la) for lo,la,_,_ in allp])
out=open('/tmp/ro_feat.csv','w');w=csv.writer(out);w.writerow(['label','kind','lon','lat','coh15','coh22','coh30','coh40','lin'])
npos=nneg=skip=0
for (lo,la,k,lab),(e,n) in zip(allp,EN):
    win=local_window(e,n,HALF)
    if win is None or win.shape!=(2*HALF,2*HALF) or np.isnan(win).mean()>0.1: skip+=1; continue
    win=np.nan_to_num(win,nan=np.nanmedian(win))
    c15,c22,c30,c40=coh(win,15),coh(win,22),coh(win,30),coh(win,40);lr=lin_ratio(win)
    w.writerow([lab,k,f'{lo:.5f}',f'{la:.5f}',f'{c15:.3f}',f'{c22:.3f}',f'{c30:.3f}',f'{c40:.3f}',f'{lr:.2f}'])
    if lab=='pos': npos+=1
    else: nneg+=1
out.close()
print(f"RO features: {npos} pozitivi + {nneg} negativi ({skip} skip cache) -> /tmp/ro_feat.csv",flush=True)
