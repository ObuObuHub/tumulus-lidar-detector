#!/usr/bin/env python3
# spatial_diag.py — diagnostic prior SPAȚIAL (WODAN): separă FP reziduale de movile reale folosind
# POZIȚIA în relief (TPI regional) + CLUSTERIZAREA (vecini gate-passing)? Pe detecțiile Catane gate-passing.
import os,math,subprocess,csv
import numpy as np
H=os.path.expanduser('~/lidar-match');CACHE="/tmp/laki3";CS=0.5;TPX=2000
APP="/Applications/QGIS-final-4_0_3.app/Contents"
ENV=dict(os.environ,DYLD_FRAMEWORK_PATH=f"{APP}/Frameworks",PROJ_DATA=f"{APP}/Resources/qgis/proj",PROJ_LIB=f"{APP}/Resources/qgis/proj",GDAL_DATA=f"{APP}/Resources/qgis/gdal")
GTb=f"{APP}/MacOS/gdaltransform"
def trans(pts,s,t):
    inp="\n".join(f"{a} {b}" for a,b in pts)+"\n";r=subprocess.run([GTb,"-s_srs",s,"-t_srs",t],input=inp,capture_output=True,text=True,env=ENV)
    return [tuple(map(float,l.split()[:2])) for l in r.stdout.strip().split("\n") if l.split()]
# detecții gate-passing la pragul recall-100 (folosesc v3 gated)
rows=list(csv.DictReader(open('/tmp/dets_v3_gated.csv')))
s1=np.array([float(r['score']) for r in rows]);istp=np.array([int(r['istp']) for r in rows])
pg=np.array([float(r['pgate']) if r['pgate']!='NA' else 1.0 for r in rows])
thr=s1[(istp==1)&(pg>=0.70)].min()
keep=(s1>=thr)&(pg>=0.70)
sub=[(float(rows[i]['lon']),float(rows[i]['lat']),istp[i]) for i in range(len(rows)) if keep[i]]
print(f"detecții gate-passing @recall100: {len(sub)} (TP {sum(1 for x in sub if x[2]==1)}, FP {sum(1 for x in sub if x[2]==0)})")
en=trans([(lo,la) for lo,la,_ in sub],"EPSG:4326","EPSG:3844")
# mosaic Catane
es=[e for e,n in en];ns=[n for e,n in en];MARG=1400
e0=int((min(es)-MARG)//1000);e1=int((max(es)+MARG)//1000);n0=int((min(ns)-MARG)//1000);n1=int((max(ns)+MARG)//1000)
xll0=e0*1000;ytop0=(n1+1)*1000;W=(e1-e0+1)*TPX;Hh=(n1-n0+1)*TPX;mos=np.full((Hh,W),np.nan,np.float32)
for nk in range(n0,n1+1):
    for ek in range(e0,e1+1):
        p=f"{CACHE}/{nk}_{ek}.npy"
        if os.path.exists(p): d=np.load(p);ox=int((ek*1000-xll0)/CS);oy=int((ytop0-(nk+1)*1000)/CS);mos[oy:oy+TPX,ox:ox+TPX]=d[:TPX,:TPX]
def tpi(e,n,rad_m):
    px=int((e-xll0)/CS);py=int((ytop0-n)/CS);r=int(rad_m/CS)
    w=mos[py-r:py+r,px-r:px+r]
    if w.shape!=(2*r,2*r): return np.nan
    c=float(mos[py,px]);return c-np.nanmean(w)
# features per detecție
feats=[]
en_arr=np.array(en)
for (lo,la,t),(e,n) in zip(sub,en):
    t300=tpi(e,n,300);t1000=tpi(e,n,1000)
    # clustering: vecini gate-passing în 400m (printre TOATE detecțiile gate-passing)
    d=np.hypot(en_arr[:,0]-e,en_arr[:,1]-n);nbr=int(((d<400)&(d>1)).sum());nn=np.sort(d[d>1])[0] if len(d)>1 else 9999
    feats.append((t,t300,t1000,nbr,nn))
F=np.array([f[1:] for f in feats]);Y=np.array([f[0] for f in feats])
names=['TPI_300m','TPI_1000m','vecini<400m','dist_vecin']
print(f"\n{'trasatura':14}{'movile(TP)':>14}{'FP':>10}{'separare':>10}")
for j,nm in enumerate(names):
    tp=F[Y==1,j];fp=F[Y==0,j];tp=tp[~np.isnan(tp)];fp=fp[~np.isnan(fp)]
    # AUC
    allv=np.concatenate([tp,fp])
    if len(tp) and len(fp):
        order=allv.argsort();ranks=np.empty(len(allv));ranks[order]=np.arange(1,len(allv)+1)
        auc=(ranks[:len(tp)].sum()-len(tp)*(len(tp)+1)/2)/(len(tp)*len(fp))
    else: auc=0.5
    print(f"  {nm:12}{np.median(tp):>12.2f}{np.median(fp):>12.2f}{auc:>10.2f}  (0.5=fără separare)")
