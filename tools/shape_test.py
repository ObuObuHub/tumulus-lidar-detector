#!/usr/bin/env python3
# shape_test.py CLON CLAT KM — calculează FILTRUL DE FORMĂ pentru fiecare celulă din /tmp/eval_map.csv:
# alungirea (raport axe) a reliefului central. Tumul = ROTUND (raport ~1-2); mal/terasă = ALUNGIT (>~2.5).
# SLRM (elev - blur local) -> prag -> moment de inerție central -> raport axe. Fără scipy.
# Iese /tmp/eval_shape.csv (idx,score,axis_ratio,round). De combinat cu marcajele lui Andrei.
import os,sys,math,subprocess,zipfile,csv
import numpy as np
H=os.path.dirname(os.path.dirname(os.path.abspath(__file__)));CACHE="/tmp/laki3";CS=0.5;TPX=2000
CLON=float(sys.argv[1]);CLAT=float(sys.argv[2]);KM=float(sys.argv[3]) if len(sys.argv)>3 else 5.0
RATIO_T=2.2  # peste = alungit = mal/terasă (resping)
APP=os.environ.get("QGIS_APP","/Applications/QGIS-final-4_0_3.app/Contents")
ENV=dict(os.environ,DYLD_FRAMEWORK_PATH=f"{APP}/Frameworks",PROJ_DATA=f"{APP}/Resources/qgis/proj",PROJ_LIB=f"{APP}/Resources/qgis/proj",GDAL_DATA=f"{APP}/Resources/qgis/gdal")
GT=f"{APP}/MacOS/gdaltransform"
def trans(p,s,t):
    r=subprocess.run([GT,"-s_srs",s,"-t_srs",t],input=f"{p[0]} {p[1]}\n",capture_output=True,text=True,env=ENV);a,b,_=r.stdout.split();return float(a),float(b)
def dl(nk,ek):
    p=f"{CACHE}/{nk}_{ek}.npy";return np.load(p) if os.path.exists(p) else None
def boxblur(a,r):
    ii=np.zeros((a.shape[0]+1,a.shape[1]+1));ii[1:,1:]=np.cumsum(np.cumsum(a,0),1)
    H2,W2=a.shape;ys=np.arange(H2);xs=np.arange(W2)
    y0=np.clip(ys-r,0,H2);y1=np.clip(ys+r+1,0,H2);x0=np.clip(xs-r,0,W2);x1=np.clip(xs+r+1,0,W2)
    return (ii[y1][:,x1]-ii[y0][:,x1]-ii[y1][:,x0]+ii[y0][:,x0])/((y1-y0)[:,None]*(x1-x0)[None,:])
est,nord=trans((CLON,CLAT),"EPSG:4326","EPSG:3844");half=KM*1000/2
e0=int((est-half)//1000);e1=int((est+half)//1000);n0=int((nord-half)//1000);n1=int((nord+half)//1000)
xll0=e0*1000;ytop0=(n1+1)*1000;W=(e1-e0+1)*TPX;Hh=(n1-n0+1)*TPX
mos=np.full((Hh,W),np.nan,np.float32)
for nk in range(n0,n1+1):
    for ek in range(e0,e1+1):
        d=dl(nk,ek)
        if d is None: continue
        ox=int((ek*1000-xll0)/CS);oy=int((ytop0-(nk+1)*1000)/CS);mos[oy:oy+TPX,ox:ox+TPX]=d[:TPX,:TPX]
hw=int(40/CS)  # 80m
def axis_ratio(e,n):
    px=int((e-xll0)/CS);py=int((ytop0-n)/CS);w=mos[py-hw:py+hw,px-hw:px+hw]
    if w.shape!=(2*hw,2*hw) or np.isnan(w).mean()>0.05: return None
    w=np.nan_to_num(w,nan=np.nanmedian(w))
    slrm=w-boxblur(w,int(24/CS))  # relief local (~mound)
    c=slrm[hw//2:hw+hw//2,hw//2:hw+hw//2]  # central 40m (focus pe trăsătura centrală)
    thr=c.mean()+1.0*c.std();mask=c>thr
    ys,xs=np.nonzero(mask)
    if len(xs)<15: return 99.0  # fără relief compact = resping
    cx,cy=xs.mean(),ys.mean();mxx=((xs-cx)**2).mean();myy=((ys-cy)**2).mean();mxy=((xs-cx)*(ys-cy)).mean()
    tr=mxx+myy;dd=tr*tr/4-(mxx*myy-mxy*mxy);s=math.sqrt(max(0,dd))
    l1=tr/2+s;l2=tr/2-s
    return math.sqrt(l1/max(l2,1e-6))
rows=list(csv.DictReader(open('/tmp/eval_map.csv')))
out=open('/tmp/eval_shape.csv','w');w=csv.writer(out);w.writerow(['idx','score','axis_ratio','round'])
for r in rows:
    e,n=trans((float(r['lon']),float(r['lat'])),"EPSG:4326","EPSG:3844")
    ar=axis_ratio(e,n)
    rnd=1 if (ar is not None and ar<=RATIO_T) else 0
    w.writerow([r['idx'],r['score'],f"{ar:.2f}" if ar else "NA",rnd])
out.close()
print(f"-> /tmp/eval_shape.csv (raport axe per celulă; round=1 dacă <= {RATIO_T})")
