#!/usr/bin/env python3
# linearity_test.py /tmp/lin_coords.csv -> /tmp/lin_out.csv
# Măsoară LINIARITATEA reliefului la scară largă pt fiecare coord (label,idx,lon,lat).
# Mal de canal/dig = creastă lungă subțire (raport axe MARE + lungime mare); tumul = movilă compactă.
# SLRM la scară largă -> prag -> componenta conexă care atinge centrul -> raport axe + lungime major în metri.
# Fără scipy. Reutilizează cache-ul /tmp/laki3 (dale 1km @0.5m).
import os,sys,math,subprocess,csv
import numpy as np
CACHE="/tmp/laki3";CS=0.5;TPX=2000
APP=os.environ.get("QGIS_APP","/Applications/QGIS-final-4_0_3.app/Contents")
ENV=dict(os.environ,DYLD_FRAMEWORK_PATH=f"{APP}/Frameworks",PROJ_DATA=f"{APP}/Resources/qgis/proj",PROJ_LIB=f"{APP}/Resources/qgis/proj",GDAL_DATA=f"{APP}/Resources/qgis/gdal")
GT=f"{APP}/MacOS/gdaltransform"
def trans(p,s,t):
    r=subprocess.run([GT,"-s_srs",s,"-t_srs",t],input=f"{p[0]} {p[1]}\n",capture_output=True,text=True,env=ENV);a,b,_=r.stdout.split();return float(a),float(b)
def boxblur(a,r):
    ii=np.zeros((a.shape[0]+1,a.shape[1]+1));ii[1:,1:]=np.cumsum(np.cumsum(a,0),1)
    H2,W2=a.shape;ys=np.arange(H2);xs=np.arange(W2)
    y0=np.clip(ys-r,0,H2);y1=np.clip(ys+r+1,0,H2);x0=np.clip(xs-r,0,W2);x1=np.clip(xs+r+1,0,W2)
    return (ii[y1][:,x1]-ii[y0][:,x1]-ii[y1][:,x0]+ii[y0][:,x0])/((y1-y0)[:,None]*(x1-x0)[None,:])

WIN=160.0  # rază fereastră (m); movila ~40m, malul de canal apare ca linie pe 160m
HALF=int(WIN/CS)  # px
def local_window(est,nord):
    e0=int((est-WIN)//1000);e1=int((est+WIN)//1000);n0=int((nord-WIN)//1000);n1=int((nord+WIN)//1000)
    xll0=e0*1000;ytop0=(n1+1)*1000;W=(e1-e0+1)*TPX;Hh=(n1-n0+1)*TPX
    mos=np.full((Hh,W),np.nan,np.float32)
    for nk in range(n0,n1+1):
        for ek in range(e0,e1+1):
            p=f"{CACHE}/{nk}_{ek}.npy"
            if not os.path.exists(p):continue
            d=np.load(p);ox=int((ek*1000-xll0)/CS);oy=int((ytop0-(nk+1)*1000)/CS);mos[oy:oy+TPX,ox:ox+TPX]=d[:TPX,:TPX]
    px=int((est-xll0)/CS);py=int((ytop0-nord)/CS)
    return mos[py-HALF:py+HALF,px-HALF:px+HALF]

def cc_from_seed(mask,sy,sx):
    # caută seed în mask în ±24px de centru
    if not mask[sy,sx]:
        best=None
        for dy in range(-24,25):
            for dx in range(-24,25):
                y,x=sy+dy,sx+dx
                if 0<=y<mask.shape[0] and 0<=x<mask.shape[1] and mask[y,x]:
                    dd=dy*dy+dx*dx
                    if best is None or dd<best[0]:best=(dd,y,x)
        if best is None:return None
        sy,sx=best[1],best[2]
    region=np.zeros_like(mask);region[sy,sx]=True
    while True:
        d=region.copy()
        d[1:,:]|=region[:-1,:];d[:-1,:]|=region[1:,:];d[:,1:]|=region[:,:-1];d[:,:-1]|=region[:,1:]
        d&=mask
        if d.sum()==region.sum():break
        region=d
    return region

def metric(est,nord):
    w=local_window(est,nord)
    if w.shape!=(2*HALF,2*HALF) or np.isnan(w).mean()>0.1:return None
    w=np.nan_to_num(w,nan=np.nanmedian(w))
    slrm=w-boxblur(w,int(200/CS))  # relief pe fundal larg
    thr=slrm.mean()+1.0*slrm.std();mask=slrm>thr
    cy=cx=HALF
    region=cc_from_seed(mask,cy,cx)
    if region is None or region.sum()<20:return (1.0,0.0,0)  # fără structură elevată = nu e liniar
    ys,xs=np.nonzero(region)
    mx,my=xs.mean(),ys.mean();mxx=((xs-mx)**2).mean();myy=((ys-my)**2).mean();mxy=((xs-mx)*(ys-my)).mean()
    tr=mxx+myy;dd=tr*tr/4-(mxx*myy-mxy*mxy);s=math.sqrt(max(0,dd))
    l1=tr/2+s;l2=tr/2-s
    ratio=math.sqrt(l1/max(l2,1e-6))
    # lungime axă majoră în metri = întinderea proiecției pe vectorul propriu principal
    th=0.5*math.atan2(2*mxy,mxx-myy);ux,uy=math.cos(th),math.sin(th)
    proj=(xs-mx)*ux+(ys-my)*uy;major_m=(proj.max()-proj.min())*CS
    return (ratio,major_m,int(region.sum()))

rows=list(csv.DictReader(open(sys.argv[1] if len(sys.argv)>1 else '/tmp/lin_coords.csv')))
out=open('/tmp/lin_out.csv','w');wr=csv.writer(out);wr.writerow(['label','idx','lon','lat','lin_ratio','major_m','px'])
print(f"{'label':5} {'idx':>3} {'lin_ratio':>9} {'major_m':>8} {'px':>6}")
for r in rows:
    e,n=trans((float(r['lon']),float(r['lat'])),"EPSG:4326","EPSG:3844")
    m=metric(e,n)
    if m is None:
        wr.writerow([r['label'],r['idx'],r['lon'],r['lat'],'NA','NA','NA']);print(f"{r['label']:5} {r['idx']:>3}   (NA - in afara cache)");continue
    ratio,major,px=m
    wr.writerow([r['label'],r['idx'],r['lon'],r['lat'],f"{ratio:.2f}",f"{major:.0f}",px])
    print(f"{r['label']:5} {r['idx']:>3} {ratio:>9.2f} {major:>8.0f} {px:>6}")
out.close();print("-> /tmp/lin_out.csv")
