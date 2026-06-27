#!/usr/bin/env python3
# hillshade_dtm.py CLON CLAT KM [DISP] — randează hillshade CURAT (OARB, fără scoruri) pt o zonă LAKI3/Oltenia 0.5m,
# ca Andrei să încercuiască tumulii pe el (validare oarbă / marcare GT). Salvează geo-transform în /tmp/hs_dtm_geo.json
# ca să pot mapa cercurile lui înapoi la lon/lat. -> review/hillshade_dtm.png
import os,sys,math,subprocess,json
import numpy as np
from PIL import Image,ImageFilter
H=os.path.expanduser('~/lidar-match');CACHE="/tmp/laki3";CS=0.5;TPX=2000
CLON=float(sys.argv[1]);CLAT=float(sys.argv[2]);KM=float(sys.argv[3]) if len(sys.argv)>3 else 1.5
DISP=int(sys.argv[4]) if len(sys.argv)>4 else 2000
APP="/Applications/QGIS-final-4_0_3.app/Contents"
ENV=dict(os.environ,DYLD_FRAMEWORK_PATH=f"{APP}/Frameworks",PROJ_DATA=f"{APP}/Resources/qgis/proj",PROJ_LIB=f"{APP}/Resources/qgis/proj",GDAL_DATA=f"{APP}/Resources/qgis/gdal")
GT=f"{APP}/MacOS/gdaltransform"
def trans(p,s,t):
    r=subprocess.run([GT,"-s_srs",s,"-t_srs",t],input=f"{p[0]} {p[1]}\n",capture_output=True,text=True,env=ENV);a,b,_=r.stdout.split();return float(a),float(b)
def dl(nk,ek):
    p=f"{CACHE}/{nk}_{ek}.npy";return np.load(p) if os.path.exists(p) else None
def hs(dem,cs,azs=(315,45,135,225,270,0),alt=35):
    gy,gx=np.gradient(dem,cs);sl=np.arctan(np.hypot(gx,gy));asp=np.arctan2(-gy,gx);o=np.zeros_like(dem);ar=math.radians(alt)
    for az in azs: azr=math.radians(360-az+90);o+=np.clip(np.sin(ar)*np.cos(sl)+np.cos(ar)*np.sin(sl)*np.cos(azr-asp),0,1)
    return o/len(azs)
est,nord=trans((CLON,CLAT),"EPSG:4326","EPSG:3844");half=KM*1000/2
e0=int((est-half)//1000);e1=int((est+half)//1000);n0=int((nord-half)//1000);n1=int((nord+half)//1000)
xll0=e0*1000;ytop0=(n1+1)*1000;W=(e1-e0+1)*TPX;Hh=(n1-n0+1)*TPX
mos=np.full((Hh,W),np.nan,np.float32);ntiles=0
for nk in range(n0,n1+1):
    for ek in range(e0,e1+1):
        d=dl(nk,ek)
        if d is None: continue
        ntiles+=1;ox=int((ek*1000-xll0)/CS);oy=int((ytop0-(nk+1)*1000)/CS);mos[oy:oy+TPX,ox:ox+TPX]=d[:TPX,:TPX]
if ntiles==0: print(f"EROARE: niciun tile laki3 pt {CLON},{CLAT}");sys.exit(1)
# crop strâns la KM box (centrat pe punct) ca movilele să fie vizibile
pxc=int((est-xll0)/CS);pyc=int((ytop0-nord)/CS);hbox=int(KM*1000/CS/2)
x0=max(0,pxc-hbox);y0=max(0,pyc-hbox);x1=min(W,pxc+hbox);y1=min(Hh,pyc+hbox)
crop=mos[y0:y1,x0:x1];cH,cW=crop.shape
bg=hs(np.nan_to_num(crop,nan=float(np.nanmin(crop))),CS)
lo,hi=np.percentile(bg,2),np.percentile(bg,98);bg=(np.clip((bg-lo)/(hi-lo+1e-6),0,1)*255).astype('uint8')
sc=min(1.0,DISP/max(cW,cH));dW,dH=int(cW*sc),int(cH*sc)
Image.fromarray(bg).resize((dW,dH)).save(f'{H}/review/hillshade_dtm.png')
geo={"east_left":xll0+x0*CS,"north_top":ytop0-y0*CS,"CS":CS,"crop_W":cW,"crop_H":cH,"disp_W":dW,"disp_H":dH,"clon":CLON,"clat":CLAT,"km":KM}
json.dump(geo,open('/tmp/hs_dtm_geo.json','w'))
print(f"-> review/hillshade_dtm.png ({dW}x{dH}, {KM}km @0.5m, {ntiles} tile) + /tmp/hs_dtm_geo.json")
