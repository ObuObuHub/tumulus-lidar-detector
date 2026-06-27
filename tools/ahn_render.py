#!/usr/bin/env python3
# ahn_render.py CLON CLAT KM TAG — fetch AHN4 DTM 0.5m (PDOK WCS) + randează hillshade 6-dir la
# REZOLUȚIE NATIVĂ 0.5m (levierul de generalizare pt relief jos olandez) + SLRM. NU scorează — doar
# produce imaginea crisp ca din baza noastră, pe care apoi o scanezi cu heatmap_image.py.
# -> /tmp/ahn_<TAG>_hs.png (hillshade nativ, full-res), /tmp/ahn_<TAG>_hs_disp.png (afișare), /tmp/ahn_<TAG>_slrm.png
import sys,os,math,subprocess
import numpy as np
from PIL import Image
CLON=float(sys.argv[1]);CLAT=float(sys.argv[2]);KM=float(sys.argv[3]) if len(sys.argv)>3 else 1.0
TAG=sys.argv[4] if len(sys.argv)>4 else 'site'
CS=0.5
APP=os.environ.get("QGIS_APP","/Applications/QGIS-final-4_0_3.app/Contents")
ENV=dict(os.environ,DYLD_FRAMEWORK_PATH=f"{APP}/Frameworks",PROJ_DATA=f"{APP}/Resources/qgis/proj",PROJ_LIB=f"{APP}/Resources/qgis/proj",GDAL_DATA=f"{APP}/Resources/qgis/gdal")
GT=f"{APP}/MacOS/gdaltransform";GTR=f"{APP}/MacOS/gdal_translate"
def trans(lon,lat,s="EPSG:4326",t="EPSG:28992"):
    r=subprocess.run([GT,"-s_srs",s,"-t_srs",t],input=f"{lon} {lat}\n",capture_output=True,text=True,env=ENV);a,b,_=r.stdout.split();return float(a),float(b)
est,nord=trans(CLON,CLAT);half=KM*1000/2
x0,x1=est-half,est+half;y0,y1=nord-half,nord+half
url=(f"https://service.pdok.nl/rws/ahn/wcs/v1_0?service=WCS&version=2.0.1&request=GetCoverage"
     f"&coverageId=dtm_05m&subset=x({x0:.1f},{x1:.1f})&subset=y({y0:.1f},{y1:.1f})&format=image/tiff")
tif=f"/tmp/ahn_{TAG}.tif";asc=f"/tmp/ahn_{TAG}.asc"
print(f"fetch AHN dtm_05m {KM}km @ {CLON},{CLAT} (EPSG28992 {est:.0f},{nord:.0f})...",flush=True)
subprocess.run(["curl","-s","--max-time","180","-o",tif,url],check=False)
sz=os.path.getsize(tif) if os.path.exists(tif) else 0
if sz<10000: sys.exit(f"EROARE fetch AHN ({sz}B): {open(tif).read()[:200] if sz else 'gol'}")
if os.path.exists(asc): os.remove(asc)
subprocess.run([GTR,"-of","AAIGrid",tif,asc],capture_output=True,env=ENV)
L=open(asc).read().split('\n');hdr={};i=0
while i<len(L) and L[i].split() and L[i].split()[0].lower() in('ncols','nrows','xllcorner','yllcorner','cellsize','nodata_value'):
    k,v=L[i].split()[:2];hdr[k.lower()]=float(v);i+=1
nc,nr=int(hdr['ncols']),int(hdr['nrows']);ce=hdr['cellsize']
dem=np.fromstring(' '.join(L[i:]),sep=' ',dtype=np.float32)[:nc*nr].reshape(nr,nc)
dem[dem>1e30]=np.nan;dem[dem==hdr.get('nodata_value',-9999)]=np.nan
nanfrac=np.isnan(dem).mean()
print(f"DTM {nc}x{nr} @ {ce}m | elev {np.nanmin(dem):.1f}-{np.nanmax(dem):.1f}m | NaN {nanfrac*100:.0f}%",flush=True)
dem=np.nan_to_num(dem,nan=float(np.nanmedian(dem)))
def hs(d,cs,azs=(315,45,135,225,270,0),alt=35):  # 6-dir alt 35, IDENTIC cu baza noastră
    gy,gx=np.gradient(d,cs);sl=np.arctan(np.hypot(gx,gy));asp=np.arctan2(-gy,gx);o=np.zeros_like(d);ar=math.radians(alt)
    for az in azs: azr=math.radians(360-az+90);o+=np.clip(np.sin(ar)*np.cos(sl)+np.cos(ar)*np.sin(sl)*np.cos(azr-asp),0,1)
    return o/len(azs)
# hillshade la REZOLUȚIE NATIVĂ 0.5m (NU pre-downsample) — păstrează movilele joase
sh=hs(dem,ce);lo,hi=np.percentile(sh,2),np.percentile(sh,98)
img=np.clip((sh-lo)/(hi-lo+1e-9)*255,0,255).astype(np.uint8)
Image.fromarray(img).save(f"/tmp/ahn_{TAG}_hs.png")
disp=Image.fromarray(img);
if nc>1200: disp=disp.resize((1200,int(1200*nr/nc)))
disp.save(f"/tmp/ahn_{TAG}_hs_disp.png")
# SLRM (relief local) pt control vizual
def boxblur(a,r):
    k=2*r+1;c=np.cumsum(np.cumsum(np.pad(a,((1,0),(1,0))),0),1)
    s=c[k:,k:]-c[:-k,k:]-c[k:,:-k]+c[:-k,:-k];out=np.full_like(a,np.nan)
    out[r:r+s.shape[0],r:r+s.shape[1]]=s/(k*k);return np.nan_to_num(out,nan=np.nanmean(a))
slrm=dem-boxblur(dem,int(round(15/ce)))  # fereastră ~15m
sl=np.clip(slrm,-1.0,1.0)
from PIL import Image as I2
# colormap simplu albastru-verde-roșu pt SLRM
t=(sl+1)/2;r=np.clip(1.5-abs(4*t-3),0,1);g=np.clip(1.5-abs(4*t-2),0,1);b=np.clip(1.5-abs(4*t-1),0,1)
slrgb=(np.stack([r,g,b],-1)*255).astype(np.uint8)
sd=Image.fromarray(slrgb)
if nc>1200: sd=sd.resize((1200,int(1200*nr/nc)))
sd.save(f"/tmp/ahn_{TAG}_slrm.png")
print(f"-> /tmp/ahn_{TAG}_hs.png (nativ {nc}px) + _hs_disp.png + _slrm.png | NaN {nanfrac*100:.0f}%",flush=True)
