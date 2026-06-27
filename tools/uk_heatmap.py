#!/usr/bin/env python3
# uk_heatmap.py CLON CLAT KM [MODEL] — vizual UK: LiDAR simplu (hillshade) + heatmap model + barrows OSM.
# EA WCS DTM 1m (EPSG:27700), OSGB pur-python. -> review/uk_lidar.jpg + review/uk_heatmap.jpg
import sys,os,math,subprocess,json
import numpy as np
from PIL import Image,ImageFilter,ImageDraw,ImageFont
import torch,torch.nn as nn
H=os.path.dirname(os.path.dirname(os.path.abspath(__file__)));dev=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
CLON=float(sys.argv[1]);CLAT=float(sys.argv[2]);KM=float(sys.argv[3]) if len(sys.argv)>3 else 2.5
MODEL=sys.argv[4] if len(sys.argv)>4 else f'{H}/combined_cnn.pt'
MPP=1.0;SCALES=[80];CID="13787b9a-26a4-4775-8523-806d13af58fc__Lidar_Composite_Elevation_DTM_1m"  # 80m = fereastra FIXĂ a setului de antrenament (insight Andrei): NU multi-scală (40/56 zoom texturi mici→fals fire)
APP=os.environ.get("QGIS_APP","/Applications/QGIS-final-4_0_3.app/Contents")
ENV=dict(os.environ,DYLD_FRAMEWORK_PATH=f"{APP}/Frameworks",PROJ_DATA=f"{APP}/Resources/qgis/proj",PROJ_LIB=f"{APP}/Resources/qgis/proj",GDAL_DATA=f"{APP}/Resources/qgis/gdal")
GTR=f"{APP}/MacOS/gdal_translate"
def osgb(lat_deg,lon_deg):
    aw=6378137.0;bw=6356752.3142;e2w=1-bw*bw/(aw*aw);lat=math.radians(lat_deg);lon=math.radians(lon_deg)
    nu=aw/math.sqrt(1-e2w*math.sin(lat)**2);x=nu*math.cos(lat)*math.cos(lon);y=nu*math.cos(lat)*math.sin(lon);z=(1-e2w)*nu*math.sin(lat)
    tx,ty,tz=-446.448,125.157,-542.060;s=-20.4894e-6
    rx=math.radians(-0.1502/3600);ry=math.radians(-0.2470/3600);rz=math.radians(-0.8421/3600)
    x2=tx+(1+s)*(x-rz*y+ry*z);y2=ty+(1+s)*(rz*x+y-rx*z);z2=tz+(1+s)*(-ry*x+rx*y+z)
    a=6377563.396;b=6356256.909;e2=1-b*b/(a*a);p=math.sqrt(x2*x2+y2*y2);lat2=math.atan2(z2,p*(1-e2))
    for _ in range(12):
        nu2=a/math.sqrt(1-e2*math.sin(lat2)**2);lat2=math.atan2(z2+e2*nu2*math.sin(lat2),p)
    lon2=math.atan2(y2,x2)
    F0=0.9996012717;lat0=math.radians(49);lon0=math.radians(-2);E0=400000;N0=-100000;n=(a-b)/(a+b)
    nu2=a*F0/math.sqrt(1-e2*math.sin(lat2)**2);rho=a*F0*(1-e2)/(1-e2*math.sin(lat2)**2)**1.5;eta2=nu2/rho-1
    M=b*F0*((1+n+1.25*n*n+1.25*n**3)*(lat2-lat0)-(3*n+3*n*n+2.625*n**3)*math.sin(lat2-lat0)*math.cos(lat2+lat0)+(1.875*n*n+1.875*n**3)*math.sin(2*(lat2-lat0))*math.cos(2*(lat2+lat0))-(35/24*n**3)*math.sin(3*(lat2-lat0))*math.cos(3*(lat2+lat0)))
    sl=math.sin(lat2);cl=math.cos(lat2);tl=math.tan(lat2)
    I=M+N0;II=nu2/2*sl*cl;III=nu2/24*sl*cl**3*(5-tl**2+9*eta2);IIIA=nu2/720*sl*cl**5*(61-58*tl**2+tl**4)
    IV=nu2*cl;Vv=nu2/6*cl**3*(nu2/rho-tl**2);VI=nu2/120*cl**5*(5-18*tl**2+tl**4+14*eta2-58*tl**2*eta2)
    dl=lon2-lon0;return E0+IV*dl+Vv*dl**3+VI*dl**5, I+II*dl**2+III*dl**4+IIIA*dl**6
cE,cN=osgb(CLAT,CLON);half=KM*1000/2
url=(f"https://environment.data.gov.uk/spatialdata/lidar-composite-digital-terrain-model-dtm-1m/wcs?service=WCS&version=2.0.1"
     f"&request=GetCoverage&coverageId={CID}&subset=E({cE-half:.0f},{cE+half:.0f})&subset=N({cN-half:.0f},{cN+half:.0f})&format=image/tiff")
print(f"fetch UK EA DTM 1m {KM}km @ {CLON},{CLAT}...",flush=True)
subprocess.run(["curl","-s","--max-time","180","-o","/tmp/ukh.tif",url],check=False)
subprocess.run([GTR,"-q","-of","AAIGrid","/tmp/ukh.tif","/tmp/ukh.asc"],env=ENV,check=False)
L=open('/tmp/ukh.asc').read().split('\n');hdr={};i=0
while i<len(L) and L[i].split() and L[i].split()[0].lower() in('ncols','nrows','xllcorner','yllcorner','cellsize','nodata_value'):
    k,v=L[i].split()[:2];hdr[k.lower()]=float(v);i+=1
nc,nr=int(hdr['ncols']),int(hdr['nrows']);xll=hdr['xllcorner'];yll=hdr['yllcorner'];ce=hdr['cellsize'];ytop=yll+nr*ce
DEM=np.fromstring(' '.join(L[i:]),sep=' ',dtype=np.float32)[:nc*nr].reshape(nr,nc);DEM[DEM<-1000]=np.nan;DEM[DEM>1e30]=np.nan;DEM=np.nan_to_num(DEM,nan=float(np.nanmedian(DEM)))
def hs(d,cs,azs=(315,45,135,225,270,0),alt=35):
    gy,gx=np.gradient(d,cs);sl=np.arctan(np.hypot(gx,gy));asp=np.arctan2(-gy,gx);o=np.zeros_like(d);ar=math.radians(alt)
    for az in azs: azr=math.radians(360-az+90);o+=np.clip(np.sin(ar)*np.cos(sl)+np.cos(ar)*np.sin(sl)*np.cos(azr-asp),0,1)
    return o/len(azs)
sh=hs(DEM,ce);lo,hi=np.percentile(sh,2),np.percentile(sh,98);A=np.clip((sh-lo)/(hi-lo+1e-9)*255,0,255).astype(np.uint8);Hh,Ww=A.shape
def homog(a):
    a2=np.asarray(Image.fromarray(a).filter(ImageFilter.GaussianBlur(0.8)),np.uint8);cdf=np.bincount(a2.ravel(),minlength=256).astype(np.float64).cumsum()
    return (cdf[a2]/cdf[-1]*255).astype(np.uint8) if cdf[-1]>0 else a2
class Net(nn.Module):
    def __init__(s):
        super().__init__();s.c=nn.Sequential(nn.Conv2d(1,16,3,2,1),nn.ReLU(),nn.Conv2d(16,32,3,2,1),nn.ReLU(),nn.Conv2d(32,64,3,2,1),nn.ReLU(),nn.AdaptiveAvgPool2d(1));s.f=nn.Linear(64,1)
    def forward(s,x): return s.f(s.c(x).flatten(1)).squeeze(1)
net=Net().to(dev);net.load_state_dict(torch.load(MODEL,map_location=dev,weights_only=True));net.eval()
# dense scan
step=int(15/MPP);acc=np.zeros((Hh,Ww),np.float32);batch=[];pos=[]
for M in SCALES:
    win=int(M/MPP);h=win//2
    for py in range(h,Hh-h,step):
        for px in range(h,Ww-h,step):
            w=A[py-h:py+h,px-h:px+h]
            if w.shape!=(2*h,2*h) or w.std()<3: continue
            # MATCH EXACT ANTRENAMENT: crop 80m -> resize 128 -> homog(blur0.8+histeq). FĂRĂ content/2m, FĂRĂ percentile.
            batch.append(homog(np.asarray(Image.fromarray(w).resize((128,128)),np.uint8)));pos.append((px,py))
            if len(batch)>=1024:
                xb=torch.tensor(np.array(batch),dtype=torch.float32).unsqueeze(1).to(dev)/255.
                with torch.no_grad(): sc=torch.sigmoid(net(xb)).cpu().numpy()
                for (qx,qy),s in zip(pos,sc): acc[qy,qx]=max(acc[qy,qx],float(s))
                batch=[];pos=[]
if batch:
    xb=torch.tensor(np.array(batch),dtype=torch.float32).unsqueeze(1).to(dev)/255.
    with torch.no_grad(): sc=torch.sigmoid(net(xb)).cpu().numpy()
    for (qx,qy),s in zip(pos,sc): acc[qy,qx]=max(acc[qy,qx],float(s))
scored=acc[acc>0]
print(f"DENSE SCAN model real: {len(scored)} ferestre scorate | min {scored.min():.3f} median {np.median(scored):.3f} max {scored.max():.3f} | %>=0.3 {100*(scored>=0.3).mean():.1f}% %>=0.5 {100*(scored>=0.5).mean():.1f}% %>=0.7 {100*(scored>=0.7).mean():.1f}%",flush=True)
# barrows OSM in tile
bar=json.load(open('/tmp/uk_barrows.json'))['elements'];bpx=[]
for e in bar:
    la,lo=(e['lat'],e['lon']) if e['type']=='node' else (e.get('center',{}).get('lat'),e.get('center',{}).get('lon'))
    if la is None: continue
    E,N=osgb(la,lo);px=int((E-xll)/ce);py=int((ytop-N)/ce)
    if 0<=px<Ww and 0<=py<Hh: bpx.append((px,py))
# downscale display
DW=1500;fac=max(1,Ww//DW);dw=Ww//fac;dh=Hh//fac
base=np.array(Image.fromarray(A).resize((dw,dh))).astype(np.float32);baseRGB=np.stack([base]*3,-1)
# 1) plain LiDAR
im1=Image.fromarray(baseRGB.astype(np.uint8));d1=ImageDraw.Draw(im1)
try: ft=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial Bold.ttf',18)
except: ft=ImageFont.load_default()
for px,py in bpx: x,y=px//fac,py//fac;d1.ellipse([x-9,y-9,x+9,y+9],outline=(0,255,0),width=2)
d1.rectangle([0,0,dw,30],fill=(12,12,14));d1.text((6,5),f"UK Wessex — LiDAR 1m (hillshade). Verde = {len(bpx)} round barrows OSM.",fill=(255,255,255),font=ft)
im1.convert('RGB').save(f'{H}/review/uk_lidar.jpg',quality=85)
# 2) heatmap — JET pe câmpul de scor REAL (max-pool pe grilă apoi upsample), normalizat la [0, vmax]
gh,gw=max(1,Hh//step),max(1,Ww//step);G=np.zeros((gh,gw),np.float32)
for j in range(gh):
    for i in range(gw):
        blk=acc[j*step:(j+1)*step,i*step:(i+1)*step]
        if blk.size: G[j,i]=blk.max()
field=np.asarray(Image.fromarray((np.clip(G,0,1)*255).astype(np.uint8)).resize((dw,dh),Image.BILINEAR),np.float32)/255.
vmax=max(0.5,float(np.percentile(scored,99)))  # normalizează ca să se VADĂ firing-ul real (chiar slab)
v=np.clip(field/vmax,0,1)
def jet(t):
    r=np.clip(1.5-np.abs(4*t-3),0,1);g=np.clip(1.5-np.abs(4*t-2),0,1);b=np.clip(1.5-np.abs(4*t-1),0,1);return np.stack([r,g,b],-1)
hm=(jet(v)*255).astype(np.float32);al=np.clip(v*1.1,0,1)[...,None]
disp=(baseRGB*(1-al*0.7)+hm*al*0.7).astype(np.uint8)
im2=Image.fromarray(disp);d2=ImageDraw.Draw(im2)
for px,py in bpx: x,y=px//fac,py//fac;d2.ellipse([x-10,y-10,x+10,y+10],outline=(255,255,255),width=2)
d2.rectangle([0,0,dw,46],fill=(12,12,14))
d2.text((6,4),f"UK Wessex — HEATMAP REAL model (jet, normalizat la {vmax:.2f}). Cerc alb = barrow OSM.",fill=(255,255,255),font=ft)
d2.text((6,25),f"scan fereastra fixa 80m (= set antrenament). scor: median {np.median(scored):.2f} max {scored.max():.2f}, %>=0.7 {100*(scored>=0.7).mean():.1f}%. Cerc=barrow OSM.",fill=(200,200,200),font=ImageFont.load_default())
im2.convert('RGB').save(f'{H}/review/uk_heatmap.jpg',quality=85)
print(f"-> review/uk_lidar.jpg + uk_heatmap.jpg ({dw}x{dh}, {len(bpx)} barrows, vmax {vmax:.2f})")
