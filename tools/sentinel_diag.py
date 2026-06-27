#!/usr/bin/env python3
# sentinel_diag.py — #3 multimodal: testează dacă movilele reale au o ANOMALIE multispectrală (NDVI/sol) față de
# împrejurimi, pe care FP-urile (mușuroaie naturale) NU o au. Sentinel-2 L2A (10m) via Element84 STAC + COG /vsicurl.
import os,json,subprocess,math,csv
import numpy as np
H=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP=os.environ.get("QGIS_APP","/Applications/QGIS-final-4_0_3.app/Contents")
ENV=dict(os.environ,DYLD_FRAMEWORK_PATH=f"{APP}/Frameworks",PROJ_DATA=f"{APP}/Resources/qgis/proj",PROJ_LIB=f"{APP}/Resources/qgis/proj",GDAL_DATA=f"{APP}/Resources/qgis/gdal",
         CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",GDAL_HTTP_MAX_RETRY="3",GDAL_HTTP_RETRY_DELAY="1",VSI_CACHE="TRUE")
GT=f"{APP}/MacOS/gdaltransform";GTR=f"{APP}/MacOS/gdal_translate"
def trans(pts,s,t):
    inp="\n".join(f"{a} {b}" for a,b in pts)+"\n";r=subprocess.run([GT,"-s_srs",s,"-t_srs",t],input=inp,capture_output=True,text=True,env=ENV)
    return [tuple(map(float,l.split()[:2])) for l in r.stdout.strip().split("\n") if l.split()]
# detecții Catane gate-passing (v3) + istp
rows=list(csv.DictReader(open('/tmp/dets_v3_gated.csv')))
s1=np.array([float(r['score']) for r in rows]);istp=np.array([int(r['istp']) for r in rows]);pg=np.array([float(r['pgate']) if r['pgate']!='NA' else 1.0 for r in rows])
thr=s1[(istp==1)&(pg>=0.70)].min();keep=(s1>=thr)&(pg>=0.70)
sub=[(float(rows[i]['lon']),float(rows[i]['lat']),int(istp[i])) for i in range(len(rows)) if keep[i]]
lons=[lo for lo,la,t in sub];lats=[la for lo,la,t in sub]
b=[min(lons)-0.02,min(lats)-0.02,max(lons)+0.02,max(lats)+0.02]
# STAC search — MULTITEMPORAL: mai multe scene cu cloud mic pe sezon (2022-2024)
q={"collections":["sentinel-2-l2a"],"bbox":b,"datetime":"2022-04-01T00:00:00Z/2024-09-30T00:00:00Z","query":{"eo:cloud_cover":{"lt":5}},"limit":40,"sortby":[{"field":"properties.eo:cloud_cover","direction":"asc"}]}
r=subprocess.run(["curl","-s","--max-time","60","-X","POST","https://earth-search.aws.element84.com/v1/search","-H","Content-Type: application/json","-d",json.dumps(q)],capture_output=True,text=True)
allf=json.loads(r.stdout)['features']
# alege ~8 scene cu date DISTINCTE (zile diferite)
scenes=[];seen=set()
for f in allf:
    day=f['properties']['datetime'][:10]
    if day in seen: continue
    seen.add(day);scenes.append(f)
    if len(scenes)>=8: break
epsg=scenes[0]['properties']['proj:epsg']
print(f"{len(scenes)} scene multitemporale, EPSG {epsg}: "+", ".join(f['properties']['datetime'][:10] for f in scenes),flush=True)
cor=trans([(b[0],b[1]),(b[2],b[3])],"EPSG:4326",f"EPSG:{epsg}")
ulx=min(cor[0][0],cor[1][0]);lrx=max(cor[0][0],cor[1][0]);uly=max(cor[0][1],cor[1][1]);lry=min(cor[0][1],cor[1][1])
def clip(url,out):
    if os.path.exists(out):os.remove(out)
    subprocess.run([GTR,"-q","-projwin",str(ulx),str(uly),str(lrx),str(lry),f"/vsicurl/{url}",out],env=ENV,check=False)
    return os.path.exists(out) and os.path.getsize(out)>1000
def loadtif(p):
    a=f"{p}.asc"
    if os.path.exists(a):os.remove(a)
    subprocess.run([GTR,"-q","-of","AAIGrid",p,a],env=ENV,check=False);L=open(a).read().split('\n');hdr={};i=0
    while i<len(L) and L[i].split() and L[i].split()[0].lower() in('ncols','nrows','xllcorner','yllcorner','cellsize','nodata_value'):
        k,v=L[i].split()[:2];hdr[k.lower()]=float(v);i+=1
    nc,nr=int(hdr['ncols']),int(hdr['nrows']);dem=np.fromstring(' '.join(L[i:]),sep=' ',dtype=np.float32)[:nc*nr].reshape(nr,nc)
    return dem,hdr['xllcorner'],hdr['yllcorner']+nr*hdr['cellsize'],hdr['cellsize']
en=trans([(lo,la) for lo,la,t in sub],"EPSG:4326",f"EPSG:{epsg}")
def sample(arr,xll,ytop,ce,e,n,rad_px):
    Hh,Ww=arr.shape;px=int((e-xll)/ce);py=int((ytop-n)/ce)
    if not(rad_px<=px<Ww-rad_px and rad_px<=py<Hh-rad_px):return np.nan,np.nan
    cen=arr[py,px];ring=arr[py-rad_px:py+rad_px+1,px-rad_px:px+rad_px+1].copy();ring[rad_px,rad_px]=np.nan
    return float(cen),float(np.nanmean(ring))
# acumulează |ΔNDVI| pe scene
acc={i:[] for i in range(len(sub))}
for sc in scenes:
    if not(clip(sc['assets']['red']['href'],'/tmp/s2_red.tif') and clip(sc['assets']['nir']['href'],'/tmp/s2_nir.tif')): continue
    R,xll,ytop,ce=loadtif('/tmp/s2_red.tif');N,_,_,_=loadtif('/tmp/s2_nir.tif')
    if R.shape!=N.shape: continue
    nd=(N-R)/(N+R+1e-6)
    for i,(e,n) in enumerate(en):
        c,rg=sample(nd,xll,ytop,ce,e,n,3)
        if not np.isnan(c): acc[i].append(abs(c-rg))
    print(f"  scenă {sc['properties']['datetime'][:10]} procesată",flush=True)
feats=[]
for i,(lo,la,t) in enumerate(sub):
    a=acc[i];feats.append((t, np.mean(a) if a else np.nan, np.max(a) if a else np.nan, len(a)))
F=np.array([f[1:3] for f in feats]);Y=np.array([f[0] for f in feats])
names=['|ΔNDVI| MEDIE multitemporal','|ΔNDVI| MAX']
def auc(tp,fp):
    tp=tp[~np.isnan(tp)];fp=fp[~np.isnan(fp)]
    if not len(tp) or not len(fp):return 0.5
    allv=np.concatenate([tp,fp]);o=allv.argsort();ranks=np.empty(len(allv));ranks[o]=np.arange(1,len(allv)+1)
    return (ranks[:len(tp)].sum()-len(tp)*(len(tp)+1)/2)/(len(tp)*len(fp))
print(f"\nmovile {(Y==1).sum()} vs FP {(Y==0).sum()} | anomalie multispectrală movilă-vs-împrejurimi (~30m inel):")
print(f"{'feature':16}{'movile':>10}{'FP':>10}{'AUC':>8}")
for j,nm in enumerate(names):
    print(f"  {nm:14}{np.nanmedian(F[Y==1,j]):>10.3f}{np.nanmedian(F[Y==0,j]):>10.3f}{auc(F[Y==1,j],F[Y==0,j]):>8.2f}")
