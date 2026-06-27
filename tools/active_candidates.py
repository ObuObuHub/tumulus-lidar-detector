#!/usr/bin/env python3
# active_candidates.py [N=6] — scoreaza negativele cu modelul, FILTREAZA la CAMPIE (exclude munte/deal
# rugos, unde nu-s tumuli), randeaza top-N candidati de campie cu coordonate pt validare Andrei.
import os,glob,csv,math,subprocess,json,sys
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import torch,torch.nn as nn
H=os.path.expanduser('~/lidar-match');dev=torch.device('mps');N=int(sys.argv[1]) if len(sys.argv)>1 else 6
R=6378137.0;C=2*math.pi*R;ORIG=-20037508.342787;ORIGY=20037508.342787;Z=17;MPP=C/(256*2**Z)
ORGm="Q2Kmg0bQDn3rySgn";TM="/tmp/mdh_tiles";CACHE="/tmp/laki3";CS=0.5;TPX=2000
APP="/Applications/QGIS-final-4_0_3.app/Contents"
ENV=dict(os.environ,DYLD_FRAMEWORK_PATH=f"{APP}/Frameworks",PROJ_DATA=f"{APP}/Resources/qgis/proj",PROJ_LIB=f"{APP}/Resources/qgis/proj",GDAL_DATA=f"{APP}/Resources/qgis/gdal")
GT=f"{APP}/MacOS/gdaltransform"
def st2ll(e,n):
    r=subprocess.run([GT,"-s_srs","EPSG:3844","-t_srs","EPSG:4326"],input=f"{e} {n}\n",capture_output=True,text=True,env=ENV);a,b,_=r.stdout.split();return float(a),float(b)
def ll2st(lo,la):
    r=subprocess.run([GT,"-s_srs","EPSG:4326","-t_srs","EPSG:3844"],input=f"{lo} {la}\n",capture_output=True,text=True,env=ENV);a,b,_=r.stdout.split();return float(a),float(b)
MDHs=[("AR_MDH_tif",(20.67,45.86,22.77,46.70)),("BH_MDH_tif",(21.37,46.36,22.83,47.61)),("HD_MDH_tif",(22.32,45.23,23.60,46.37)),("AB_MDH_tif",(22.66,45.44,23.82,46.59))]
def pickm(lo,la):
    for s,(a,b,c,d) in MDHs:
        if a<=lo<=c and b<=la<=d: return s
    return None
def mtile(svc,col,row):
    fn=f"{TM}/{svc}_{col}_{row}.png"
    if not os.path.exists(fn): subprocess.run(["curl","-s","--max-time","20","-o",fn,f"https://tiles.arcgis.com/tiles/{ORGm}/arcgis/rest/services/{svc}/MapServer/tile/{Z}/{row}/{col}"],check=False)
    try: return Image.open(fn).convert('L')
    except: return None
def mdh_win(lo,la,M=360):
    svc=pickm(lo,la)
    if not svc: return None
    half=M/2/MPP;x=R*math.radians(lo);y=R*math.log(math.tan(math.pi/4+math.radians(la)/2));px=(x-ORIG)/MPP;py=(ORIGY-y)/MPP;x0=px-half;y0=py-half;W=int(2*half)
    cv=Image.new('L',(W,W),0)
    for col in range(int(x0//256),int((x0+W)//256)+1):
        for row in range(int(y0//256),int((y0+W)//256)+1):
            t=mtile(svc,col,row)
            if t: cv.paste(t,(col*256-int(x0),row*256-int(y0)))
    return np.asarray(cv,float)
def dl(nk,ek):
    p=f"{CACHE}/{nk}_{ek}.npy";return np.load(p) if os.path.exists(p) else None
def dtm_win(est,nord,M=360):
    half=M/2;e0=int((est-half)//1000);e1=int((est+half)//1000);n0=int((nord-half)//1000);n1=int((nord+half)//1000)
    xll0=e0*1000;ytop0=(n1+1)*1000;cv=np.full(((n1-n0+1)*TPX,(e1-e0+1)*TPX),np.nan,np.float32)
    for nk in range(n0,n1+1):
        for ek in range(e0,e1+1):
            d=dl(nk,ek)
            if d is None:continue
            ox=int((ek*1000-xll0)/CS);oy=int((ytop0-(nk+1)*1000)/CS);cv[oy:oy+TPX,ox:ox+TPX]=d[:TPX,:TPX]
    col=(est-xll0)/CS;row=(ytop0-nord)/CS;hp=int(half/CS)
    return cv[int(row-hp):int(row+hp),int(col-hp):int(col+hp)]
def hs(dem,cs,azs=(315,45,135,225,270,0),alt=35):
    gy,gx=np.gradient(dem,cs);sl=np.arctan(np.hypot(gx,gy));asp=np.arctan2(-gy,gx);o=np.zeros_like(dem);ar=math.radians(alt)
    for az in azs:
        azr=math.radians(360-az+90);o+=np.clip(np.sin(ar)*np.cos(sl)+np.cos(ar)*np.sin(sl)*np.cos(azr-asp),0,1)
    return o/len(azs)
# rugozitate: DTM=amplitudine elevatie (m); MDH=std hillshade brut. Praguri campie.
def is_plain(src,lo,la,est=None,nord=None):
    if src=='DTM':
        w=dtm_win(est,nord)
        if w is None or w.size==0 or np.isnan(w).mean()>0.5: return False,999
        v=w[~np.isnan(w)]; rng=np.percentile(v,98)-np.percentile(v,2)
        return rng<18, rng  # campie/terasa < 18 m amplitudine pe 360 m
    else:
        w=mdh_win(lo,la)
        if w is None or w.std()<0.3: return False,999
        return w.std()<14, w.std()  # MDH brut: campie spalacita std mic; munte std mare
# model
class Net(nn.Module):
    def __init__(s):
        super().__init__();s.c=nn.Sequential(nn.Conv2d(1,16,3,2,1),nn.ReLU(),nn.Conv2d(16,32,3,2,1),nn.ReLU(),nn.Conv2d(32,64,3,2,1),nn.ReLU(),nn.AdaptiveAvgPool2d(1));s.f=nn.Linear(64,1)
    def forward(s,x): return s.f(s.c(x).flatten(1)).squeeze(1)
net=Net().to(dev);net.load_state_dict(torch.load(f'{H}/combined_cnn.pt'));net.eval()
def load(f): return np.asarray(Image.open(f).convert('L').resize((128,128)),np.float32)/255.
dtm={};mdh={}
for r in csv.DictReader(open(f'{H}/dataset_neg/manifest.csv')): dtm[os.path.basename(r['file'])]=(float(r['est']),float(r['nord']))
for r in csv.DictReader(open(f'{H}/dataset_neg_mdh/manifest.csv')): mdh[os.path.basename(r['file'])]=(float(r['lon']),float(r['lat']))
files=glob.glob(f'{H}/dataset_neg/*.png')+glob.glob(f'{H}/dataset_neg_mdh/*.png')
sc=[]
with torch.no_grad():
    for i in range(0,len(files),256):
        b=files[i:i+256];xb=torch.tensor(np.array([load(f) for f in b])).unsqueeze(1).float().to(dev)
        s=torch.sigmoid(net(xb)).cpu().numpy();sc+=list(zip(s.tolist(),b))
sc.sort(reverse=True)
plain=[];skip_mt=0
for s,f in sc:
    if len(plain)>=N: break
    bn=os.path.basename(f)
    if bn in dtm: e,n=dtm[bn]; lo,la=st2ll(e,n); ok,rg=is_plain('DTM',lo,la,e,n); src='DTM'
    elif bn in mdh: lo,la=mdh[bn]; ok,rg=is_plain('MDH',lo,la); src='MDH'; e=n=None
    else: continue
    if s<0.5: break
    if not ok: skip_mt+=1; continue
    plain.append((float(s),bn,src,float(lo),float(la),float(rg)))
    print(f"  CAMPIE {s:.3f} {bn} [{src}] {lo:.4f},{la:.4f} rug={rg:.1f}")
print(f"sarit {skip_mt} candidati muntosi; {len(plain)} de campie")
json.dump(plain,open('/tmp/cand_plain.json','w'))
# montaj context
def ctx(src,lo,la,out=240):
    if src=='MDH':
        w=mdh_win(lo,la,360)
        if w is None: return None
        l2,h2=np.percentile(w,2),np.percentile(w,98);a=np.clip((w-l2)/(h2-l2+1e-6)*255,0,255)
    else:
        e,n=ll2st(lo,la);w=dtm_win(e,n,360)
        if w is None or w.size==0: return None
        w=np.where(np.isnan(w),np.nanmean(w),w).astype(np.float32);h=hs(w,CS);l2,h2=np.percentile(h,2),np.percentile(h,98);a=np.clip((h-l2)/(h2-l2+1e-6)*255,0,255)
    return Image.fromarray(a.astype('uint8')).resize((out,out))
cell=250;sh=Image.new('RGB',(3*cell,2*cell+24),(20,20,20));dr=ImageDraw.Draw(sh)
try: fnt=ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf",15)
except: fnt=ImageFont.load_default()
dr.text((4,3),"CANDIDATI DE CAMPIE (cruce=centru). Incercuieste DOAR tumulii reali.",fill=(0,255,0),font=fnt)
for i,(s,bn,src,lo,la,rg) in enumerate(plain):
    im=ctx(src,lo,la);x=(i%3)*cell;y=(i//3)*cell+22
    if im:
        im=im.convert('RGB');d2=ImageDraw.Draw(im);cc=im.size[0]//2
        d2.line([(cc-8,cc),(cc+8,cc)],fill=(255,0,0),width=2);d2.line([(cc,cc-8),(cc,cc+8)],fill=(255,0,0),width=2)
        sh.paste(im,(x+2,y+2))
    dr.text((x+4,y+2),f"{i+1}: {s:.2f} {src} {lo:.3f},{la:.3f}",fill=(0,255,255),font=fnt)
sh.save(f'{H}/review/candidates_plain.png');print("-> review/candidates_plain.png")