#!/usr/bin/env python3
# sweep_national.py CLON CLAT KM [THRESH=0.5] [TOPN=12]
# Detector pe stratul NAȚIONAL ROLiDAR „1m agregare" (acoperă tot RO, incl. Moldova/Iași).
# Sursa tile: tiles.arcgis.com/wCvLzGFkz06gCfBg/.../1m/MapServer, nivel max 16 (~1.6 m/px la lat 47).
# ⚠ pe Moldova datele sunt agregare din 5m -> detaliu real ~5m, OUT-OF-DISTRIBUTION pt model.
# Fereastra glisantă 80m (convenție identică sweep_detector.py) -> scor -> top candidați dedup + crop-uri.
import os,sys,math,subprocess,json
import numpy as np
from PIL import Image,ImageDraw,ImageFont,ImageFilter
import torch,torch.nn as nn
H=os.path.expanduser('~/lidar-match');dev=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
CLON=float(sys.argv[1]);CLAT=float(sys.argv[2]);KM=float(sys.argv[3]) if len(sys.argv)>3 else 4.0
THRESH=float(sys.argv[4]) if len(sys.argv)>4 else 0.5
TOPN=int(sys.argv[5]) if len(sys.argv)>5 else 12
R=6378137.0;C=2*math.pi*R;ORIG=-20037508.342787;ORIGY=20037508.342787;Z=16;MPP=C/(256*2**Z)
ORG="wCvLzGFkz06gCfBg";svc="1m";TM="/tmp/nat_tiles";os.makedirs(TM,exist_ok=True)
def tile(col,row):
    fn=f"{TM}/{svc}_{Z}_{col}_{row}.png"
    if not os.path.exists(fn): subprocess.run(["curl","-s","--max-time","20","-o",fn,f"https://tiles.arcgis.com/tiles/{ORG}/arcgis/rest/services/{svc}/MapServer/tile/{Z}/{row}/{col}"],check=False)
    try: return Image.open(fn).convert('L')
    except: return None
halfm=KM*1000/2;half=halfm/MPP
x=R*math.radians(CLON);y=R*math.log(math.tan(math.pi/4+math.radians(CLAT)/2));px=(x-ORIG)/MPP;py=(ORIGY-y)/MPP
x0=px-half;y0=py-half;W=int(2*half)
cv=Image.new('L',(W,W),0);nt=0
for col in range(int(x0//256),int((x0+W)//256)+1):
    for row in range(int(y0//256),int((y0+W)//256)+1):
        t=tile(col,row)
        if t: cv.paste(t,(col*256-int(x0),row*256-int(y0)));nt+=1
print(f"mozaic {W}x{W}px ({KM}km), {nt} tile-uri",flush=True)
mos=np.asarray(cv,np.float32)
class Net(nn.Module):
    def __init__(s):
        super().__init__();s.c=nn.Sequential(nn.Conv2d(1,16,3,2,1),nn.ReLU(),nn.Conv2d(16,32,3,2,1),nn.ReLU(),nn.Conv2d(32,64,3,2,1),nn.ReLU(),nn.AdaptiveAvgPool2d(1));s.f=nn.Linear(64,1)
    def forward(s,x): return s.f(s.c(x).flatten(1)).squeeze(1)
net=Net().to(dev);net.load_state_dict(torch.load(f'{H}/combined_cnn.pt',weights_only=True));net.eval()
WIN_M=float(sys.argv[6]) if len(sys.argv)>6 else 80.0   # fereastra în metri teren ("înălțimea" de privire)
wpx=int(WIN_M/MPP);stride=max(1,int(WIN_M/4/MPP))
ys=list(range(0,W-wpx,stride));xs=list(range(0,W-wpx,stride))
print(f"fereastra {wpx}px (~{wpx*MPP*math.cos(math.radians(CLAT)):.0f}m teren), {len(ys)}x{len(xs)} poziții",flush=True)
f=max(1,int(round(2.0/MPP)))
cands=[];batch=[];meta=[]
def gxy_to_ll(gx,gy):
    lon=(ORIG+gx*MPP)/R*180/math.pi;lat=(2*math.atan(math.exp((ORIGY-gy*MPP)/R))-math.pi/2)*180/math.pi
    return lon,lat
def flush():
    global batch,meta
    if not batch: return
    xb=torch.tensor(np.array(batch)).unsqueeze(1).float().to(dev)
    with torch.no_grad(): s=torch.sigmoid(net(xb)).cpu().numpy()
    for (cx,cy,im128),sc in zip(meta,s):
        if sc>=THRESH:
            lon,lat=gxy_to_ll(cx,cy); cands.append((float(sc),lon,lat,im128))
    batch=[];meta=[]
npos=0
for yy in ys:
    for xx in xs:
        w=mos[yy:yy+wpx,xx:xx+wpx]
        if w.shape!=(wpx,wpx) or w.std()<0.3: continue
        d=w[:w.shape[0]//f*f,:w.shape[1]//f*f].reshape(w.shape[0]//f,f,w.shape[1]//f,f).mean((1,3))
        lo,hi=np.percentile(d,2),np.percentile(d,98)
        if hi-lo<1e-6: continue
        d=np.clip((d-lo)/(hi-lo),0,1)
        base=Image.fromarray((d*255).astype('uint8')).resize((128,128)).filter(ImageFilter.GaussianBlur(0.8))  # OMOGENIZARE = ca train
        a=np.asarray(base,np.uint8);cdf=np.bincount(a.ravel(),minlength=256).astype(np.float64).cumsum()
        if cdf[-1]>0: a=(cdf[a]/cdf[-1]*255).astype(np.uint8)
        im=a.astype(np.float32)/255.
        batch.append(im);meta.append((x0+xx+wpx/2,y0+yy+wpx/2,a));npos+=1
        if len(batch)>=512: flush()
flush()
cands.sort(reverse=True,key=lambda c:c[0])
# dedup spatial ~80m
kept=[]
for sc,lon,lat,im in cands:
    if any((lon-k[1])**2+((lat-k[2]))**2<(80/111000)**2 for k in kept): continue
    kept.append((sc,lon,lat,im))
n05=sum(1 for c in cands if c[0]>=0.5);n07=sum(1 for c in cands if c[0]>=0.7);n09=sum(1 for c in cands if c[0]>=0.9)
print(f"ferestre scorate {npos} | >0.5:{n05} >0.7:{n07} >0.9:{n09} | candidați dedup:{len(kept)}",flush=True)
top=kept[:TOPN]
json.dump([[s,lo,la] for s,lo,la,_ in top],open('/tmp/iasi_cands.json','w'))
for i,(s,lo,la,_) in enumerate(top): print(f"  #{i+1} scor {s:.3f} @ {lo:.5f},{la:.5f}",flush=True)
# contact sheet crop-uri (input model 128px)
if top:
    cols=4;rows=(len(top)+cols-1)//cols;cell=150;M=Image.new('RGB',(cols*cell,rows*cell),(20,20,20));dr=ImageDraw.Draw(M)
    try: ft=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial Bold.ttf',13)
    except: ft=ImageFont.load_default()
    for i,(s,lo,la,im) in enumerate(top):
        x_=(i%cols)*cell;y_=(i//cols)*cell
        M.paste(Image.fromarray(im).convert('RGB').resize((cell-4,cell-22)),(x_+2,y_+20))
        dr.text((x_+3,y_+3),f"#{i+1} s{s:.2f}",fill=(255,220,0),font=ft)
    M.save(f'{H}/review/iasi_candidates.png');print("-> review/iasi_candidates.png",flush=True)
# heatmap overlay
heat=np.zeros((len(ys),len(xs)),np.float32)  # rebuild light heat for viz from cands grid is skipped; save mosaic
Image.fromarray(np.clip((mos-np.percentile(mos,2))/(np.percentile(mos,98)-np.percentile(mos,2)+1e-6)*255,0,255).astype('uint8')).resize((700,700)).save(f'{H}/review/iasi_mosaic.png')
print("mozaic -> review/iasi_mosaic.png",flush=True)
