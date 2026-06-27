#!/usr/bin/env python3
# heatmap_national.py CLON CLAT KM [MODEL] [STEP_M] — HEATMAP OARBĂ pe stratul NAȚIONAL ROLiDAR „1m"
# (acoperă tot RO, incl. Moldova/Botoșani). ⚠ pe Moldova = agregare din 5m -> detaliu real ~5m =
# OUT-OF-DISTRIBUTION pt modelul antrenat pe 0.5-0.6m. Scor dens via pipeline-ul tile-uri randate
# (fereastra 80m -> stretch percentile -> homog -> 128), FĂRĂ seeding la ground-truth.
# Suprapune marcaje RAN (din arg env RAN_PTS = "lon,lat,eticheta;...") pt validare oarbă.
# -> review/heatmap_national.png + scor la fiecare punct RAN + statistici.
import os,sys,math,subprocess
import numpy as np
from PIL import Image,ImageFilter,ImageDraw,ImageFont
import torch,torch.nn as nn
H=os.path.expanduser('~/lidar-match');dev=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
CLON=float(sys.argv[1]);CLAT=float(sys.argv[2]);KM=float(sys.argv[3]) if len(sys.argv)>3 else 2.0
MODEL=sys.argv[4] if len(sys.argv)>4 else f'{H}/combined_cnn.pt'
STEP_M=float(sys.argv[5]) if len(sys.argv)>5 else 10.0
WIN_M=float(sys.argv[6]) if len(sys.argv)>6 else 40.0  # fereastra în METRI TEREN (scara corectă pt rezoluția dată)
R=6378137.0;ORIG=-20037508.342787;ORIGY=20037508.342787;Z=16;MPP=2*math.pi*R/(256*2**Z)  # ~2.39 merc m/px
MPPg=MPP*math.cos(math.radians(CLAT))  # m/px REAL la sol (corecție cos lat — fără ea fereastra e prea mică la N)
ORG="wCvLzGFkz06gCfBg";svc="1m";TDIR="/tmp/nat_tiles";os.makedirs(TDIR,exist_ok=True)
def merc(lo,la): return R*math.radians(lo), R*math.log(math.tan(math.pi/4+math.radians(la)/2))
def ll_to_px(lo,la,x0,y0):
    x,y=merc(lo,la);return (x-ORIG)/MPP-x0,(ORIGY-y)/MPP-y0
def tilepx(col,row):
    fn=f"{TDIR}/{svc}_{Z}_{col}_{row}.png"
    if not os.path.exists(fn): subprocess.run(["curl","-s","--max-time","25","-o",fn,f"https://tiles.arcgis.com/tiles/{ORG}/arcgis/rest/services/{svc}/MapServer/tile/{Z}/{row}/{col}"],check=False)
    try: return Image.open(fn).convert('L')
    except: return None
cx,cy=merc(CLON,CLAT);half=KM*1000/2/MPP
x0=(cx-ORIG)/MPP-half;y0=(ORIGY-cy)/MPP-half;W=int(2*half)
mos=Image.new('L',(W,W),0);nt=0
for col in range(int(x0//256),int((x0+W)//256)+1):
    for row in range(int(y0//256),int((y0+W)//256)+1):
        t=tilepx(col,row)
        if t: mos.paste(t,(col*256-int(x0),row*256-int(y0)));nt+=1
mosA=np.asarray(mos,np.float32)
print(f"mozaic {W}x{W}px ({KM}km, ~{MPP:.2f}m/px merc), {nt} tile-uri, model {os.path.basename(MODEL)}",flush=True)
def homog(a):
    a2=np.asarray(Image.fromarray(a).filter(ImageFilter.GaussianBlur(0.8)),np.uint8);cdf=np.bincount(a2.ravel(),minlength=256).astype(np.float64).cumsum()
    return (cdf[a2]/cdf[-1]*255).astype(np.uint8) if cdf[-1]>0 else a2
def stamp_px(px,py,meters=WIN_M,eff=2.0,out=128):
    hw=int(meters/2/MPPg);w=mosA[py-hw:py+hw,px-hw:px+hw]
    if w.shape!=(2*hw,2*hw) or w.std()<0.3: return None
    lo,hi=np.percentile(w,2),np.percentile(w,98);a=np.clip((w-lo)/(hi-lo+1e-6),0,1)
    f=max(1,int(round(eff/MPPg)));Hh,Ww=a.shape;a=a[:Hh//f*f,:Ww//f*f].reshape(Hh//f,f,Ww//f,f).mean((1,3))
    return homog(np.asarray(Image.fromarray((a*255).astype('uint8')).resize((out,out)),np.uint8))
class Net(nn.Module):
    def __init__(s):
        super().__init__();s.c=nn.Sequential(nn.Conv2d(1,16,3,2,1),nn.ReLU(),nn.Conv2d(16,32,3,2,1),nn.ReLU(),nn.Conv2d(32,64,3,2,1),nn.ReLU(),nn.AdaptiveAvgPool2d(1));s.f=nn.Linear(64,1)
    def forward(s,x): return s.f(s.c(x).flatten(1)).squeeze(1)
net=Net().to(dev);net.load_state_dict(torch.load(MODEL,map_location=dev,weights_only=True));net.eval()
hw=int(max(WIN_M,80)/2/MPPg)+2;step=max(1,int(STEP_M/MPPg))
print(f"  fereastra {WIN_M:.0f}m teren ({int(WIN_M/MPPg)}px @ {MPPg:.2f}m/px)",flush=True)
gxs=list(range(hw,W-hw,step));gys=list(range(hw,W-hw,step))
grid=np.full((len(gys),len(gxs)),np.nan,np.float32);batch=[];pos=[]
for iy,py in enumerate(gys):
    for ix,px in enumerate(gxs):
        s=stamp_px(px,py)
        if s is not None: batch.append(s);pos.append((iy,ix))
X=torch.tensor(np.array(batch,dtype=np.uint8));sc=[]
with torch.no_grad():
    for k in range(0,len(X),512): sc.extend(torch.sigmoid(net(X[k:k+512].unsqueeze(1).float().to(dev)/255.)).cpu().numpy().tolist())
for (iy,ix),v in zip(pos,sc): grid[iy,ix]=v
valid=grid[~np.isnan(grid)]
print(f"  {valid.size} celule | medie {valid.mean():.3f} | mediană {np.median(valid):.3f} | %>=0.7 {(valid>=0.7).mean()*100:.1f}% | %>=0.9 {(valid>=0.9).mean()*100:.1f}%",flush=True)
# jet colormap
def jet(s):
    r=np.clip(1.5-np.abs(4*s-3),0,1);g=np.clip(1.5-np.abs(4*s-2),0,1);b=np.clip(1.5-np.abs(4*s-1),0,1)
    return np.stack([r,g,b],-1)
g=np.nan_to_num(grid,nan=0.0)
field=np.asarray(Image.fromarray((g*255).astype('uint8')).resize((W,W),Image.BICUBIC),np.float32)/255.
bg=np.clip((mosA-np.percentile(mosA,2))/(np.percentile(mosA,98)-np.percentile(mosA,2)+1e-6),0,1)
rgb=(np.stack([bg,bg,bg],-1)*255).astype(np.float32)
col=jet(field)*255;alpha=(0.25+0.55*field)[...,None]
out=(rgb*(1-alpha)+col*alpha).astype(np.uint8)
img=Image.fromarray(out);dr=ImageDraw.Draw(img)
try: fnt=ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf",22)
except: fnt=ImageFont.load_default()
# RAN markers
pts=os.environ.get('RAN_PTS','')
print(f"--- scor model la punctele RAN (fereastra {WIN_M:.0f}m teren) ---")
if pts:
    for item in pts.split(';'):
        if not item.strip(): continue
        lo,la,lab=item.split(',');lo=float(lo);la=float(la)
        px,py=ll_to_px(lo,la,x0,y0)
        # score at marker: nearest grid cell
        if 0<=px<W and 0<=py<W:
            s=stamp_px(int(px),int(py))
            sv=float(torch.sigmoid(net(torch.tensor(s[None,None],dtype=torch.float32).to(dev)/255.)).item()) if s is not None else float('nan')
        else: sv=float('nan')
        print(f"  {lab:16s} ({lo},{la}) -> scor {sv:.3f}  {'IN' if 0<=px<W and 0<=py<W else 'OUT-of-tile'}")
        if 0<=px<W and 0<=py<W:
            dr.ellipse([px-26,py-26,px+26,py+26],outline=(255,255,255),width=4)
            dr.text((px+28,py-14),lab,fill=(255,255,255),font=fnt)
# center marker (coordonata dată)
pcx,pcy=ll_to_px(CLON,CLAT,x0,y0)
dr.line([pcx-16,pcy,pcx+16,pcy],fill=(255,255,0),width=3);dr.line([pcx,pcy-16,pcx,pcy+16],fill=(255,255,0),width=3)
os.makedirs(f"{H}/review",exist_ok=True)
img.save(f"{H}/review/heatmap_national.png")
print(f"-> {H}/review/heatmap_national.png")
