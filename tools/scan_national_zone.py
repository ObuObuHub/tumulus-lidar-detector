#!/usr/bin/env python3
# scan_national_zone.py CLON CLAT KM [MODEL] [STEP_M] [THR] — SURVEY pe stratul NAȚIONAL ROLiDAR
# (acoperă tot RO, incl. Moldova/Iași unde NU există 0.5m). ⚠ pe Moldova detaliu real ~5m =
# OUT-OF-DISTRIBUTION pt model (antrenat 0.5-0.6m) -> aprinde difuz. Scopul aici: scoatem
# candidații DISCREȚI (hotspot compact izolat prin NMS), nu zgomotul de fond.
# Output:
#   review/movileni_heatmap.jpg  (hillshade național + heatmap jet + candidați numerotați)
#   review/movileni_board.jpg    (crop-uri ale candidaților păstrați, numerotate + scor)
#   /tmp/movileni_cand.csv        (idx,lon,lat,score)
import os,sys,math,subprocess,csv
import numpy as np
from PIL import Image,ImageFilter,ImageDraw,ImageFont
import torch,torch.nn as nn
H=os.path.dirname(os.path.dirname(os.path.abspath(__file__)));dev=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
CLON=float(sys.argv[1]);CLAT=float(sys.argv[2]);KM=float(sys.argv[3]) if len(sys.argv)>3 else 6.0
MODEL=sys.argv[4] if len(sys.argv)>4 else f'{H}/combined_cnn.pt'
STEP_M=float(sys.argv[5]) if len(sys.argv)>5 else 12.0
THR=float(sys.argv[6]) if len(sys.argv)>6 else 0.90
WIN_M=40.0
TAG=os.environ.get('TAG','movileni')
R=6378137.0;ORIG=-20037508.342787;ORIGY=20037508.342787;Z=16;MPP=2*math.pi*R/(256*2**Z)
MPPg=MPP*math.cos(math.radians(CLAT))
ORG="wCvLzGFkz06gCfBg";svc=os.environ.get('NATSVC','1m');TDIR="/tmp/nat_tiles";os.makedirs(TDIR,exist_ok=True)
def merc(lo,la): return R*math.radians(lo), R*math.log(math.tan(math.pi/4+math.radians(la)/2))
def ll_to_px(lo,la,x0,y0):
    x,y=merc(lo,la);return (x-ORIG)/MPP-x0,(ORIGY-y)/MPP-y0
def px_to_ll(px,py,x0,y0):
    x=(px+x0)*MPP+ORIG;y=ORIGY-(py+y0)*MPP
    return math.degrees(x/R), math.degrees(2*math.atan(math.exp(y/R))-math.pi/2)
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
cov=(mosA>0).mean()*100
print(f"mozaic {W}x{W}px ({KM}km, ~{MPPg:.2f}m/px sol), {nt} tile-uri, acoperire {cov:.0f}%, model {os.path.basename(MODEL)}",flush=True)
def homog(a):
    a2=np.asarray(Image.fromarray(a).filter(ImageFilter.GaussianBlur(0.8)),np.uint8);cdf=np.bincount(a2.ravel(),minlength=256).astype(np.float64).cumsum()
    return (cdf[a2]/cdf[-1]*255).astype(np.uint8) if cdf[-1]>0 else a2
CLEAN=os.environ.get('CLEAN','0')=='1'   # normalizare GLOBALĂ pe regiune (câmp plat rămâne plat), fără over-stretch/homog
_valid=mosA[mosA>0];GLO,GHI=(np.percentile(_valid,1),np.percentile(_valid,99)) if _valid.size else (0,255)
if CLEAN: print(f"  CLEAN: normalizare globală {GLO:.0f}-{GHI:.0f} (fără per-window stretch/homog)",flush=True)
def stamp_px(px,py,meters=WIN_M,eff=2.0,out=128):
    hw=int(meters/2/MPPg);w=mosA[py-hw:py+hw,px-hw:px+hw]
    if w.shape!=(2*hw,2*hw): return None
    if CLEAN:
        if (w>0).mean()<0.5: return None
        a=np.clip((w-GLO)/(GHI-GLO+1e-6),0,1)
        return np.asarray(Image.fromarray((a*255).astype('uint8')).resize((out,out)),np.uint8)
    if w.std()<0.3: return None
    lo,hi=np.percentile(w,2),np.percentile(w,98);a=np.clip((w-lo)/(hi-lo+1e-6),0,1)
    f=max(1,int(round(eff/MPPg)));Hh,Ww=a.shape;a=a[:Hh//f*f,:Ww//f*f].reshape(Hh//f,f,Ww//f,f).mean((1,3))
    return homog(np.asarray(Image.fromarray((a*255).astype('uint8')).resize((out,out)),np.uint8))
class Net(nn.Module):
    def __init__(s):
        super().__init__();s.c=nn.Sequential(nn.Conv2d(1,16,3,2,1),nn.ReLU(),nn.Conv2d(16,32,3,2,1),nn.ReLU(),nn.Conv2d(32,64,3,2,1),nn.ReLU(),nn.AdaptiveAvgPool2d(1));s.f=nn.Linear(64,1)
    def forward(s,x): return s.f(s.c(x).flatten(1)).squeeze(1)
net=Net().to(dev);net.load_state_dict(torch.load(MODEL,map_location=dev,weights_only=True));net.eval()
hw=int(max(WIN_M,80)/2/MPPg)+2;step=max(1,int(STEP_M/MPPg))
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
# ---- NMS: candidați discreți ----
MIN_SEP_M=120.0;sep_px=MIN_SEP_M/MPPg
cells=[(grid[iy,ix],gys[iy],gxs[ix]) for iy in range(len(gys)) for ix in range(len(gxs)) if not np.isnan(grid[iy,ix]) and grid[iy,ix]>=THR]
cells.sort(reverse=True)
cand=[]
for v,py,px in cells:
    if all((px-cx2)**2+(py-cy2)**2 >= sep_px**2 for _,py2,px2,cy2,cx2 in [(c[0],c[1],c[2],c[1],c[2]) for c in cand]):
        cand.append((v,py,px))
    if len(cand)>=40: break
print(f"  candidați discreți (>= {THR}, NMS {MIN_SEP_M:.0f}m): {len(cand)}",flush=True)
# ---- heatmap ----
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
try: fnt=ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf",26)
except: fnt=ImageFont.load_default()
for i,(v,py,px) in enumerate(cand,1):
    dr.ellipse([px-24,py-24,px+24,py+24],outline=(255,255,255),width=4)
    dr.text((px+26,py-16),str(i),fill=(255,255,0),font=fnt)
os.makedirs(f"{H}/review",exist_ok=True)
# resize heatmap for Telegram (<1MB)
disp=img;mx=1600
if W>mx: disp=img.resize((mx,int(mx*W/W)))
disp.save(f"{H}/review/{TAG}_heatmap.jpg",quality=85)
print(f"-> review/{TAG}_heatmap.jpg")
# ---- board crops ----
with open(f"/tmp/{TAG}_cand.csv","w",newline="") as f:
    wr=csv.writer(f);wr.writerow(["idx","lon","lat","score"])
    crops=[]
    for i,(v,py,px) in enumerate(cand,1):
        lo,la=px_to_ll(px,py,x0,y0);wr.writerow([i,f"{lo:.6f}",f"{la:.6f}",f"{v:.3f}"])
        hwc=int(80/2/MPPg);w=mosA[py-hwc:py+hwc,px-hwc:px+hwc]
        if w.shape!=(2*hwc,2*hwc): crops.append((i,v,None,lo,la));continue
        lo2,hi2=np.percentile(w,2),np.percentile(w,98);a=np.clip((w-lo2)/(hi2-lo2+1e-6),0,1)
        crops.append((i,v,(a*255).astype('uint8'),lo,la))
# tile board
if crops:
    CS_=160;cols=6;rows=(len(crops)+cols-1)//cols
    bd=Image.new('RGB',(cols*CS_,rows*CS_+24),(20,20,20));bdr=ImageDraw.Draw(bd)
    try: f2=ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf",15)
    except: f2=ImageFont.load_default()
    bdr.text((6,4),f"{TAG} — {len(crops)} candidati discreti (strat national ~5m, OOD)",fill=(255,255,0),font=f2)
    for k,(i,v,arr,lo,la) in enumerate(crops):
        cx2=(k%cols)*CS_;cy2=(k//cols)*CS_+24
        if arr is not None:
            im=Image.fromarray(arr).resize((CS_-8,CS_-28)).convert('RGB');bd.paste(im,(cx2+4,cy2+4))
        bdr.text((cx2+6,cy2+CS_-22),f"#{i} {v:.2f}",fill=(0,255,120),font=f2)
    bd.save(f"{H}/review/{TAG}_board.jpg",quality=88)
    print(f"-> review/{TAG}_board.jpg")
print(f"-> /tmp/{TAG}_cand.csv")
