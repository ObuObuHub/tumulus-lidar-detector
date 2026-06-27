#!/usr/bin/env python3
# heatmap.py CLON CLAT KM [MODEL] [STEP_M] — HARTĂ DE CĂLDURĂ OARBĂ.
# Scorează DENS toată zona (fereastră glisantă, pas fin), FĂRĂ niciun ground-truth, și randează scorul ca un
# câmp de căldură peste hillshade. Arată ONEST unde "aprinde" modelul pe TOATĂ suprafața (over-firing vizibil),
# nu doar la movilele cunoscute [[feedback-ml-blind-validation]]. Reutilizează pipeline-ul MDH L18.
# -> review/heatmap.png + statistici (medie, mediană, %>=0.7, %>=0.9) la stdout.
import sys,os,math,subprocess
import numpy as np
from PIL import Image,ImageFilter,ImageDraw,ImageFont
import torch,torch.nn as nn
H=os.path.dirname(os.path.dirname(os.path.abspath(__file__)));dev=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
CLON=float(sys.argv[1]);CLAT=float(sys.argv[2]);KM=float(sys.argv[3]) if len(sys.argv)>3 else 1.5
MODEL=sys.argv[4] if len(sys.argv)>4 else f'{H}/combined_cnn.pt'
STEP_M=float(sys.argv[5]) if len(sys.argv)>5 else 12.0   # pas grilă în metri (fin = heatmap netedă)
R=6378137.0;ORIG=-20037508.342787;ORIGY=20037508.342787;Z=18;MPP=2*math.pi*R/(256*2**Z)  # L18=0.6m/px
MDH=[("AR_MDH_tif",(20.67,45.86,22.77,46.70)),("BH_MDH_tif",(21.37,46.36,22.83,47.61)),("HD_MDH_tif",(22.32,45.23,23.60,46.37)),("AB_MDH_tif",(22.66,45.44,23.82,46.59))]
ORG="Q2Kmg0bQDn3rySgn";TDIR="/tmp/mdh_tiles_L18";os.makedirs(TDIR,exist_ok=True)
def pick(lo,la):
    for svc,(a,b,c,d) in MDH:
        if a<=lo<=c and b<=la<=d: return svc
def merc(lo,la): return R*math.radians(lo), R*math.log(math.tan(math.pi/4+math.radians(la)/2))
def tilepx(svc,col,row):
    fn=f"{TDIR}/{svc}_{col}_{row}.png"
    if not os.path.exists(fn): subprocess.run(["curl","-s","--max-time","25","-o",fn,f"https://tiles.arcgis.com/tiles/{ORG}/arcgis/rest/services/{svc}/MapServer/tile/{Z}/{row}/{col}"],check=False)
    try: return Image.open(fn).convert('L')
    except: return None
svc=pick(CLON,CLAT)
if svc is None: print(f"EROARE: {CLON},{CLAT} nu e în acoperirea MDH (AR/BH/HD/AB)");sys.exit(1)
cx,cy=merc(CLON,CLAT);half=KM*1000/2; pxc=(cx-ORIG)/MPP; pyc=(ORIGY-cy)/MPP; hp=half/MPP
x0=pxc-hp;y0=pyc-hp;W=int(2*hp)
mos=Image.new('L',(W,W),0)
for col in range(int(x0//256),int((x0+W)//256)+1):
    for row in range(int(y0//256),int((y0+W)//256)+1):
        t=tilepx(svc,col,row)
        if t: mos.paste(t,(col*256-int(x0),row*256-int(y0)))
mosA=np.asarray(mos,np.float32)
print(f"mozaic {W}x{W}px ({KM}km, ~{MPP:.2f}m/px), svc {svc}, model {os.path.basename(MODEL)}",flush=True)
def homog(a):
    a2=np.asarray(Image.fromarray(a).filter(ImageFilter.GaussianBlur(0.8)),np.uint8);cdf=np.bincount(a2.ravel(),minlength=256).astype(np.float64).cumsum()
    return (cdf[a2]/cdf[-1]*255).astype(np.uint8) if cdf[-1]>0 else a2
def stamp_px(px,py,meters=80,eff=2.0,out=128):
    hw=int(meters/2/MPP);w=mosA[py-hw:py+hw,px-hw:px+hw]
    if w.shape!=(2*hw,2*hw) or w.std()<0.5: return None
    lo,hi=np.percentile(w,2),np.percentile(w,98);a=np.clip((w-lo)/(hi-lo+1e-6),0,1)
    f=max(1,int(round(eff/MPP)));Hh,Ww=a.shape;a=a[:Hh//f*f,:Ww//f*f].reshape(Hh//f,f,Ww//f,f).mean((1,3))
    return homog(np.asarray(Image.fromarray((a*255).astype('uint8')).resize((out,out)),np.uint8))
class Net(nn.Module):
    def __init__(s):
        super().__init__();s.c=nn.Sequential(nn.Conv2d(1,16,3,2,1),nn.ReLU(),nn.Conv2d(16,32,3,2,1),nn.ReLU(),nn.Conv2d(32,64,3,2,1),nn.ReLU(),nn.AdaptiveAvgPool2d(1));s.f=nn.Linear(64,1)
    def forward(s,x): return s.f(s.c(x).flatten(1)).squeeze(1)
net=Net().to(dev);net.load_state_dict(torch.load(MODEL,map_location=dev,weights_only=True));net.eval()
step=max(1,int(STEP_M/MPP));hw=int(40/MPP)
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
print(f"  {valid.size} celule scorate | medie {valid.mean():.3f} | mediană {np.median(valid):.3f} | %>=0.7 {(valid>=0.7).mean()*100:.1f}% | %>=0.9 {(valid>=0.9).mean()*100:.1f}%",flush=True)
# câmp de scor -> rezoluția mozaicului (bicubic), NaN->0
g=np.nan_to_num(grid,nan=0.0)
field=np.asarray(Image.fromarray((g*255).astype('uint8')).resize((W,W),Image.BICUBIC),np.float32)/255.
# colormap jet (albastru->cyan->verde->galben->roșu)
s4=field*4
Rr=np.clip(s4-2,0,1);Gg=np.clip(np.minimum(s4,4-s4),0,1);Bb=np.clip(2-s4,0,1)
cmap=np.stack([Rr,Gg,Bb],-1)*255
# alpha: sub 0.3 = invizibil (negativ adevărat), creste pâna la 0.8 la scor mare -> over-firing vizibil
alpha=np.clip((field-0.3)/0.5,0,1)[...,None]*0.78
base=np.asarray(Image.fromarray(mosA.astype('uint8')).convert('RGB'),np.float32)
out=(base*(1-alpha)+cmap*alpha).astype('uint8')
ov=Image.fromarray(out)
# bară-legendă jos
dr=ImageDraw.Draw(ov)
try: ft=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial Bold.ttf',max(13,W//90))
except: ft=ImageFont.load_default()
dr.text((6,4),f"HEATMAP OARBA {os.path.basename(MODEL)} | medie {valid.mean():.2f} %>=.9 {(valid>=0.9).mean()*100:.0f}% | albastru=scor mic, rosu=aprinde",fill=(255,255,255),font=ft)
disp=ov.resize((min(W,1200),int(min(W,1200))))
out_path=f"{H}/review/heatmap.png";disp.save(out_path)
print(f"  -> {out_path}")
