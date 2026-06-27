#!/usr/bin/env python3
# sweep_mdh.py CLON CLAT KM [MODEL] — sweep pe zonă MDH (Arad/Bihor/Hunedoara/Alba) prin pipeline-ul MDH.
# Grid -> stamp MDH (80m@2m autocontrast) -> homog (ca la train) -> scor model -> board numerotat scor-ascuns.
import sys,os,math,subprocess,csv
import numpy as np
from PIL import Image,ImageFilter,ImageDraw,ImageFont
import torch,torch.nn as nn
H=os.path.dirname(os.path.dirname(os.path.abspath(__file__)));dev=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
CLON=float(sys.argv[1]);CLAT=float(sys.argv[2]);KM=float(sys.argv[3]) if len(sys.argv)>3 else 4.0
MODEL=sys.argv[4] if len(sys.argv)>4 else f'{H}/combined_cleanpool.pt'
R=6378137.0;C=2*math.pi*R;ORIG=-20037508.342787;ORIGY=20037508.342787;Z=18;MPP=C/(256*2**Z)  # L18=0.6m/px
MDH=[("AR_MDH_tif",(20.67,45.86,22.77,46.70)),("BH_MDH_tif",(21.37,46.36,22.83,47.61)),
     ("HD_MDH_tif",(22.32,45.23,23.60,46.37)),("AB_MDH_tif",(22.66,45.44,23.82,46.59))]
ORG="Q2Kmg0bQDn3rySgn";TDIR="/tmp/mdh_tiles_L18";os.makedirs(TDIR,exist_ok=True)
def pick(lon,lat):
    for svc,(a,b,c,d) in MDH:
        if a<=lon<=c and b<=lat<=d: return svc
    return None
_tc={}
def tile(svc,col,row):
    k=(svc,col,row)
    if k in _tc: return _tc[k]
    fn=f"{TDIR}/{svc}_{col}_{row}.png"
    if not os.path.exists(fn):
        subprocess.run(["curl","-s","--max-time","25","-o",fn,f"https://tiles.arcgis.com/tiles/{ORG}/arcgis/rest/services/{svc}/MapServer/tile/{Z}/{row}/{col}"],check=False)
    try: im=Image.open(fn).convert('L')
    except: im=None
    _tc[k]=im;return im
def stamp(lon,lat,meters=80,eff=2.0,out=128):
    svc=pick(lon,lat)
    if not svc: return None
    half=meters/2/MPP;x=R*math.radians(lon);y=R*math.log(math.tan(math.pi/4+math.radians(lat)/2))
    px=(x-ORIG)/MPP;py=(ORIGY-y)/MPP;x0=px-half;y0=py-half;W=int(2*half)
    cv=Image.new('L',(W,W),0);ok=False
    for col in range(int(x0//256),int((x0+W)//256)+1):
        for row in range(int(y0//256),int((y0+W)//256)+1):
            t=tile(svc,col,row)
            if t: cv.paste(t,(col*256-int(x0),row*256-int(y0)));ok=True
    if not ok: return None
    a=np.asarray(cv,dtype=np.float32)
    if a.std()<0.5: return None
    lo,hi=np.percentile(a,2),np.percentile(a,98);a=np.clip((a-lo)/(hi-lo+1e-6),0,1)
    f=max(1,int(round(eff/MPP)));Hh,Ww=a.shape;a=a[:Hh//f*f,:Ww//f*f].reshape(Hh//f,f,Ww//f,f).mean((1,3))
    return np.asarray(Image.fromarray((a*255).astype('uint8')).resize((out,out)),np.uint8)
def homog(a):
    a2=np.asarray(Image.fromarray(a).filter(ImageFilter.GaussianBlur(0.8)),np.uint8);cdf=np.bincount(a2.ravel(),minlength=256).astype(np.float64).cumsum()
    return (cdf[a2]/cdf[-1]*255).astype(np.uint8) if cdf[-1]>0 else a2
class Net(nn.Module):
    def __init__(s):
        super().__init__();s.c=nn.Sequential(nn.Conv2d(1,16,3,2,1),nn.ReLU(),nn.Conv2d(16,32,3,2,1),nn.ReLU(),nn.Conv2d(32,64,3,2,1),nn.ReLU(),nn.AdaptiveAvgPool2d(1));s.f=nn.Linear(64,1)
    def forward(s,x): return s.f(s.c(x).flatten(1)).squeeze(1)
net=Net().to(dev);net.load_state_dict(torch.load(MODEL,map_location=dev,weights_only=True));net.eval()
print(f"sweep MDH {pick(CLON,CLAT)} la {CLAT},{CLON} ({KM}km), model {os.path.basename(MODEL)}",flush=True)
step_m=60;dlat=step_m/111000;dlon=step_m/(111000*math.cos(math.radians(CLAT)))
half=KM*1000/2;nlat=int(half/111000/dlat);nlon=int(half/(111000*math.cos(math.radians(CLAT)))/dlon)
raws=[];raw_st=[];coords=[]
for i in range(-nlat,nlat+1):
    for j in range(-nlon,nlon+1):
        la=CLAT+i*dlat;lo=CLON+j*dlon;st=stamp(lo,la)
        if st is None: continue
        raws.append(homog(st));raw_st.append(st);coords.append((lo,la))
print(f"  {len(raws)} celule cu acoperire MDH",flush=True)
if not raws: print("  ⚠ ZERO acoperire MDH la coordonatele astea");sys.exit()
X=torch.tensor(np.array(raws,dtype=np.uint8))
sc=[]
with torch.no_grad():
    for k in range(0,len(X),512): sc.extend(torch.sigmoid(net(X[k:k+512].unsqueeze(1).float().to(dev)/255.)).cpu().numpy().tolist())
sc=np.array(sc)
order=np.argsort(-sc)
NBOARD=int(sys.argv[5]) if len(sys.argv)>5 else 54
kept=[]
for k in order:
    if sc[k]<0.5: break
    lo,la=coords[k]
    if any((lo-q[0])**2*math.cos(math.radians(la))**2+(la-q[1])**2 < (150/111000)**2 for q in kept): continue
    kept.append((lo,la,float(sc[k]),raw_st[k]))
    if len(kept)>=NBOARD: break
print(f"  {int((sc>=0.7).sum())} celule >=0.7 | {int((sc>=0.9).sum())} >=0.9 | {len(kept)} detecții dedup top",flush=True)
COLS=9;cell=156;PER=COLS*6;nb=(len(kept)+PER-1)//PER
try: ft=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial Bold.ttf',16)
except: ft=ImageFont.load_default()
for b in range(nb):
    chunk=kept[b*PER:(b+1)*PER];rows=(len(chunk)+COLS-1)//COLS
    img=Image.new('RGB',(COLS*cell,rows*cell+26),(15,15,15));dr=ImageDraw.Draw(img)
    dr.text((6,4),f"Arad/MDH board {b+1}/{nb} (rang {b*PER+1}-{b*PER+len(chunk)} din {len(kept)}). VERDE=tumul.",fill=(255,255,80),font=ft)
    mp=open(f'/tmp/mdh_board_map_{b+1}.csv','w');mw=csv.writer(mp);mw.writerow(['idx','lon','lat','score'])
    for j,(lo,la,s,st) in enumerate(chunk):
        x=(j%COLS)*cell;y=(j//COLS)*cell+26
        img.paste(Image.fromarray(st).convert('RGB').resize((150,150)),(x+3,y+20));dr.text((x+4,y+2),f"#{j+1}",fill=(120,230,255),font=ft)
        mw.writerow([j+1,f"{lo:.5f}",f"{la:.5f}",f"{s:.3f}"])
    mp.close();img.save(f'{H}/review/board_mdh_{b+1}.png');print(f"  -> review/board_mdh_{b+1}.png ({len(chunk)})")
mp.close();img.save(f'{H}/review/board_mdh.png')
print(f"  -> review/board_mdh.png + /tmp/mdh_board_map.csv")
