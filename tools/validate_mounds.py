#!/usr/bin/env python3
# validate_mounds.py MODEL_A MODEL_B — scorează movilele REALE Arad (labels.csv) cu 2 modele și face montaj
# de VALIDARE MANUALĂ: stampe hillshade numerotate + scor_A / scor_B, grupate (ambele văd / A vede B nu / ambele rateaza).
# Întrebare pt Andrei: movilele pe care B (v2) le suprimă — sunt CHIAR movile, sau erau slabe/dubioase (deci B corect)?
import sys,os,math,subprocess,csv
import numpy as np
from PIL import Image,ImageFilter,ImageDraw,ImageFont
import torch,torch.nn as nn
H=os.path.dirname(os.path.dirname(os.path.abspath(__file__)));dev=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
MA=sys.argv[1];MB=sys.argv[2]
LOMIN,LOMAX,LAMIN,LAMAX=20.67,22.77,45.86,46.70
R=6378137.0;ORIG=-20037508.342787;ORIGY=20037508.342787;Z=18;MPP=2*math.pi*R/(256*2**Z)
MDH=[("AR_MDH_tif",(20.67,45.86,22.77,46.70)),("BH_MDH_tif",(21.37,46.36,22.83,47.61)),("HD_MDH_tif",(22.32,45.23,23.60,46.37)),("AB_MDH_tif",(22.66,45.44,23.82,46.59))]
ORG="Q2Kmg0bQDn3rySgn";TDIR="/tmp/mdh_tiles_L18";os.makedirs(TDIR,exist_ok=True)
def pick(lo,la):
    for svc,(a,b,c,d) in MDH:
        if a<=lo<=c and b<=la<=d: return svc
def tile(svc,col,row):
    fn=f"{TDIR}/{svc}_{col}_{row}.png"
    if not os.path.exists(fn): subprocess.run(["curl","-s","--max-time","25","-o",fn,f"https://tiles.arcgis.com/tiles/{ORG}/arcgis/rest/services/{svc}/MapServer/tile/{Z}/{row}/{col}"],check=False)
    try: return Image.open(fn).convert('L')
    except: return None
def homog(a):
    a2=np.asarray(Image.fromarray(a).filter(ImageFilter.GaussianBlur(0.8)),np.uint8);cdf=np.bincount(a2.ravel(),minlength=256).astype(np.float64).cumsum()
    return (cdf[a2]/cdf[-1]*255).astype(np.uint8) if cdf[-1]>0 else a2
def stamp(lo,la,meters=80,eff=2.0,out=128):
    svc=pick(lo,la)
    if not svc: return None
    half=meters/2/MPP;x=R*math.radians(lo);y=R*math.log(math.tan(math.pi/4+math.radians(la)/2))
    px=(x-ORIG)/MPP;py=(ORIGY-y)/MPP;x0=px-half;y0=py-half;W=int(2*half);cv=Image.new('L',(W,W),0);ok=False
    for col in range(int(x0//256),int((x0+W)//256)+1):
        for row in range(int(y0//256),int((y0+W)//256)+1):
            t=tile(svc,col,row)
            if t: cv.paste(t,(col*256-int(x0),row*256-int(y0)));ok=True
    if not ok: return None
    a=np.asarray(cv,np.float32)
    if a.std()<0.5: return None
    lo2,hi2=np.percentile(a,2),np.percentile(a,98);a=np.clip((a-lo2)/(hi2-lo2+1e-6),0,1)
    f=max(1,int(round(eff/MPP)));Hh,Ww=a.shape;a=a[:Hh//f*f,:Ww//f*f].reshape(Hh//f,f,Ww//f,f).mean((1,3))
    return homog(np.asarray(Image.fromarray((a*255).astype('uint8')).resize((out,out)),np.uint8))
class Net(nn.Module):
    def __init__(s):
        super().__init__();s.c=nn.Sequential(nn.Conv2d(1,16,3,2,1),nn.ReLU(),nn.Conv2d(16,32,3,2,1),nn.ReLU(),nn.Conv2d(32,64,3,2,1),nn.ReLU(),nn.AdaptiveAvgPool2d(1));s.f=nn.Linear(64,1)
    def forward(s,x): return s.f(s.c(x).flatten(1)).squeeze(1)
def load(m):
    n=Net().to(dev);n.load_state_dict(torch.load(m,map_location=dev,weights_only=True));n.eval();return n
na,nb=load(MA),load(MB)
def sc(net,st):
    with torch.no_grad(): return float(torch.sigmoid(net(torch.tensor(st).unsqueeze(0).unsqueeze(0).float().to(dev)/255.)).item())
rows=[r for r in csv.DictReader(open(f'{H}/labeled/labels.csv')) if r.get('verdict')=='mound']
pts=[(float(r['lon']),float(r['lat'])) for r in rows if LOMIN<=float(r['lon'])<=LOMAX and LAMIN<=float(r['lat'])<=LAMAX]
# CENTRARE (cerut Andrei): scor = MAX pe fereastră toleranță ±30m (protocol detecție), nu pixel exact.
# Stampa afișată = centrată pe vârful găsit de v0 (modelul permisiv găsește domul). Scor v0,v2 la fereastra fiecăruia.
OFF=[(dlo,dla) for dlo in np.linspace(-30,30,9) for dla in np.linspace(-30,30,9)]  # ±30m, 9x9
def scores_window(net,lo,la,stamps):
    X=torch.tensor(np.array(stamps),dtype=torch.uint8).unsqueeze(1).float().to(dev)/255.
    with torch.no_grad(): return torch.sigmoid(net(X)).cpu().numpy()
items=[]
for lo,la in pts:
    stamps=[];offs=[]
    for dlo,dla in OFF:
        la2=la+dla/111000;lo2=lo+dlo/(111000*math.cos(math.radians(la)))
        st=stamp(lo2,la2)
        if st is not None: stamps.append(st);offs.append((lo2,la2))
    if not stamps: continue
    va=scores_window(na,lo,la,stamps);vb=scores_window(nb,lo,la,stamps)
    ka=int(va.argmax())                       # vârf v0 = centrul afișat
    items.append((float(va.max()),float(vb.max()),stamps[ka],offs[ka][0],offs[ka][1]))
supp=[it for it in items if it[0]>=0.6 and it[1]<0.6]      # A vede, B suprimă = recall contestat
supp.sort(key=lambda x:-(x[0]-x[1]))                       # cele mai mari căderi întâi
print(f"total {len(items)} movile | SUPRIMATE de B (A>=.6, B<.6): {len(supp)} | ambele>=.6: {sum(1 for i in items if i[0]>=.6 and i[1]>=.6)} | ambele<.6: {sum(1 for i in items if i[0]<.6 and i[1]<.6)}")
# montaj suprimate (max 48)
show=supp[:48];C=4;rw=(len(show)+C-1)//C;cell=216;hh=30
img=Image.new('RGB',(C*cell,hh+rw*cell),(15,15,15));dr=ImageDraw.Draw(img)
try: ft=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial Bold.ttf',17)
except: ft=ImageFont.load_default()
dr.text((4,6),f"VALIDARE: movile etichetate pe care v2 le SUPRIMĂ (v0>=.6, v2<.6). Sunt CHIAR movile? Eticheta #idx v0|v2",fill=(255,230,90),font=ft)
mp=open('/tmp/validate_supp_map.csv','w');mw=csv.writer(mp);mw.writerow(['idx','lon','lat','scor_v0','scor_v2'])
for k,(sa,sb,st,lo,la) in enumerate(show,1):
    x=((k-1)%C)*cell;y=hh+((k-1)//C)*cell
    img.paste(Image.fromarray(st).convert('RGB').resize((cell-4,cell-4)),(x+2,y+16))
    dr.text((x+3,y+1),f"#{k} {sa:.2f}|{sb:.2f}",fill=(120,255,120) if sa>=0.6 else (255,120,120),font=ft)
    mw.writerow([k,f"{lo:.5f}",f"{la:.5f}",f"{sa:.3f}",f"{sb:.3f}"])
mp.close();img.save(f"{H}/review/validate_suppressed.png");print(f"-> review/validate_suppressed.png ({len(show)}) + /tmp/validate_supp_map.csv")
