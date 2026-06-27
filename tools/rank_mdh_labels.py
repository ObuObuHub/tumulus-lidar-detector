#!/usr/bin/env python3
# rank_mdh_labels.py MODEL — auditează etichetele 'mound' Arad (labels.csv): pt fiecare, scor model (centrat,
# fereastră ±30m) + LINIARITATE (coerență direcțională pe stampa centrată = cât de „șanț" arată). Rankează
# descrescător după liniaritate (cele mai probabil ȘANȚURI greșit etichetate întâi) -> /tmp/mdh_label_audit.csv.
# Apoi mound_context.py pe CSV -> board vizual pt confirmare cull. Coords ORIGINALE (pt cull din labels.csv).
import sys,os,math,subprocess,csv
import numpy as np
from PIL import Image,ImageFilter
import torch,torch.nn as nn
H=os.path.dirname(os.path.dirname(os.path.abspath(__file__)));dev=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
MODEL=sys.argv[1] if len(sys.argv)>1 else f'{H}/combined_pre_linear.pt'
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
def coherence(st):
    a=np.asarray(Image.fromarray(st).filter(ImageFilter.GaussianBlur(1.0)),np.float32)
    gy,gx=np.gradient(a);Jxx=(gx*gx).mean();Jyy=(gy*gy).mean();Jxy=(gx*gy).mean()
    return math.sqrt((Jxx-Jyy)**2+4*Jxy*Jxy)/(Jxx+Jyy+1e-9)
class Net(nn.Module):
    def __init__(s):
        super().__init__();s.c=nn.Sequential(nn.Conv2d(1,16,3,2,1),nn.ReLU(),nn.Conv2d(16,32,3,2,1),nn.ReLU(),nn.Conv2d(32,64,3,2,1),nn.ReLU(),nn.AdaptiveAvgPool2d(1));s.f=nn.Linear(64,1)
    def forward(s,x): return s.f(s.c(x).flatten(1)).squeeze(1)
net=Net().to(dev);net.load_state_dict(torch.load(MODEL,map_location=dev,weights_only=True));net.eval()
OFF=[(dlo,dla) for dlo in np.linspace(-30,30,9) for dla in np.linspace(-30,30,9)]
rows=[r for r in csv.DictReader(open(f'{H}/labeled/labels.csv')) if r.get('verdict')=='mound']
mounds=[r for r in rows if LOMIN<=float(r['lon'])<=LOMAX and LAMIN<=float(r['lat'])<=LAMAX]
out=[]
for i,r in enumerate(mounds):
    lo,la=float(r['lon']),float(r['lat']);stamps=[]
    for dlo,dla in OFF:
        la2=la+dla/111000;lo2=lo+dlo/(111000*math.cos(math.radians(la)))
        st=stamp(lo2,la2)
        if st is not None: stamps.append(st)
    if not stamps: continue
    X=torch.tensor(np.array(stamps),dtype=torch.uint8).unsqueeze(1).float().to(dev)/255.
    with torch.no_grad(): sc=torch.sigmoid(net(X)).cpu().numpy()
    k=int(sc.argmax());coh=coherence(stamps[k])
    out.append((i,r.get('id',str(i)),lo,la,float(sc.max()),coh))
out.sort(key=lambda x:-x[5])  # cele mai liniare (șanț) întâi
with open('/tmp/mdh_label_audit.csv','w') as f:
    w=csv.writer(f);w.writerow(['idx','labels_id','lon','lat','scor_v0','coh_liniaritate'])
    for i,(li,lid,lo,la,sv,coh) in enumerate(out,1): w.writerow([i,lid,f"{lo:.6f}",f"{la:.6f}",f"{sv:.3f}",f"{coh:.3f}"])
print(f"{len(out)} etichete Arad rankate dupa liniaritate -> /tmp/mdh_label_audit.csv")
print("Top 12 cele mai 'sant' (coh mare):")
for i,(li,lid,lo,la,sv,coh) in enumerate(out[:12],1): print(f"  #{i} id={lid} coh={coh:.2f} v0={sv:.2f} @{lo:.4f},{la:.4f}")
print(f"Mediana coh={np.median([o[5] for o in out]):.2f}; coh>0.5 (probabil sant): {sum(1 for o in out if o[5]>0.5)}/{len(out)}")
