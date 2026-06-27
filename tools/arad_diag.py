#!/usr/bin/env python3
# arad_diag.py CLON CLAT KM [MODEL] [TH] [RATIO_T] — diagnostic over-firing pe zona MDH (Arad & co):
#  1. scanează mozaicul hillshade MDH -> detecții (NMS), EXACT stampele pe care le vede modelul
#  2. filtru de LINIARITATE pe stampă (raport axe al structurii centrale; mal/canal = alungit -> CUT)
#  3. Grad-CAM pe FIECARE detecție (unde se uită modelul: centru=movilă; margini/dungă=shortcut)
# OUT: review/arad_overlay_filtrat.png (verde=compact/KEPT, roșu=liniar/CUT) + review/arad_gradcam.png (montaj)
# NB: Aradul NU are elevație 0.5m (laki3 e doar sudul); deci liniaritatea se măsoară pe HILLSHADE (input model),
#     proxy mai slab decât SLRM-pe-elevație folosit pe Catane, dar consistent cu ce scorează rețeaua.
import sys,os,math,subprocess
import numpy as np
from PIL import Image,ImageFilter,ImageDraw,ImageFont
import torch,torch.nn as nn,torch.nn.functional as Fn
H=os.path.dirname(os.path.dirname(os.path.abspath(__file__)));dev=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
CLON=float(sys.argv[1]);CLAT=float(sys.argv[2]);KM=float(sys.argv[3]) if len(sys.argv)>3 else 1.5
MODEL=sys.argv[4] if len(sys.argv)>4 else f'{H}/combined_cnn.pt'
TH=float(sys.argv[5]) if len(sys.argv)>5 else 0.6
RATIO_T=float(sys.argv[6]) if len(sys.argv)>6 else 2.2   # peste = alungit = mal/canal -> CUT
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
svc=pick(CLON,CLAT);cx,cy=merc(CLON,CLAT)
half=KM*1000/2; pxc=(cx-ORIG)/MPP; pyc=(ORIGY-cy)/MPP; hp=half/MPP
x0=pxc-hp;y0=pyc-hp;W=int(2*hp)
mos=Image.new('L',(W,W),0)
for col in range(int(x0//256),int((x0+W)//256)+1):
    for row in range(int(y0//256),int((y0+W)//256)+1):
        t=tilepx(svc,col,row)
        if t: mos.paste(t,(col*256-int(x0),row*256-int(y0)))
mosA=np.asarray(mos,np.float32)
print(f"mozaic {W}x{W}px ({KM}km, ~{MPP:.2f}m/px), svc {svc}",flush=True)
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
# Grad-CAM hooks pe ultimul conv (c[4]=Conv32->64)
acts={};grads={}
net.c[4].register_forward_hook(lambda m,i,o: acts.__setitem__('v',o.detach()))
net.c[4].register_full_backward_hook(lambda m,gi,go: grads.__setitem__('v',go[0].detach()))
def gradcam(img128):
    x=torch.tensor(img128).unsqueeze(0).unsqueeze(0).float().to(dev)/255.;x.requires_grad_(True)
    net.zero_grad();out=net(x);out.backward()
    A=acts['v'][0];G=grads['v'][0];wts=G.mean(dim=(1,2));cam=Fn.relu((wts[:,None,None]*A).sum(0))
    cam=cam.cpu().numpy();cam=cam/(cam.max()+1e-9)
    return float(torch.sigmoid(out).item()),np.asarray(Image.fromarray((cam*255).astype('uint8')).resize((128,128)))
def lin_ratio(s128):
    # raport axe al structurii centrale pe stampa hillshade (proxy liniaritate)
    c=s128[32:96,32:96].astype(np.float32)   # central 40m
    thr=c.mean()+1.0*c.std();ys,xs=np.nonzero(c>thr)
    if len(xs)<15: return 99.0                # fără structură compactă -> resping
    cx2,cy2=xs.mean(),ys.mean();mxx=((xs-cx2)**2).mean();myy=((ys-cy2)**2).mean();mxy=((xs-cx2)*(ys-cy2)).mean()
    tr=mxx+myy;dd=tr*tr/4-(mxx*myy-mxy*mxy);sg=math.sqrt(max(0,dd));l1=tr/2+sg;l2=tr/2-sg
    return math.sqrt(l1/max(l2,1e-6))
# ---- scan ----
step=int(40/MPP);hw=int(40/MPP);batch=[];pos=[]
for py in range(hw,W-hw,step):
    for px in range(hw,W-hw,step):
        s=stamp_px(px,py)
        if s is not None: batch.append(s);pos.append((px,py))
X=torch.tensor(np.array(batch,dtype=np.uint8));sc=[]
with torch.no_grad():
    for k in range(0,len(X),512): sc.extend(torch.sigmoid(net(X[k:k+512].unsqueeze(1).float().to(dev)/255.)).cpu().numpy().tolist())
sc=np.array(sc)
idx=np.argsort(-sc);kept=[]
for k in idx:
    if sc[k]<TH: break
    px,py=pos[k]
    if any((px-q[0])**2+(py-q[1])**2 < (60/MPP)**2 for q in kept): continue
    kept.append((px,py,float(sc[k])))
print(f"  {len(kept)} detecții (NMS) >= {TH}",flush=True)
# ---- per-detecție: stampă + gradcam + liniaritate ----
dets=[]
for px,py,s in kept:
    st=stamp_px(px,py)
    if st is None: continue
    _,cam=gradcam(st);lr=lin_ratio(st)
    dets.append({'px':px,'py':py,'sc':s,'stamp':st,'cam':cam,'lr':lr,'cut':lr>RATIO_T})
ncut=sum(d['cut'] for d in dets);nkeep=len(dets)-ncut
print(f"  liniaritate (RATIO_T={RATIO_T}): {nkeep} compacte (KEPT) | {ncut} liniare (CUT)")
# ---- overlay filtrat ----
ov=Image.fromarray(mosA.astype('uint8')).convert('RGB');dr=ImageDraw.Draw(ov)
for d in dets:
    r=int(40/MPP);col=(255,40,40) if d['cut'] else (40,220,40)   # roșu=CUT liniar, verde=KEPT compact
    dr.ellipse([d['px']-r,d['py']-r,d['px']+r,d['py']+r],outline=col,width=3)
ov.resize((min(W,1200),min(W,1200))).save(f"{H}/review/arad_overlay_filtrat.png")
print(f"  -> review/arad_overlay_filtrat.png (verde KEEP={nkeep}, roșu CUT={ncut})")
# ---- montaj Grad-CAM (sortat scor desc): per detecție stampă(sus)+cam(jos) ----
dets.sort(key=lambda d:-d['sc'])
N=len(dets);cols=min(13,N) or 1;rowsN=math.ceil(N/cols);cell=104;hh=24
img=Image.new('RGB',(cols*cell,hh+rowsN*(2*cell+18)),(15,15,15));drw=ImageDraw.Draw(img)
try: ft=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial Bold.ttf',11)
except: ft=ImageFont.load_default()
drw.text((4,5),f"Grad-CAM Arad ({os.path.basename(MODEL)}) — SUS=hillshade, JOS=atenție(roșu). verde=compact, roșu=liniar(CUT). scor·raport",fill=(255,230,90),font=ft)
for i,d in enumerate(dets):
    cxg=(i%cols)*cell;cyg=hh+(i//cols)*(2*cell+18)
    base=Image.fromarray(d['stamp']).convert('RGB')
    heat=Image.fromarray(np.stack([d['cam'],np.zeros_like(d['cam']),255-d['cam']],-1).astype('uint8'))
    over=Image.blend(base,heat,0.55)
    img.paste(base.resize((cell-4,cell-4)),(cxg+2,cyg+16));img.paste(over.resize((cell-4,cell-4)),(cxg+2,cyg+16+cell))
    col=(255,110,110) if d['cut'] else (120,255,120)
    drw.text((cxg+3,cyg+2),f"{d['sc']:.2f} r{d['lr']:.1f}{'C' if d['cut'] else ''}",fill=col,font=ft)
img.save(f"{H}/review/arad_gradcam.png");print(f"  -> review/arad_gradcam.png ({N} detecții)")
