#!/usr/bin/env python3
# arad_diag2.py CLON CLAT KM [MODEL_2CH] — testează modelul 2-CANALE (hillshade+curbură) pe over-firing Arad.
# Răspunde la întrebarea lui Andrei: canalul de CURBURĂ taie FP-urile de pe liniile drepte de șanț?
# Scanează o dată, raportează detecții la 0.6/0.8/0.9 (vs 78/42/18 ale modelului 1-canal) + montaj ch0|curbură|Grad-CAM.
# Preprocesare IDENTICĂ cu train_2ch.py: ch0=homog(_histeq·blur0.8), ch1=curv(Laplacian·blur1.2). ch0 == stampa 1ch (comparație corectă).
import sys,os,math,subprocess
import numpy as np
from PIL import Image,ImageFilter,ImageDraw,ImageFont
import torch,torch.nn as nn,torch.nn.functional as Fn
H=os.path.dirname(os.path.dirname(os.path.abspath(__file__)));dev=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
CLON=float(sys.argv[1]);CLAT=float(sys.argv[2]);KM=float(sys.argv[3]) if len(sys.argv)>3 else 1.5
MODEL=sys.argv[4] if len(sys.argv)>4 else f'{H}/combined_2ch.pt'
R=6378137.0;ORIG=-20037508.342787;ORIGY=20037508.342787;Z=18;MPP=2*math.pi*R/(256*2**Z)
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
# ---- preprocesare IDENTICĂ train_2ch ----
def _histeq(a):
    h=np.bincount(a.ravel(),minlength=256).astype(np.float64);cdf=h.cumsum()
    return a if cdf[-1]==0 else (cdf[a]/cdf[-1]*255).astype(np.uint8)
def homog(a): return _histeq(np.asarray(Image.fromarray(a).filter(ImageFilter.GaussianBlur(0.8)),np.uint8))
def curv(a):
    g=np.asarray(Image.fromarray(a).filter(ImageFilter.GaussianBlur(1.2)),np.float32)
    lap=(np.roll(g,1,0)+np.roll(g,-1,0)+np.roll(g,1,1)+np.roll(g,-1,1)-4*g)
    lo,hi=np.percentile(lap,2),np.percentile(lap,98)
    return np.clip((lap-lo)/(hi-lo+1e-6)*255,0,255).astype(np.uint8)
def stamp_raw(px,py,meters=80,eff=2.0,out=128):
    hw=int(meters/2/MPP);w=mosA[py-hw:py+hw,px-hw:px+hw]
    if w.shape!=(2*hw,2*hw) or w.std()<0.5: return None
    lo,hi=np.percentile(w,2),np.percentile(w,98);a=np.clip((w-lo)/(hi-lo+1e-6),0,1)
    f=max(1,int(round(eff/MPP)));Hh,Ww=a.shape;a=a[:Hh//f*f,:Ww//f*f].reshape(Hh//f,f,Ww//f,f).mean((1,3))
    return np.asarray(Image.fromarray((a*255).astype('uint8')).resize((out,out)),np.uint8)
def stack2(raw): return np.stack([homog(raw),curv(raw)])  # [2,128,128]
class Net(nn.Module):
    def __init__(s):
        super().__init__();s.c=nn.Sequential(nn.Conv2d(2,16,3,2,1),nn.ReLU(),nn.Conv2d(16,32,3,2,1),nn.ReLU(),nn.Conv2d(32,64,3,2,1),nn.ReLU(),nn.AdaptiveAvgPool2d(1));s.f=nn.Linear(64,1)
    def forward(s,x): return s.f(s.c(x).flatten(1)).squeeze(1)
net=Net().to(dev);net.load_state_dict(torch.load(MODEL,map_location=dev,weights_only=True));net.eval()
acts={};grads={}
net.c[4].register_forward_hook(lambda m,i,o: acts.__setitem__('v',o.detach()))
net.c[4].register_full_backward_hook(lambda m,gi,go: grads.__setitem__('v',go[0].detach()))
def gradcam(st2):
    x=torch.tensor(st2).unsqueeze(0).float().to(dev)/255.;x.requires_grad_(True)
    net.zero_grad();out=net(x);out.backward()
    A=acts['v'][0];G=grads['v'][0];wts=G.mean(dim=(1,2));cam=Fn.relu((wts[:,None,None]*A).sum(0))
    cam=cam.cpu().numpy();cam=cam/(cam.max()+1e-9)
    return float(torch.sigmoid(out).item()),np.asarray(Image.fromarray((cam*255).astype('uint8')).resize((128,128)))
# ---- scan ----
step=int(40/MPP);hw=int(40/MPP);raws=[];pos=[]
for py in range(hw,W-hw,step):
    for px in range(hw,W-hw,step):
        r=stamp_raw(px,py)
        if r is not None: raws.append(r);pos.append((px,py))
ST=np.array([stack2(r) for r in raws],dtype=np.uint8)
X=torch.tensor(ST);sc=[]
with torch.no_grad():
    for k in range(0,len(X),512): sc.extend(torch.sigmoid(net(X[k:k+512].float().to(dev)/255.)).cpu().numpy().tolist())
sc=np.array(sc)
def nms(th):
    idx=np.argsort(-sc);kept=[]
    for k in idx:
        if sc[k]<th: break
        px,py=pos[k]
        if any((px-q[0])**2+(py-q[1])**2 < (60/MPP)**2 for q in kept): continue
        kept.append((px,py,float(sc[k]),k))
    return kept
print("  MODEL 2-CANALE (hillshade+curbură) — detecții (NMS):")
for th in (0.6,0.8,0.9):
    print(f"    >= {th}: {len(nms(th))}   (1-canal era: {{0.6:78,0.8:42,0.9:18}}[{th}])")
kept=nms(0.6)
# ---- overlay ----
ov=Image.fromarray(mosA.astype('uint8')).convert('RGB');dr=ImageDraw.Draw(ov)
for px,py,s,_ in kept:
    r=int(40/MPP);col=(255,40,40) if s>=0.9 else (255,170,40)
    dr.ellipse([px-r,py-r,px+r,py+r],outline=col,width=3)
ov.resize((min(W,1200),min(W,1200))).save(f"{H}/review/arad_2ch_overlay.png")
print(f"  -> review/arad_2ch_overlay.png ({len(kept)} detecții @0.6)")
# ---- montaj ch0|curbură|gradcam pt top-24 ----
top=kept[:24];cell=120;hh=26;cols=6;rowsN=math.ceil(len(top)/cols)
img=Image.new('RGB',(cols*3*cell+ (cols-1)*8, hh+rowsN*(cell+18)),(15,15,15));drw=ImageDraw.Draw(img)
try: ft=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial Bold.ttf',12)
except: ft=ImageFont.load_default()
drw.text((4,5),"2-CANALE top-24 @0.6 — triada: HILLSHADE | CURBURĂ(canal2) | Grad-CAM(roșu=atenție). Dom=calotă curbă; șanț=linie.",fill=(255,230,90),font=ft)
for i,(px,py,s,k) in enumerate(top):
    st=stack2(raws[k]);_,cam=gradcam(st)
    ch0=Image.fromarray(st[0]).convert('RGB');ch1=Image.fromarray(st[1]).convert('RGB')
    heat=Image.fromarray(np.stack([cam,np.zeros_like(cam),255-cam],-1).astype('uint8'))
    over=Image.blend(ch0,heat,0.55)
    gx=(i%cols)*(3*cell+8);gy=hh+(i//cols)*(cell+18)
    for j,im in enumerate((ch0,ch1,over)): img.paste(im.resize((cell-2,cell-2)),(gx+j*cell,gy+14))
    drw.text((gx+2,gy+1),f"#{i+1} {s:.2f}",fill=(255,210,90),font=ft)
img.save(f"{H}/review/arad_2ch_montaj.png");print(f"  -> review/arad_2ch_montaj.png")
