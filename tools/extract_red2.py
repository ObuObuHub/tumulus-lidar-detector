#!/usr/bin/env python3
# extract_red2.py photo LON LAT TILE [source] [--write]
# REGULA (gandita din "cum se antreneaza corect modelul"):
#  1. cercul rosu/magenta = ZONA de interes, nu centrul exact.
#  2. centrul = domul intunecat real din SLRM din interiorul cercului (SNAP la min local).
#  3. un cerc = o movila; inel intrerupt -> arcurile se unesc (sigilare minima 1px) + snap+dedup.
#  4. dedup centre < 20 m.
#  5. cerc fara dom clar -> raportat separat (nu bagat tacut).
# Dry-run by default: overlay /tmp/extract_overlay.png (verde=centru, galben=fara dom clar).
import sys,math,csv,os
import numpy as np
from PIL import Image,ImageDraw,ImageFilter
from collections import deque
photo=sys.argv[1]; LON=float(sys.argv[2]); LAT=float(sys.argv[3]); TILE=sys.argv[4]
SRC='expert_validated'; WRITE=False
for a in sys.argv[5:]:
    if a=='--write': WRITE=True
    else: SRC=a
WINM=1194.0  # latura plăcii native (1000px @ z17 1m); placa Telegram poate fi rescalata
im=Image.open(photo).convert('RGB'); W,H=im.size
mpp=WINM/W; Ddedup=20.0/mpp   # 20 m in px
a=np.asarray(im).astype(int); Rr,G,B=a[:,:,0],a[:,:,1],a[:,:,2]
red=((Rr>140)&(Rr-G>55)&(G<120))
gray=np.asarray(im.convert('L')).astype(np.float32); gray[red]=255
graysm=np.asarray(Image.fromarray(gray.astype('uint8')).filter(ImageFilter.GaussianBlur(3))).astype(np.float32)
sealed=np.asarray(Image.fromarray((red*255).astype('uint8')).filter(ImageFilter.MaxFilter(3)))>0
# connected components
lbl=-np.ones(sealed.shape,int); comps=[]
ys,xs=np.where(sealed)
for y,x in zip(ys.tolist(),xs.tolist()):
    if lbl[y,x]>=0: continue
    q=deque([(y,x)]); lbl[y,x]=len(comps); pix=[]
    while q:
        cy,cx=q.popleft(); pix.append((cx,cy))
        for dy,dx in((1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)):
            ny,nx=cy+dy,cx+dx
            if 0<=ny<H and 0<=nx<W and sealed[ny,nx] and lbl[ny,nx]<0:
                lbl[ny,nx]=lbl[y,x]; q.append((ny,nx))
    comps.append(pix)
comps=[c for c in comps if len(c)>=120]
def snap(cx,cy,bb):
    r=int(min(max(bb*0.5,12),30))
    x0=max(0,int(cx-r)); x1=min(W,int(cx+r)); y0=max(0,int(cy-r)); y1=min(H,int(cy+r))
    win=graysm[y0:y1,x0:x1]
    if win.size==0 or win.min()>win.mean()-10: return (cx,cy,False)
    iy,ix=np.unravel_index(np.argmin(win),win.shape); return (x0+ix,y0+iy,True)
cands=[]
for c in comps:
    xs2=[x for x,y in c]; ys2=[y for x,y in c]
    cx=sum(xs2)/len(c); cy=sum(ys2)/len(c); bb=max(max(xs2)-min(xs2),max(ys2)-min(ys2))
    cands.append(snap(cx,cy,bb))
uniq=[]
for p in cands:
    if all(math.hypot(p[0]-u[0],p[1]-u[1])>=Ddedup for u in uniq): uniq.append(p)
S=1000; sc=S/W; Z=17
R=6378137.0;C=2*math.pi*R;ORIG=-20037508.342787;ORIGY=20037508.342787
res=C/(256*2**Z); xC=R*math.radians(LON); yC=R*math.log(math.tan(math.pi/4+math.radians(LAT)/2))
x0=(xC-ORIG)/res-S//2; y0=(ORIGY-yC)/res-S//2
out=[]; noflag=0
for cx,cy,ok in uniq:
    if not ok: noflag+=1; continue   # cerc fara dom clar: NU il scriu (raportez)
    X=ORIG+(x0+cx*sc)*res; Y=ORIGY-(y0+cy*sc)*res
    out.append({'tile':TILE,'id':'','lon':round((X/R)*180/math.pi,6),'lat':round((2*math.atan(math.exp(Y/R))-math.pi/2)*180/math.pi,6),'verdict':'mound','type':'tumul','source':SRC})
ov=im.copy(); dr=ImageDraw.Draw(ov)
for i,(cx,cy,ok) in enumerate(uniq,1):
    col=(0,255,0) if ok else (255,230,0)
    dr.ellipse([cx-9,cy-9,cx+9,cy+9],outline=col,width=3); dr.text((cx+10,cy-10),str(i),fill=col)
ov.save('/tmp/extract_overlay.png')
print(f'cercuri {len(uniq)} -> movile {len(out)} | fara dom clar (galben, neraportate): {noflag} | dedup 20m')
if WRITE:
    L=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'labeled'); mf=f'{L}/labels.csv'
    allr=list(csv.DictReader(open(mf)))+out
    with open(mf,'w',newline='') as fh:
        w=csv.DictWriter(fh,fieldnames=['tile','id','lon','lat','verdict','type','source']); w.writeheader(); w.writerows(allr)
    from collections import Counter
    print(f'SCRIS {len(out)} | totals',dict(Counter(r['source'] for r in allr)))
else:
    print(f'DRY-RUN {len(out)} -> /tmp/extract_overlay.png (--write ca sa scrii)')
