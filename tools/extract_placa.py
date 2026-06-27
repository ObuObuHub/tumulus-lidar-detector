#!/usr/bin/env python3
# extract_placa.py photo CLON CLAT WINDOW_M [source] [--dark|--bright] [--write]
# Extrage cercurile rosii/magenta dintr-o placa centrata (CLON,CLAT) cu latura WINDOW_M metri.
# Geometrie LINIARA (nu presupune z17): offset = (px/W - 0.5)*WINDOW_M. Merge pt ORICE placa.
# Snap pe dom: --dark (MDH/hillshade: movila=intunecata) implicit, --bright (SLRM-DTM: luminoasa).
import sys,math,csv,os
import numpy as np
from PIL import Image,ImageFilter,ImageDraw
from collections import deque
photo=sys.argv[1]; CLON=float(sys.argv[2]); CLAT=float(sys.argv[3]); WINM=float(sys.argv[4])
SRC='expert_validated'; WRITE=False; DARK=True
for a in sys.argv[5:]:
    if a=='--write': WRITE=True
    elif a=='--bright': DARK=False
    elif a=='--dark': DARK=True
    else: SRC=a
im=Image.open(photo).convert('RGB'); W,Hh=im.size
mpp=WINM/W; Ddedup=20.0/mpp
a=np.asarray(im).astype(int); Rr,G,B=a[:,:,0],a[:,:,1],a[:,:,2]
red=((Rr>140)&(Rr-G>55)&(G<120))
gray=np.asarray(im.convert('L')).astype(np.float32)
gray[red]=255 if DARK else 0   # masca marcajul ca sa nu fie ales ca dom
graysm=np.asarray(Image.fromarray(gray.astype('uint8')).filter(ImageFilter.GaussianBlur(3))).astype(np.float32)
sealed=np.asarray(Image.fromarray((red*255).astype('uint8')).filter(ImageFilter.MaxFilter(3)))>0
lbl=-np.ones(sealed.shape,int); comps=[]
ys,xs=np.where(sealed)
for y,x in zip(ys.tolist(),xs.tolist()):
    if lbl[y,x]>=0: continue
    q=deque([(y,x)]); lbl[y,x]=len(comps); pix=[]
    while q:
        cy,cx=q.popleft(); pix.append((cx,cy))
        for dy,dx in((1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)):
            ny,nx=cy+dy,cx+dx
            if 0<=ny<Hh and 0<=nx<W and sealed[ny,nx] and lbl[ny,nx]<0:
                lbl[ny,nx]=lbl[y,x]; q.append((ny,nx))
    comps.append(pix)
comps=[c for c in comps if len(c)>=120]
def snap(cx,cy,bb):
    r=int(min(max(bb*0.5,12),30))
    x0=max(0,int(cx-r)); x1=min(W,int(cx+r)); y0=max(0,int(cy-r)); y1=min(Hh,int(cy+r))
    win=graysm[y0:y1,x0:x1]
    if win.size==0: return (cx,cy,False)
    if DARK:
        if win.min()>win.mean()-10: return (cx,cy,False)
        iy,ix=np.unravel_index(np.argmin(win),win.shape)
    else:
        if win.max()<win.mean()+10: return (cx,cy,False)
        iy,ix=np.unravel_index(np.argmax(win),win.shape)
    return (x0+ix,y0+iy,True)
cands=[]
for c in comps:
    xs2=[x for x,y in c]; ys2=[y for x,y in c]
    cx=sum(xs2)/len(c); cy=sum(ys2)/len(c); bb=max(max(xs2)-min(xs2),max(ys2)-min(ys2))
    cands.append(snap(cx,cy,bb))
uniq=[]
for p in cands:
    if all(math.hypot(p[0]-u[0],p[1]-u[1])>=Ddedup for u in uniq): uniq.append(p)
out=[]; noflag=0
mlon=111320*math.cos(math.radians(CLAT)); mlat=110540
for cx,cy,ok in uniq:
    if not ok: noflag+=1; continue
    dlon=((cx/W)-0.5)*WINM/mlon; dlat=(0.5-(cy/Hh))*WINM/mlat
    out.append({'tile':SRC[:6].upper()+'_PLACA','id':'','lon':round(CLON+dlon,6),'lat':round(CLAT+dlat,6),'verdict':'mound','type':'tumul','source':SRC})
ov=im.copy(); dr=ImageDraw.Draw(ov)
for i,(cx,cy,ok) in enumerate(uniq,1):
    col=(0,255,0) if ok else (255,230,0)
    dr.ellipse([cx-9,cy-9,cx+9,cy+9],outline=col,width=3); dr.text((cx+10,cy-10),str(i),fill=col)
ov.save('/tmp/extract_overlay.png')
print(f'cercuri {len(uniq)} -> movile {len(out)} | fara dom clar {noflag} | snap={"DARK" if DARK else "BRIGHT"}')
if WRITE:
    mf=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'labeled/labels.csv')
    allr=list(csv.DictReader(open(mf)))+out
    with open(mf,'w',newline='') as fh:
        w=csv.DictWriter(fh,fieldnames=['tile','id','lon','lat','verdict','type','source']); w.writeheader(); w.writerows(allr)
    from collections import Counter
    print(f'SCRIS {len(out)} | totals',dict(Counter(r['source'] for r in allr)))
else:
    print(f'DRY-RUN {len(out)} -> /tmp/extract_overlay.png')
