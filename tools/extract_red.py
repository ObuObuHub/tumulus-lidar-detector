import sys,numpy as np,math,csv,os
from PIL import Image
from collections import deque
photo=sys.argv[1]; LON=float(sys.argv[2]); LAT=float(sys.argv[3]); TILE=sys.argv[4]
SRC=sys.argv[5] if len(sys.argv)>5 else 'expert_validated'
im=Image.open(photo).convert('RGB'); W,H=im.size
a=np.asarray(im).astype(int); Rr,G,B=a[:,:,0],a[:,:,1],a[:,:,2]
red=((Rr>140)&(Rr-G>55)&(G<120)).astype(np.uint8)  # red+magenta+pink, exclude gray
lbl=-np.ones(red.shape,dtype=int); comps=[]
for y in range(H):
    for x in range(W):
        if red[y,x] and lbl[y,x]<0:
            q=deque([(y,x)]); lbl[y,x]=len(comps); pix=[]
            while q:
                cy,cx=q.popleft(); pix.append((cx,cy))
                for dy,dx in((1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)):
                    ny,nx=cy+dy,cx+dx
                    if 0<=ny<H and 0<=nx<W and red[ny,nx] and lbl[ny,nx]<0:
                        lbl[ny,nx]=lbl[y,x]; q.append((ny,nx))
            comps.append(pix)
circ=[c for c in comps if len(c)>=25]
S=1000; sc=S/W; Z=17
R=6378137.0;C=2*math.pi*R;ORIG=-20037508.342787;ORIGY=20037508.342787
res=C/(256*2**Z); xC=R*math.radians(LON); yC=R*math.log(math.tan(math.pi/4+math.radians(LAT)/2))
x0=(xC-ORIG)/res-S//2; y0=(ORIGY-yC)/res-S//2
out=[]
for c in circ:
    cx=sum(x for x,y in c)/len(c)*sc; cy=sum(y for x,y in c)/len(c)*sc
    X=ORIG+(x0+cx)*res; Y=ORIGY-(y0+cy)*res
    out.append({'tile':TILE,'id':'','lon':round((X/R)*180/math.pi,6),'lat':round((2*math.atan(math.exp(Y/R))-math.pi/2)*180/math.pi,6),'verdict':'mound','type':'tumul','source':SRC})
L=os.path.expanduser('~/lidar-match/labeled'); mf=f'{L}/labels.csv'
allr=list(csv.DictReader(open(mf)))+out
with open(mf,'w',newline='') as fh:
    w=csv.DictWriter(fh,fieldnames=['tile','id','lon','lat','verdict','type','source']); w.writeheader(); w.writerows(allr)
from collections import Counter
print(f'{TILE}: extracted {len(out)} mounds | totals',dict(Counter(r['source'] for r in allr)))
