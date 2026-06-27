#!/usr/bin/env python3
# dk_cull_rank.py [N] — ACTIVE LEARNING: scorează toți pozitivii DK cu liniaritate periferică (proxy „șanț mare lângă
# movilă"), arată top-N cei mai probabil contaminați pt confirmare Andrei (randament mare). NU șterge — doar sortează.
# -> review/dk_cull_rank.png + /tmp/dk_cull_map.csv (idx->file). Andrei confirmă cu roșu, apoi detect+ștergere.
import os,sys,glob,math,csv
import numpy as np
from PIL import Image,ImageFilter,ImageDraw,ImageFont
H=os.path.expanduser('~/lidar-match');N=int(sys.argv[1]) if len(sys.argv)>1 else 54
yy,xx=np.mgrid[0:128,0:128];rad=np.hypot(yy-64,xx-64);ring=(rad>=34)&(rad<=62);cent=rad<=22
def feats(png):
    a=np.asarray(Image.open(png).convert('L').resize((128,128)).filter(ImageFilter.GaussianBlur(1.0)),np.float32)
    gy,gx=np.gradient(a)
    gx2=gx[ring];gy2=gy[ring];Jxx=(gx2*gx2).mean();Jyy=(gy2*gy2).mean();Jxy=(gx2*gy2).mean()
    coh=math.sqrt((Jxx-Jyy)**2+4*Jxy*Jxy)/(Jxx+Jyy+1e-9);energy=float(np.hypot(gx2,gy2).mean())
    # convexitate centrală (movilă curată = calotă puternică); contaminat = șanțul domină periferia
    lap=(np.roll(a,1,0)+np.roll(a,-1,0)+np.roll(a,1,1)+np.roll(a,-1,1)-4*a)
    central=abs(float(lap[cent].mean()))
    return coh*energy, central
fs=sorted(glob.glob(f'{H}/dataset_pos_dk/*.png'))
print(f"scorez {len(fs)} pozitivi DK...",flush=True)
scored=[]
for i,f in enumerate(fs):
    pe,central=feats(f);score=pe-0.4*central   # liniaritate periferică MARE, convexitate centrală MICĂ = contaminat
    scored.append((score,f))
    if i%3000==0: print(f"  {i}/{len(fs)}",flush=True)
scored.sort(reverse=True);top=scored[:N]
COLS=9;rows=(len(top)+COLS-1)//COLS;CELL=156;HDR=26
img=Image.new('RGB',(COLS*CELL,rows*CELL+HDR),(15,15,15));dr=ImageDraw.Draw(img)
try: ft=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial Bold.ttf',16)
except: ft=ImageFont.load_default()
dr.text((6,4),f"Top {len(top)} pozitivi DK cei mai PROBABIL contaminați (șanț mare). Roșu = confirmi că-i de eliminat. Nemarcat = de fapt curat.",fill=(255,255,80),font=ft)
mp=open('/tmp/dk_cull_map.csv','w');mw=csv.writer(mp);mw.writerow(['idx','lon','lat','score','file'])
for i,(sc,f) in enumerate(top,1):
    im=Image.open(f).convert('RGB').resize((150,150))
    x=((i-1)%COLS)*CELL;y=((i-1)//COLS)*CELL+HDR
    img.paste(im,(x+3,y+20));dr.text((x+4,y+2),f"#{i}",fill=(120,230,255),font=ft)
    mw.writerow([i,"0","0",f"{sc:.2f}",os.path.basename(f)])
mp.close();img.save(f'{H}/review/dk_cull_rank.png')
print(f"-> review/dk_cull_rank.png (top {len(top)}) + /tmp/dk_cull_map.csv")
