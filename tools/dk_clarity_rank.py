#!/usr/bin/env python3
# dk_clarity_rank.py [N] [SEED] — TRIERE DUPĂ CLARITATE. Scorează fiecare pozitiv DK după cât de CLARĂ e movila
# (amplitudinea reliefului central = cât de bine se vede domul), independent de model (NU circular).
# Arată cele mai NECLARE N (faint/șters/ambiguu) pt confirmare Andrei → ține doar imagini clare de antrenament.
# Complementar cu dk_cull_rank.py (care țintește șanțul periferic). NU șterge — doar sortează + planșă.
# -> review/dk_clarity_rank.png + /tmp/dk_clarity_map.csv (idx->file). Roșu = elimină (nu-i movilă clară).
import os,sys,glob,csv
import numpy as np
from PIL import Image,ImageFilter,ImageDraw,ImageFont
H=os.path.expanduser('~/lidar-match')
N=int(sys.argv[1]) if len(sys.argv)>1 else 54
SEED=int(sys.argv[2]) if len(sys.argv)>2 else 0
yy,xx=np.mgrid[0:128,0:128];rad=np.hypot(yy-64,xx-64)
cent=rad<=26;ring=(rad>=34)&(rad<=62)
def clarity(png):
    a=np.asarray(Image.open(png).convert('L').resize((128,128)).filter(ImageFilter.GaussianBlur(1.0)),np.float32)
    c=a[cent]
    # amplitudine relief central (dom clar = mare; șters/plat = mic), normalizată la contrastul global
    amp=float(np.percentile(c,90)-np.percentile(c,10))
    glob=float(np.percentile(a,90)-np.percentile(a,10))+1e-6
    prominence=amp/glob               # cât iese movila din fundal
    return amp*0.5+prominence*40      # scor compozit; MIC = neclar
fs=sorted(glob.glob(f'{H}/dataset_pos_dk/*.png'))
print(f"scorez claritatea a {len(fs)} pozitivi DK...",flush=True)
scored=[]
for i,f in enumerate(fs):
    scored.append((clarity(f),f))
    if i%4000==0: print(f"  {i}/{len(fs)}",flush=True)
vals=np.array([s for s,_ in scored])
print(f"  claritate: min {vals.min():.1f} | mediană {np.median(vals):.1f} | p25 {np.percentile(vals,25):.1f} | max {vals.max():.1f}",flush=True)
print(f"  sub p10 ({np.percentile(vals,10):.1f}): {int((vals<np.percentile(vals,10)).sum())} pozitivi 'neclari'",flush=True)
scored.sort()                         # ascendent: cei mai NECLARI primii
# eșantion din coada neclară (cei mai răi 4*N), randomizat reproductibil pt diversitate
pool=scored[:max(N*4,N)];rng=np.random.default_rng(SEED);pick=rng.permutation(len(pool))[:N]
top=[pool[k] for k in sorted(pick)]
COLS=9;rows=(len(top)+COLS-1)//COLS;CELL=156;HDR=26
img=Image.new('RGB',(COLS*CELL,rows*CELL+HDR),(15,15,15));dr=ImageDraw.Draw(img)
try: ft=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial Bold.ttf',16)
except: ft=ImageFont.load_default()
dr.text((6,4),"Pozitivi DK cei mai NECLARI (movila abia se vede). ROSU = elimina (nu-i imagine clara de antrenament). Nemarcat = pastreaza.",fill=(255,255,80),font=ft)
mp=open('/tmp/dk_clarity_map.csv','w');mw=csv.writer(mp);mw.writerow(['idx','score','file'])
for i,(sc,f) in enumerate(top,1):
    im=Image.open(f).convert('RGB').resize((150,150))
    x=((i-1)%COLS)*CELL;y=((i-1)//COLS)*CELL+HDR
    img.paste(im,(x+3,y+20));dr.text((x+4,y+2),f"#{i}",fill=(120,230,255),font=ft)
    mw.writerow([i,f"{sc:.2f}",os.path.basename(f)])
mp.close();img.save(f'{H}/review/dk_clarity_rank.png')
print(f"-> review/dk_clarity_rank.png (cei mai neclari {len(top)}) + /tmp/dk_clarity_map.csv")
