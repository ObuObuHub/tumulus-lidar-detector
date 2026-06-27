#!/usr/bin/env python3
# samples_native.py CAND_CSV [WIN_M] [TAG] — construiește SAMPLES la rezoluția NATIVĂ a stratului
# disponibil, FĂRĂ downsample la 2m-efectiv (cf. Andrei: „fara sa reducim la 8 pixeli").
# Citește candidați (idx,lon,lat,score) -> crop z16 național la px nativ (~1.62 m/px sol) ->
# board mare + crop-uri individuale mari pt cei marcați în env BIG="4,26,27,39".
import os,sys,math,subprocess,csv
import numpy as np
from PIL import Image,ImageDraw,ImageFont,ImageFilter
CAND=sys.argv[1];WIN_M=float(sys.argv[2]) if len(sys.argv)>2 else 100.0
TAG=sys.argv[3] if len(sys.argv)>3 else 'movileni_native'
BIG=set(int(x) for x in os.environ.get('BIG','').split(',') if x.strip())
H=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R=6378137.0;ORIG=-20037508.342787;ORIGY=20037508.342787;Z=16;MPP=2*math.pi*R/(256*2**Z)
ORG="wCvLzGFkz06gCfBg";svc="1m";TDIR="/tmp/nat_tiles";os.makedirs(TDIR,exist_ok=True)
def merc(lo,la): return R*math.radians(lo), R*math.log(math.tan(math.pi/4+math.radians(la)/2))
def tile(col,row):
    fn=f"{TDIR}/{svc}_{Z}_{col}_{row}.png"
    if not os.path.exists(fn): subprocess.run(["curl","-s","--max-time","25","-o",fn,f"https://tiles.arcgis.com/tiles/{ORG}/arcgis/rest/services/{svc}/MapServer/tile/{Z}/{row}/{col}"],check=False)
    try: return Image.open(fn).convert('L')
    except: return None
def crop(lo,la,win_m):
    MPPg=MPP*math.cos(math.radians(la));hw=int(win_m/2/MPPg)
    cx,cy=merc(lo,la);pxc=(cx-ORIG)/MPP;pyc=(ORIGY-cy)/MPP
    x0=pxc-hw;y0=pyc-hw;W=2*hw;mos=Image.new('L',(W,W),0);nt=0
    for col in range(int(x0//256),int((x0+W)//256)+1):
        for row in range(int(y0//256),int((y0+W)//256)+1):
            t=tile(col,row)
            if t: mos.paste(t,(col*256-int(x0),row*256-int(y0)));nt+=1
    a=np.asarray(mos,np.float32)
    if a.std()<0.3: return None,hw
    lo2,hi2=np.percentile(a,2),np.percentile(a,98);a=np.clip((a-lo2)/(hi2-lo2+1e-6),0,1)
    return (a*255).astype('uint8'),hw
rows=list(csv.DictReader(open(CAND)))
print(f"{len(rows)} candidați | fereastră {WIN_M:.0f}m | native z16 ~{MPP*math.cos(math.radians(47.33)):.2f} m/px sol, FĂRĂ downsample")
try: fnt=ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf",16)
except: fnt=ImageFont.load_default()
# ---- board ----
DISP=240;cols=5;n=len(rows);rr=(n+cols-1)//cols
bd=Image.new('RGB',(cols*DISP,rr*DISP+26),(18,18,18));bdr=ImageDraw.Draw(bd)
bdr.text((6,5),f"{TAG} — {n} samples native z16 (~1.6 m/px), fereastra {WIN_M:.0f}m, FARA downsample",fill=(255,255,0),font=fnt)
for k,r in enumerate(rows):
    i=int(r['idx']);lo=float(r['lon']);la=float(r['lat']);v=float(r['score'])
    arr,hw=crop(lo,la,WIN_M)
    cx2=(k%cols)*DISP;cy2=(k//cols)*DISP+26
    if arr is not None:
        im=Image.fromarray(arr).resize((DISP-8,DISP-30),Image.BICUBIC).convert('RGB')
        bd.paste(im,(cx2+4,cy2+4))
        if i in BIG: bdr.rectangle([cx2+3,cy2+3,cx2+DISP-5,cy2+DISP-26],outline=(0,200,255),width=3)
    px_native=int(WIN_M/(MPP*math.cos(math.radians(la))))
    bdr.text((cx2+6,cy2+DISP-24),f"#{i} {v:.2f} ({px_native}px)",fill=(0,255,120),font=fnt)
os.makedirs(f"{H}/review",exist_ok=True)
bd.save(f"{H}/review/{TAG}_board.jpg",quality=90)
print(f"-> review/{TAG}_board.jpg")
# ---- crop-uri mari individuale ----
if BIG:
    big=[r for r in rows if int(r['idx']) in BIG]
    BS=380;bcols=min(4,len(big));brr=(len(big)+bcols-1)//bcols
    bg=Image.new('RGB',(bcols*BS,brr*BS+26),(18,18,18));bgr=ImageDraw.Draw(bg)
    bgr.text((6,5),f"{TAG} — crop-uri mari (fereastra {WIN_M:.0f}m, native)",fill=(255,255,0),font=fnt)
    for k,r in enumerate(big):
        i=int(r['idx']);lo=float(r['lon']);la=float(r['lat']);v=float(r['score'])
        arr,hw=crop(lo,la,WIN_M)
        cx2=(k%bcols)*BS;cy2=(k//bcols)*BS+26
        if arr is not None:
            im=Image.fromarray(arr).resize((BS-8,BS-30),Image.BICUBIC).convert('RGB');bg.paste(im,(cx2+4,cy2+4))
        bgr.text((cx2+6,cy2+BS-24),f"#{i} scor {v:.2f}  {lo:.5f},{la:.5f}",fill=(0,255,120),font=fnt)
    bg.save(f"{H}/review/{TAG}_big.jpg",quality=92)
    print(f"-> review/{TAG}_big.jpg")
