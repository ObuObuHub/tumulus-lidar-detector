#!/usr/bin/env python3
# dk_cull_board.py [N] [seed] — montaj numerotat cu N pozitivi DANEZI random pt CURĂȚARE manuală.
# Andrei marchează cu ROȘU cele care NU seamănă cu un tumul românesc (de eliminat, ar confuza modelul).
# -> review/dk_cull_board.png + /tmp/dk_cull_map.csv (idx,file). Apoi detect_board_marks + ștergere.
import os,sys,glob,random,csv
from PIL import Image,ImageDraw,ImageFont
H=os.path.expanduser('~/lidar-match')
N=int(sys.argv[1]) if len(sys.argv)>1 else 54
SEED=int(sys.argv[2]) if len(sys.argv)>2 else 42
fs=sorted(glob.glob(f'{H}/dataset_pos_dk/*.png'))
random.seed(SEED);samp=random.sample(fs,min(N,len(fs)))
COLS=9;rows=(len(samp)+COLS-1)//COLS;CELL=156;HDR=26
img=Image.new('RGB',(COLS*CELL,rows*CELL+HDR),(15,15,15));dr=ImageDraw.Draw(img)
try: ft=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial Bold.ttf',16)
except: ft=ImageFont.load_default()
dr.text((6,4),f"Pozitivi DANEZI (eșantion {len(samp)}). Marchează cu ROȘU cele care NU seamănă cu tumul RO (de ELIMINAT, ar confuza modelul). Restul = păstrăm.",fill=(255,255,80),font=ft)
mp=open('/tmp/dk_cull_map.csv','w');mw=csv.writer(mp);mw.writerow(['idx','lon','lat','score','file'])
for i,f in enumerate(samp,1):
    im=Image.open(f).convert('RGB').resize((150,150))
    x=((i-1)%COLS)*CELL;y=((i-1)//COLS)*CELL+HDR
    img.paste(im,(x+3,y+20));dr.text((x+4,y+2),f"#{i}",fill=(120,230,255),font=ft)
    mw.writerow([i,"0","0","0",os.path.basename(f)])  # lon/lat/score dummy ca să meargă detect_board_marks
mp.close()
img.save(f'{H}/review/dk_cull_board.png')
print(f"-> review/dk_cull_board.png ({len(samp)} pozitivi danezi) + /tmp/dk_cull_map.csv")
