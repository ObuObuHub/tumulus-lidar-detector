#!/usr/bin/env python3
# dk_clarity_cull.py [PCT] [--commit] — TĂIERE AUTOMATĂ după claritate, sub percentila PCT (implicit 10).
# Fără --commit = DRY-RUN: scorează toți, alege candidații de tăiat (sub prag), respectă ce-a marcat Andrei
# „păstrează" (removed=0 în labeled/dk_clarity_labels.csv), și scoate 2 montaje de confirmare:
#   review/dk_cut_sample.png   = 54 random din setul de tăiat (chiar e gunoi?)
#   review/dk_cut_boundary.png = 54 la graniță (27 sub prag = TAI / 27 peste = ȚIN) ca să vezi linia.
# Cu --commit = mută candidații în dataset_pos_dk_culled/ + log labeled/dk_clarity_labels.csv (removed=1).
import os,sys,glob,csv
import numpy as np
from PIL import Image,ImageFilter,ImageDraw,ImageFont
H=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PCT=float(sys.argv[1]) if len(sys.argv)>1 and not sys.argv[1].startswith('--') else 10.0
COMMIT='--commit' in sys.argv
yy,xx=np.mgrid[0:128,0:128];rad=np.hypot(yy-64,xx-64);cent=rad<=26
def clarity(png):
    a=np.asarray(Image.open(png).convert('L').resize((128,128)).filter(ImageFilter.GaussianBlur(1.0)),np.float32)
    c=a[cent];amp=float(np.percentile(c,90)-np.percentile(c,10))
    glob_=float(np.percentile(a,90)-np.percentile(a,10))+1e-6
    return amp*0.5+(amp/glob_)*40
# marcaje umane: nu tăia ce-a zis Andrei „păstrează"
kept_human=set()
labf=f'{H}/labeled/dk_clarity_labels.csv'
if os.path.exists(labf):
    for r in csv.DictReader(open(labf)):
        if r['removed']=='0': kept_human.add(r['file'])
fs=sorted(glob.glob(f'{H}/dataset_pos_dk/*.png'))
print(f"scorez {len(fs)} pozitivi DK (PCT={PCT}, {'COMMIT' if COMMIT else 'DRY-RUN'})...",flush=True)
scored=[]
for i,f in enumerate(fs):
    scored.append((clarity(f),f))
    if i%5000==0: print(f"  {i}/{len(fs)}",flush=True)
vals=np.array([s for s,_ in scored]);thr=np.percentile(vals,PCT)
scored.sort()
cut=[(s,f) for s,f in scored if s<thr and os.path.basename(f) not in kept_human]
keepside=[(s,f) for s,f in scored if s>=thr]
print(f"  prag p{PCT:.0f} = {thr:.1f} | de tăiat {len(cut)} (excluse {sum(1 for s,f in scored if s<thr and os.path.basename(f) in kept_human)} marcate de tine păstrează) | rămân {len(fs)-len(cut)}",flush=True)
def montage(items,path,title):
    COLS=9;rows=(len(items)+COLS-1)//COLS;CELL=156;HDR=26
    img=Image.new('RGB',(COLS*CELL,rows*CELL+HDR),(15,15,15));dr=ImageDraw.Draw(img)
    try: ft=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial Bold.ttf',15)
    except: ft=ImageFont.load_default()
    dr.text((6,5),title,fill=(255,255,80),font=ft)
    for i,(sc,f,tag,col) in enumerate(items):
        im=Image.open(f).convert('RGB').resize((150,150));x=(i%COLS)*CELL;y=(i//COLS)*CELL+HDR
        img.paste(im,(x+3,y+20));dr.text((x+4,y+2),f"{tag}{sc:.0f}",fill=col,font=ft)
    img.save(path);print(f"  -> {path}")
if not COMMIT:
    rng=np.random.default_rng(7)
    samp=[cut[k] for k in sorted(rng.permutation(len(cut))[:54])]
    montage([(s,f,'',(255,120,120)) for s,f in samp],f'{H}/review/dk_cut_sample.png',
            f"EȘANTION 54 din {len(cut)} care s-ar TĂIA (sub p{PCT:.0f}={thr:.0f}). Toate gunoi? Daca vezi movile bune, oprește.")
    below=cut[-27:]                       # cele mai aproape de prag, dedesubt (tăiate)
    above=keepside[:27]                   # cele mai aproape de prag, deasupra (ținute)
    bnd=[(s,f,'TAI ',(255,120,120)) for s,f in below]+[(s,f,'TIN ',(120,255,120)) for s,f in above]
    montage(bnd,f'{H}/review/dk_cut_boundary.png',
            f"GRANITA p{PCT:.0f}: rosu=ultimele TAIATE / verde=primele TINUTE. Linia separa gunoi de movile?")
    print("  DRY-RUN: nimic mutat. Rulează cu --commit după confirmare.")
else:
    import shutil
    os.makedirs(f'{H}/dataset_pos_dk_culled',exist_ok=True)
    new=not os.path.exists(labf);lg=open(labf,'a');w=csv.writer(lg)
    if new: w.writerow(['file','removed'])
    n=0
    for s,f in cut:
        bn=os.path.basename(f);dst=f'{H}/dataset_pos_dk_culled/{bn}'
        if os.path.exists(f): shutil.move(f,dst);w.writerow([bn,1]);n+=1
    lg.close();print(f"  COMMIT: mutat {n} în dataset_pos_dk_culled/. DK rămași {len(glob.glob(f'{H}/dataset_pos_dk/*.png'))}.")
