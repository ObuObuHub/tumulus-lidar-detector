#!/usr/bin/env python3
# clean_positives.py [DIR=dataset_pos_dk] [--apply]
# Filtreaza pozitivii zgomotosi (cf. judecatii lui Andrei: dom clar pe fundal calm).
# Scor curatenie = scor_model (combined_cnn.pt) + (contrast_dom_central - 0.5*zgomot_periferic).
# Calibreaza pragul pe marcajele lui Andrei (/tmp/dk_marks.json pe dataset_pos_dk.bak299),
# il aplica pe DIR, scoate montaje KEPT vs DROPPED + lista. NU sterge fara --apply (muta in DIR.dropped/).
import sys,os,glob,json,random,shutil
import numpy as np
from PIL import Image,ImageDraw
import torch,torch.nn as nn
H=os.path.expanduser('~/lidar-match'); dev=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
DIR=sys.argv[1] if len(sys.argv)>1 and not sys.argv[1].startswith('--') else f'{H}/dataset_pos_dk'
APPLY='--apply' in sys.argv
class Net(nn.Module):
    def __init__(s):
        super().__init__();s.c=nn.Sequential(nn.Conv2d(1,16,3,2,1),nn.ReLU(),nn.Conv2d(16,32,3,2,1),nn.ReLU(),nn.Conv2d(32,64,3,2,1),nn.ReLU(),nn.AdaptiveAvgPool2d(1));s.f=nn.Linear(64,1)
    def forward(s,x): return s.f(s.c(x).flatten(1)).squeeze(1)
net=Net().to(dev);net.load_state_dict(torch.load(f'{H}/combined_cnn.pt',weights_only=True));net.eval()
def feats(path):
    im=Image.open(path).convert('L').resize((128,128)); a=np.asarray(im,np.float32)/255.
    n=a.shape[0]; m=int(n*0.28)
    mask=np.ones_like(a,bool); mask[m:n-m,m:n-m]=False
    periph=a[mask].std(); cen=a[m:n-m,m:n-m]; cen_rng=np.percentile(cen,90)-np.percentile(cen,10)
    with torch.no_grad(): ms=float(torch.sigmoid(net(torch.tensor(a).unsqueeze(0).unsqueeze(0).to(dev))).cpu())
    return float(ms + (cen_rng-0.5*periph))   # scor curatenie (mare=curat)
# 1. CALIBRARE pe marcajele lui Andrei
marks=json.load(open('/tmp/dk_marks.json')); BAK=f'{H}/dataset_pos_dk.bak299/'
lab=[(o['marked'],feats(BAK+o['file'])) for o in marks]
vals=sorted(s for _,s in lab); best=(0,vals[0])
for t in vals:
    ok=sum(1 for mk,s in lab if (s<t)==mk)  # s<t -> zgomotos
    if ok>best[0]: best=(ok,t)
acc,T=best; T=float(T)
noisy=[s for mk,s in lab if mk]; clean=[s for mk,s in lab if not mk]
print(f"CALIBRARE pe 48 marcaje: acc {acc}/48 ({acc/48:.0%}) | prag T={T:.3f}")
print(f"  curat med {np.median(clean):.3f} | zgomot med {np.median(noisy):.3f}")
# 2. APLIC pe DIR
fs=sorted(glob.glob(f'{DIR}/*.png'))
sc=[(f,feats(f)) for f in fs]
keep=[f for f,s in sc if s>=T]; drop=[f for f,s in sc if s<T]
print(f"\n{DIR}: {len(fs)} pozitivi -> PASTREZ {len(keep)} ({len(keep)/max(1,len(fs)):.0%}) | ARUNC {len(drop)} ({len(drop)/max(1,len(fs)):.0%})")
json.dump({"keep":[os.path.basename(f) for f in keep],"drop":[os.path.basename(f) for f in drop],"T":T},open('/tmp/clean_keep.json','w'))
# 3. montaje KEPT (random) + DROPPED (cei mai zgomotosi)
def montage(files,title,out,n=48):
    random.seed(2); ff=list(files);
    if 'DROP' in title.upper(): ff=[f for f,s in sorted(sc,key=lambda x:x[1]) if f in set(files)][:n]  # cei mai zgomotosi
    else:
        random.shuffle(ff); ff=ff[:n]
    cols=8;rows=(len(ff)+cols-1)//cols;cell=128
    M=Image.new('RGB',(cols*(cell+4)+4,rows*(cell+4)+24),(25,25,25));d=ImageDraw.Draw(M)
    d.text((6,6),title,fill=(200,200,120))
    for i,f in enumerate(ff):
        M.paste(Image.open(f).convert('RGB').resize((cell,cell)),((i%cols)*(cell+4)+4,(i//cols)*(cell+4)+20))
    M.save(out);print("  ->",out,len(ff))
montage(keep,f"PASTRATI (curati) - esantion din {len(keep)}",f'{H}/review/dk_kept.png')
montage(drop,f"ARUNCATI (zgomotosi) - cei mai zgomotosi din {len(drop)}",f'{H}/review/dk_dropped.png')
# 4. aplica efectiv doar cu --apply
if APPLY:
    dd=f'{DIR}.dropped'; os.makedirs(dd,exist_ok=True)
    man=f'{DIR}/manifest.csv'; keepset=set(os.path.basename(f) for f in keep)
    for f in drop: shutil.move(f,f'{dd}/{os.path.basename(f)}')
    if os.path.exists(man):
        lines=open(man).read().splitlines(); hdr=lines[0]
        kept=[l for l in lines[1:] if os.path.basename(l.split(',')[0]) in keepset]
        open(man,'w').write(hdr+'\n'+'\n'.join(kept)+'\n')
    print(f"APLICAT: {len(drop)} mutati in {dd}, manifest actualizat ({len(keep)} ramasi)")
else:
    print("\n(dry-run: nimic mutat. Ruleaza cu --apply dupa OK-ul lui Andrei.)")
