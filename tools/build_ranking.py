#!/usr/bin/env python3
# build_ranking.py — ranking modele de detecție tumuli/movile din LiDAR (literatură) + modelul nostru. -> review/ranking.png
from PIL import Image,ImageDraw,ImageFont
H="/Users/crustaceu/lidar-match"
def F(sz,b=True):
    p='/System/Library/Fonts/Supplemental/Arial Bold.ttf' if b else '/System/Library/Fonts/Supplemental/Arial.ttf'
    try:return ImageFont.truetype(p,sz)
    except:return ImageFont.load_default()
# (rank, model, metoda, date/regiune, performanta, nota)  — OURS marcat
rows=[
 ("1","Yang 2025\n(npj Herit. Sci.)","TransUNet / U-Net,\n5 canale (DEM+pantă+\nhillshade+rugoz.+CURBURĂ)","LiDAR, cimitir\nsub pădure","P 0.92 / R 0.94\nIoU 0.90","SOTA pe metrici; segmentare pe zonă\nCURATĂ, nu scanare de peisaj. Validează\ncurbura+multi-canal.",0),
 ("2","Niculiță 2020\n(Sensors)","Random Forest +\ngeomorfometrie","LiDAR 0.5m,\nPod. Moldovenesc (RO)","R 0.93 /\nFPR 0.004","Reproductibil (Zenodo), tumuli, RO,\nsemi-automat. Cel mai solid comparabil.",0),
 ("3","Németh & Benedek\n2020 (ISPRS)","Marked Point Process\n(NEsupervizat)","DTM 1m,\nAlsacia / Ungaria","P 0.83 / R 0.85\nF1 0.84","Fără date de antrenare → generalizează.\nElegant, peer-reviewed.",0),
 ("4","Berganzo-Besga\n2021 (Rem. Sens.)","YOLOv3 (LiDAR MSRM)\n+ RF (Sentinel-2)","NW Iberia","P 0.97 / R 0.64\nAP 0.67","10.527 tumuli pe 30.000 km² =\nCEA MAI MARE scară reală.",0),
 ("5","Guyot et al. 2018\n(Rem. Sens.)","Multi-scară (poziție\ntopografică) + RF","LiDAR, Carnac (FR)","F1 0.72\n(kappa 0.98 subset)","Fundamental — a deschis domeniul.",0),
 ("6","Verschoof-vd Vaart\n& Lambers 2019","Faster R-CNN „WODAN\"","LiDAR, Veluwe (NL),\ntumuli","F1 0.79 (test mic)\n~50% (peisaj)","Raportare ONESTĂ a căderii la scară\nmare. Tumuli NL.",0),
 ("7","NOI (Chiper/Aurel\n2026)","CNN mic + filtru\ncurbură SCALE-ADAPTIV","LiDAR 0.5m, RO\n(+transfer NL)","R 100% / AUROC\n0.998 / prec. 42%*","Recall-max + cross-țară + invariant la\nrezoluție + validare OARBĂ. NEpublicat;\nprecizie sub lideri (*scanare completă).",1),
 ("8","Trier et al.\n2019/2021","R-CNN multi-clasă","LiDAR, Norvegia","75% R / 24% FP\n(test)","Multi-clasă; cade mult la scară de peisaj.",0),
 ("9","Caspari & Crespo\n2019 (JAS)","CNN","„Princely tombs\",\noptic/satelit","CNN > alți algor.","Scop mai îngust; mixt optic.",0),
 ("10","Davis et al. 2020\n(PNAS)","ML multi-senzor","Satelit multitemporal\n(SE SUA)","—","Modalitate diferită (nu LiDAR pur).",0),
]
ftt=F(23);fth=F(14);ft=F(13,False);ftb=F(13)
colw=[40,150,210,165,150,330];rh=78;hh=40;pad=12
W=sum(colw)+pad*2;Hh=hh+rh*len(rows)+pad*2+78
img=Image.new('RGB',(W,Hh),(248,248,250));d=ImageDraw.Draw(img)
d.text((pad,10),"Ranking modele de detecție tumuli/movile din LiDAR (literatură + al nostru)",fill=(20,20,30),font=ftt)
y0=44+pad
heads=["#","Model / an","Metodă","Date / regiune","Performanță","Notă"]
x=pad
for j,hh_t in enumerate(heads):
    d.rectangle([x,y0,x+colw[j],y0+hh],fill=(38,48,72));d.text((x+6,y0+11),hh_t,fill=(255,255,255),font=fth);x+=colw[j]
for i,row in enumerate(rows):
    yy=y0+hh+i*rh;ours=row[6]==1
    bg=(214,238,255) if ours else ((240,242,247) if i%2 else (250,250,252))
    d.rectangle([pad,yy,pad+sum(colw),yy+rh],fill=bg)
    if ours: d.rectangle([pad,yy,pad+sum(colw),yy+rh],outline=(30,90,200),width=2)
    x=pad
    for j in range(6):
        cell=row[j];col=(20,40,90) if ours else (30,30,35)
        fnt=ftb if (j==0 or ours) else (ft if j!=1 else ftb)
        for k,ln in enumerate(cell.split("\n")):
            d.text((x+6,yy+6+k*15),ln,fill=col,font=fnt)
        x+=colw[j]
fy=y0+hh+rh*len(rows)+10
notes=["Ranking pe MERIT GLOBAL (performanță + rigoare + scară de validare + reproductibilitate + relevanță tumuli + maturitate). ⚠ NU e apples-to-apples —",
 "fiecare e testat pe alt set/regiune/scară. Tendință clară: pe zone CURATE F1≈0.8–0.94; la scanare de PEISAJ ÎNTREG toate cad (WODAN ~50%, fortificații ~0.36)."]
for k,n in enumerate(notes): d.text((pad,fy+k*18),n,fill=(70,70,80),font=F(12,False))
img.save(f"{H}/review/ranking.png");print(f"-> review/ranking.png {img.size}")
