#!/usr/bin/env python3
# build_compare.py — tabel side-by-side: modelul nostru vs cele mai bune din literatură. -> review/compare_lit.png
from PIL import Image,ImageDraw,ImageFont
H="/Users/crustaceu/lidar-match"
def F(sz,b=True):
    p='/System/Library/Fonts/Supplemental/Arial Bold.ttf' if b else '/System/Library/Fonts/Supplemental/Arial.ttf'
    try:return ImageFont.truetype(p,sz)
    except:return ImageFont.load_default()
cols=["",  "MODELUL NOSTRU\nCNN 0.5m + curbură", "Németh & Benedek 2020\n(MPP, Alsacia)", "Niculiță 2020\n(RF geomorfometrie, RO)"]
rows=[
 ("Metodă","CNN mic + cascadă\nfiltru curbură","Marked Point Process\n(geometrie stocastică)","Random Forest +\ngeomorfometrie"),
 ("Date / rezoluție","LiDAR 0.5 m (Oltenia)","DTM 1 m","LiDAR DEM 0.5 m\n(Pod. Moldovenesc)"),
 ("Mod de testare","scanare OARBĂ\n47 km² (held-out)","53 tumuli\n(regiune curată)","candidați peak\n(regiune)"),
 ("Recall","100%  (10/10)","85%","93%"),
 ("Precizie","37–42% @recall100","83%","— (FPR 0.004)"),
 ("F1","0.54 (scan oarbă)","0.84 (regiune)","—"),
 ("Fals-pozitive","17 (0.36 / km²)","~9","FPR 0.004"),
 ("AUROC discriminare","0.998","—","—"),
]
# highlight: green where we lead, amber where lit leads
hl={ (3,1):'g',(3,2):'r',(3,3):'r',  # recall: we lead
     (4,1):'r',(4,2):'g',            # precision: lit leads
     (5,1):'r',(5,2):'g',
     (7,1):'g'}                      # AUROC: only us
cw=[210,250,250,250];rh=58;hh=66;pad=14
W=sum(cw)+pad*2;Hh=hh+rh*len(rows)+pad*2+150
img=Image.new('RGB',(W,Hh),(248,248,250));d=ImageDraw.Draw(img)
ft=F(17);fth=F(16);ftt=F(24);ftn=F(14,False)
d.text((pad,12),"Detector tumuli LiDAR — comparație cu literatura",fill=(20,20,30),font=ftt)
y0=46+pad
# header
x=pad
for j,c in enumerate(cols):
    d.rectangle([x,y0,x+cw[j],y0+hh],fill=(38,48,72))
    for k,ln in enumerate(c.split("\n")):
        d.text((x+8,y0+10+k*20),ln,fill=(255,255,255),font=fth)
    x+=cw[j]
# rows
for i,row in enumerate(rows):
    yy=y0+hh+i*rh;bg=(238,240,246) if i%2 else (248,248,250)
    d.rectangle([pad,yy,pad+sum(cw),yy+rh],fill=bg)
    x=pad
    for j,cell in enumerate(row):
        col=(25,25,30)
        if (i,j) in hl:
            c=hl[(i,j)]
            d.rectangle([x+2,yy+2,x+cw[j]-2,yy+rh-2],fill=(214,242,214) if c=='g' else (250,230,214))
            col=(20,110,40) if c=='g' else (150,80,10)
        for k,ln in enumerate(cell.split("\n")):
            d.text((x+8,yy+8+k*19),ln,fill=col,font=ft if j==0 else (F(16) if (i,j) in hl else F(15,False)))
        x+=cw[j]
# grid lines
x=pad
for j in range(len(cols)+1):
    d.line([x,y0,x,y0+hh+rh*len(rows)],fill=(200,200,205),width=1);x+=cw[j] if j<len(cols) else 0
# footnote
fy=y0+hh+rh*len(rows)+16
notes=["NOTĂ onestă: F1-ul mare din literatură (0.84) e pe REGIUNI CURATE. La căutare pe PEISAJ ÎNTREG (orb), F1 scade mult în",
 "toată literatura — ex. căutarea de fortificații pe toată Anglia: F1 ≈ 0.36. Cifra noastră 0.54 e pe scanare OARBĂ 47 km².",
 "→ Conducem la RECALL (100%) și rezoluție; filtrul de curbură ne-a dublat precizia (17%→37%). Pe precizie pură rămânem",
 "sub MPP-ul curat, dar pe regim de scanare completă (cazul real de teren) suntem competitivi/peste."]
for k,n in enumerate(notes): d.text((pad,fy+k*22),n,fill=(70,70,80),font=ftn)
img.save(f"{H}/review/compare_lit.png")
print(f"-> review/compare_lit.png {img.size}")
