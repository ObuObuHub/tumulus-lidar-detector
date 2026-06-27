import csv,sys
import numpy as np
from PIL import Image
# detect_board_marks.py [marked_img] [map_csv] [out_json] — roșu/verde per celulă, grilă scalată la imaginea primită
IMG    = sys.argv[1] if len(sys.argv)>1 else "/Users/crustaceu/.claude/channels/telegram/inbox/1782092569493-AQADYQ1rG6u1yFF-.jpg"
MAPCSV = sys.argv[2] if len(sys.argv)>2 else '/tmp/dolj_roundfp_map.csv'
OUTJSON= sys.argv[3] if len(sys.argv)>3 else '/tmp/dolj_marks.json'
# original board geometry
COLS,ROWS,CELL,HDR=9,6,156,26
OW,OH=COLS*CELL, ROWS*CELL+HDR   # 1404 x 962
im=Image.open(IMG).convert('RGB'); W,Hh=im.size
print(f"received image {W}x{Hh} (orig board {OW}x{OH})")
a=np.asarray(im).astype(int)
R,G,B=a[:,:,0],a[:,:,1],a[:,:,2]
red = (R>110)&(R-G>45)&(R-B>45)
grn = (G>90)&(G-R>35)&(G-B>35)
yel = (R>120)&(G>120)&(R-B>55)&(G-B>55)   # galben = R+G mari, B mic (de investigat)
sx,sy=W/OW, Hh/OH
cw,ch=CELL*sx, CELL*sy; hdr=HDR*sy
mapc={int(r['idx']):(r['lon'],r['lat'],r['score']) for r in csv.DictReader(open(MAPCSV))}
reds=[];greens=[];yellows=[];blank=[]
for i in range(1,COLS*ROWS+1):
    c=(i-1)%COLS; r=(i-1)//COLS
    x0=int(c*cw); x1=int((c+1)*cw); y0=int(hdr+r*ch); y1=int(hdr+(r+1)*ch)
    # shrink slightly to avoid border bleed between cells
    mx=int((x1-x0)*0.06); my=int((y1-y0)*0.06)
    rc=red[y0+my:y1-my, x0+mx:x1-mx]; gc=grn[y0+my:y1-my, x0+mx:x1-mx]; yc=yel[y0+my:y1-my, x0+mx:x1-mx]
    area=rc.size
    rf=rc.sum()/area; gf=gc.sum()/area; yf=yc.sum()/area
    # convenție: verde=tumul(poz), galben=investighează, roșu=FP, nemarcat=negativ
    if gf>0.012 and gf>=yf and gf>=rf: greens.append(i)
    elif yf>0.012 and yf>=rf: yellows.append(i)
    elif rf>0.012: reds.append(i)
    else: blank.append((i,rf,gf,yf))
print(f"\nVERDE (tumul → POZITIV): {len(greens)}\n  {greens}")
for i in greens:
    lo,la,s=mapc[i]; print(f"   #{i}: {la},{lo}  https://maps.google.com/?q={la},{lo}")
print(f"\nGALBEN (de investigat satelit): {len(yellows)}\n  {yellows}")
for i in yellows:
    lo,la,s=mapc[i]; print(f"   #{i}: {la},{lo}  https://maps.google.com/?q={la},{lo}")
if reds: print(f"\nROȘU (FP explicit): {len(reds)}\n  {reds}")
print(f"\nNEMARCATE (→ NEGATIVE): {len(blank)}\n  {[b[0] for b in blank]}")
# save
with open(OUTJSON,'w') as f:
    import json; json.dump({'red':reds,'green':greens,'yellow':yellows,'blank':[b[0] for b in blank]},f)
print(f"\n-> {OUTJSON}")
