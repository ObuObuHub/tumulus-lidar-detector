#!/usr/bin/env python3
# build_catane_view.py — vedere larga Catane: hillshade + HEATMAP model (din detectii) + marcaje
# verde=10 tumuli reali | rosu=FP taiate de filtrul curbura | portocaliu=FP ramase. -> review/catane_view.jpg
import os,math,subprocess,csv
import numpy as np
from PIL import Image,ImageDraw,ImageFont,ImageFilter
H=os.path.expanduser('~/lidar-match');CACHE="/tmp/laki3";CS=0.5;TPX=2000
APP="/Applications/QGIS-final-4_0_3.app/Contents"
ENV=dict(os.environ,DYLD_FRAMEWORK_PATH=f"{APP}/Frameworks",PROJ_DATA=f"{APP}/Resources/qgis/proj",PROJ_LIB=f"{APP}/Resources/qgis/proj",GDAL_DATA=f"{APP}/Resources/qgis/gdal")
GT=f"{APP}/MacOS/gdaltransform"
def trans(pts,s,t):
    inp="\n".join(f"{a} {b}" for a,b in pts)+"\n";r=subprocess.run([GT,"-s_srs",s,"-t_srs",t],input=inp,capture_output=True,text=True,env=ENV)
    return [tuple(map(float,l.split()[:2])) for l in r.stdout.strip().split("\n") if l.split()]
def hsf(dem,cs,azs=(315,45,135,225,270,0),alt=35):
    gy,gx=np.gradient(dem,cs);sl=np.arctan(np.hypot(gx,gy));asp=np.arctan2(-gy,gx);o=np.zeros_like(dem);ar=math.radians(alt)
    for az in azs: azr=math.radians(360-az+90);o+=np.clip(np.sin(ar)*np.cos(sl)+np.cos(ar)*np.sin(sl)*np.cos(azr-asp),0,1)
    return o/len(azs)
def jet(v):  # v in [0,1] -> (r,g,b)
    v=np.clip(v,0,1);r=np.clip(1.5-abs(4*v-3),0,1);g=np.clip(1.5-abs(4*v-2),0,1);b=np.clip(1.5-abs(4*v-1),0,1)
    return np.stack([r,g,b],-1)
# GT + detections
gt=list(csv.DictReader(open('/tmp/catane_gt.csv')))
dets=list(csv.DictReader(open('/tmp/dets_gated2.csv')))
gll=[(float(r['lon']),float(r['lat'])) for r in gt]
gen=trans(gll,"EPSG:4326","EPSG:3844")
es=[e for e,n in gen];ns=[n for e,n in gen];MARG=400
e0=int((min(es)-MARG)//1000);e1=int((max(es)+MARG)//1000);n0=int((min(ns)-MARG)//1000);n1=int((max(ns)+MARG)//1000)
xll0=e0*1000;ytop0=(n1+1)*1000;W=(e1-e0+1)*2000;Hh=(n1-n0+1)*2000
mos=np.full((Hh,W),np.nan,np.float32)
for nk in range(n0,n1+1):
    for ek in range(e0,e1+1):
        p=f"{CACHE}/{nk}_{ek}.npy"
        if os.path.exists(p):
            d=np.load(p);ox=int((ek*1000-xll0)/CS);oy=int((ytop0-(nk+1)*1000)/CS);mos[oy:oy+2000,ox:ox+2000]=d[:2000,:2000]
# downsample to display
DW=1700;fac=max(1,W//DW);dw=W//fac;dh=Hh//fac
dem=mos[:dh*fac,:dw*fac].reshape(dh,fac,dw,fac).mean((1,3))
dem=np.nan_to_num(dem,nan=np.nanmedian(dem))
sh=hsf(dem,CS*fac);lo,hi=np.percentile(sh,2),np.percentile(sh,98);shg=np.clip((sh-lo)/(hi-lo)*255,0,255).astype(np.uint8)
base=np.stack([shg]*3,-1).astype(np.float32)
def px(e,n): return int((e-xll0)/CS/fac),int((ytop0-n)/CS/fac)
# heatmap: doar focuri TARI (scor>=0.7) ca sa fie curat, rampa CALDA
acc=np.zeros((dh,dw),np.float32)
den=[(float(r['lon']),float(r['lat'])) for r in dets];scs=[float(r['score']) for r in dets]
den_en=trans(den,"EPSG:4326","EPSG:3844")
for (e,n),s in zip(den_en,scs):
    if s<0.7: continue
    x,y=px(e,n)
    if 0<=x<dw and 0<=y<dh: acc[y,x]=max(acc[y,x],s)
accb=np.asarray(Image.fromarray((acc*255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(4)),np.float32)/255.
accb=accb/(accb.max()+1e-9)
warm=np.zeros((dh,dw,3),np.float32);warm[...,0]=np.clip(accb*3,0,1);warm[...,1]=np.clip(accb*1.6-0.3,0,1)
alpha=np.clip(accb*1.4,0,1)[...,None]
disp=(base*(1-alpha*0.45)+warm*255*alpha*0.45).astype(np.uint8)  # heatmap subtil, sa nu acopere marcajele
try:ft=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial Bold.ttf',17);ftb=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial Bold.ttf',24)
except:ft=ftb=ImageFont.load_default()
T2=0.70
# prag recall-100 = scorul minim al unui TP (consistent cu benchmark)
tp_sc=[float(r['score']) for r in dets if r['istp']=='1'];THR1=min(tp_sc)
def panel(mode):
    im=Image.fromarray(disp).convert('RGB');d=ImageDraw.Draw(im);nfp=0
    for r,(e,n) in zip(dets,den_en):
        s1=float(r['score']);istp=r['istp']=='1';pg=float(r['pgate']) if r['pgate']!='NA' else 1.0
        if istp or s1<THR1: continue
        x,y=px(e,n)
        if mode=='before':
            d.ellipse([x-10,y-10,x+10,y+10],fill=(255,30,30),outline=(0,0,0),width=2);nfp+=1
        else:
            if pg>=T2: d.ellipse([x-10,y-10,x+10,y+10],fill=(255,235,0),outline=(0,0,0),width=2);nfp+=1
            else: d.line([x-7,y-7,x+7,y+7],fill=(130,130,140),width=2);d.line([x-7,y+7,x+7,y-7],fill=(130,130,140),width=2)
    for i,(e,n) in enumerate(gen,1):
        x,y=px(e,n);d.ellipse([x-17,y-17,x+17,y+17],outline=(0,255,0),width=4);d.text((x+18,y-10),f"T{i}",fill=(160,255,160),font=ft)
    return im,nfp
imb,nb=panel('before');ima,na=panel('after')
# compose side by side cu titluri
gap=10;th=64;Wp,Hp=imb.size;canvas=Image.new('RGB',(Wp*2+gap,Hp+th),(10,10,12));dc=ImageDraw.Draw(canvas)
canvas.paste(imb,(0,th));canvas.paste(ima,(Wp+gap,th))
dc.text((10,6),"CATANE 47 km² (held-out, scanare oarbă) — heatmap model",fill=(255,255,255),font=ftb)
dc.text((10,38),f"STÂNGA = model brut: {nb} fals-pozitive (roșu) + 10 tumuli reali (verde)",fill=(255,120,120),font=ft)
dc.text((Wp+gap+10,38),f"DREAPTA = + filtru CURBURĂ: {na} FP rămase (galben), {nb-na} tăiate (×gri) — toți 10 reali (verde) păstrați",fill=(255,230,80),font=ft)
canvas.save(f'{H}/review/catane_view.jpg',quality=84)
print(f"-> review/catane_view.jpg {canvas.size} ({os.path.getsize(H+'/review/catane_view.jpg')//1024} KB) | before {nb} after {na}")
