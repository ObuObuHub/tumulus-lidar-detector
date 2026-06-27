import sys,math,subprocess,os,csv
import numpy as np
from PIL import Image,ImageFilter,ImageOps,ImageDraw
LON=float(sys.argv[1]); LAT=float(sys.argv[2]); SVC=sys.argv[3]; ORG=sys.argv[4]; Z=int(sys.argv[5]); WIN=int(sys.argv[6]); OUT=sys.argv[7]
R=6378137.0;C=2*math.pi*R;ORIG=-20037508.342787;ORIGY=20037508.342787
res=C/(256*2**Z)
xC=R*math.radians(LON); yC=R*math.log(math.tan(math.pi/4+math.radians(LAT)/2))
x0=(xC-ORIG)/res-WIN//2; y0=(ORIGY-yC)/res-WIN//2
def px_lonlat(ix,iy):
    X=ORIG+(x0+ix)*res; Y=ORIGY-(y0+iy)*res
    return (X/R)*180/math.pi,(2*math.atan(math.exp(Y/R))-math.pi/2)*180/math.pi
os.makedirs('/tmp/hegyi_tiles',exist_ok=True)
canvas=Image.new('L',(WIN,WIN),0)
for col in range(int(x0//256),int((x0+WIN)//256)+1):
    for row in range(int(y0//256),int((y0+WIN)//256)+1):
        fn=f'/tmp/hegyi_tiles/{SVC}_{Z}_{col}_{row}.png'
        if not os.path.exists(fn):
            subprocess.run(['curl','-s','--max-time','25','-o',fn,f'https://tiles.arcgis.com/tiles/{ORG}/arcgis/rest/services/{SVC}/MapServer/tile/{Z}/{row}/{col}'],check=False)
        try: canvas.paste(Image.open(fn).convert('L'),(col*256-int(x0),row*256-int(y0)))
        except: pass
arr=np.asarray(canvas,dtype=np.float32)
# SLRM local relief (display) — reveals subtle mounds
blur=np.asarray(canvas.filter(ImageFilter.GaussianBlur(30)),dtype=np.float32)
relief=arr-blur
lo,hi=np.percentile(relief,2),np.percentile(relief,98)
slrm=Image.fromarray(np.clip((relief-lo)/(hi-lo+1e-6)*255,0,255).astype('uint8'))
# DETECTOR: abs multi-scale DoG (catch bright+dark domes), elongation + isolation filters
def g(s): return np.asarray(canvas.filter(ImageFilter.GaussianBlur(s)),dtype=np.float32)
resp=None
for s1,s2 in [(3,8),(6,14)]:
    r=np.abs(g(s1)-g(s2)); r=(r-r.mean())/(r.std()+1e-6)
    resp=r if resp is None else np.maximum(resp,r)
rng=float(resp.max()-resp.min()) or 1
respImg=Image.fromarray(np.clip((resp-resp.min())/rng*255,0,255).astype('uint8'))
rs=np.asarray(respImg,dtype=np.float32)
mx=np.asarray(respImg.filter(ImageFilter.MaxFilter(25)),dtype=np.float32)
thr=rs.mean()+1.9*rs.std()
peaks=np.argwhere((rs==mx)&(rs>thr))
rel=np.asarray(slrm,dtype=np.float32)
def elong(y,x,h=16):
    if y-h<0 or y+h>=WIN or x-h<0 or x+h>=WIN: return 99
    p=rel[y-h:y+h+1,x-h:x+h+1]; m=p.mean()
    yy,xx=np.where(np.abs(p-m)>0.6*p.std())
    if len(xx)<10: return 99
    cov=np.cov(np.vstack([xx-h,yy-h])); ev=np.clip(np.linalg.eigvalsh(cov),1e-3,None)
    return (ev[1]/ev[0])**0.5
kept=[]
for y,x in peaks:
    if not(16<=x<WIN-16 and 16<=y<WIN-16): continue
    if any((y-yy)**2+(x-xx)**2<22*22 for yy,xx in kept): continue
    if elong(y,x)<2.5: kept.append((int(y),int(x)))
disp=slrm.convert('RGB'); dr=ImageDraw.Draw(disp)
rows=[]
for i,(y,x) in enumerate(kept,1):
    dr.ellipse([x-12,y-12,x+12,y+12],outline=(40,210,40),width=2)
    dr.rectangle([x+9,y-13,x+9+7*len(str(i)),y-2],fill=(0,0,0)); dr.text((x+10,y-13),str(i),fill=(255,255,0))
    lon,lat=px_lonlat(x,y); rows.append({'id':i,'px':x,'py':y,'lon':round(lon,6),'lat':round(lat,6)})
disp.save(OUT)
with open(OUT.replace('.png','_candidates.csv'),'w',newline='') as fh:
    w=csv.DictWriter(fh,fieldnames=['id','px','py','lon','lat']); w.writeheader(); w.writerows(rows)
print(f"{OUT}: {len(kept)} candidates | SLRM display | {WIN*res*math.cos(math.radians(LAT)):.0f}m")
