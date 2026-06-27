import math,subprocess,os,csv
from PIL import Image
R=6378137.0; C=2*math.pi*R; ORIG=-20037508.342787; ORIGY=20037508.342787
REPO=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); DS=f"{REPO}/dataset"
for d in ['positives','positives_base','positives_aug']:
    os.makedirs(f"{DS}/{d}",exist_ok=True)
TILEDIR={'banat':'/tmp/banat_tiles','hegyi':'/tmp/hegyi_tiles'}
def merc(lon,lat): return R*math.radians(lon), R*math.log(math.tan(math.pi/4+math.radians(lat)/2))
_c={}
def tile(svc,org,z,col,row,cachedir):
    k=(svc,z,col,row)
    if k in _c: return _c[k]
    fn=f"{cachedir}/{svc}_{z}_{col}_{row}.png" if cachedir=='/tmp/hegyi_tiles' else f"{cachedir}/{z}_{col}_{row}.png"
    if not os.path.exists(fn):
        subprocess.run(["curl","-s","--max-time","25","-o",fn,
            f"https://tiles.arcgis.com/tiles/{org}/arcgis/rest/services/{svc}/MapServer/tile/{z}/{row}/{col}"],check=False)
    try: im=Image.open(fn).convert('L')
    except Exception: im=None
    _c[k]=im; return im
def window(lon,lat,svc,org,z,win,cachedir):
    x,y=merc(lon,lat); res=C/(256*2**z); gx=(x-ORIG)/res; gy=(ORIGY-y)/res
    x0=gx-win//2; y0=gy-win//2; cv=Image.new('L',(win,win),0)
    for col in range(int(math.floor(x0/256)),int(math.floor((x0+win)/256))+1):
        for rw in range(int(math.floor(y0/256)),int(math.floor((y0+win)/256))+1):
            t=tile(svc,org,z,col,rw,cachedir)
            if t: cv.paste(t,(col*256-int(x0),rw*256-int(y0)))
    return cv
WIN=384
pos=[]
# 1) WESTERN clear (already have base384)
vp=f"{REPO}/vest_pilot"
for m in csv.DictReader(open(f"{vp}/manifest.csv")):
    if float(m['black'])<=0.4 and float(m['mound'])>=10:
        src=f"{vp}/{m['base']}"; name=f"W_{os.path.basename(m['base'])}"
        Image.open(src).save(f"{DS}/positives_base/{name}")
        pos.append({'src':'vest','jud':m['jud'],'res':m['res'],'lon':m['lon'],'lat':m['lat'],'base':name})
# 2) TIMIS clear top-15 (rebuild base384 from Banat z17, cached)
tp=f"{REPO}/timis_pilot"
for m in csv.DictReader(open(f"{tp}/manifest_ranked.csv")):
    if int(m['rank'])<=15:
        cv=window(float(m['lon']),float(m['lat']),"Banat_3_5_H_tif","Q2Kmg0bQDn3rySgn",17,WIN,'/tmp/banat_tiles')
        name=f"T_{m['oid']}_{(m['loc'] or 'NA').replace(' ','_')[:16]}.png"
        cv.save(f"{DS}/positives_base/{name}")
        pos.append({'src':'timis','jud':'Timiș','res':'3m','lon':m['lon'],'lat':m['lat'],'base':name})
# center 256 crop for each positive
s=(WIN-256)//2
for p in pos:
    Image.open(f"{DS}/positives_base/{p['base']}").crop((s,s,s+256,s+256)).save(f"{DS}/positives/{p['base']}")
# 3) AUGMENTATION: 8 rotations x 2 flips, rotate 384 then center-crop 256 (no black corners)
angles=[0,45,90,135,180,225,270,315]
naug=0
for p in pos:
    base=Image.open(f"{DS}/positives_base/{p['base']}").convert('L')
    stem=p['base'][:-4]
    for flip in (False,True):
        im0=base.transpose(Image.FLIP_LEFT_RIGHT) if flip else base
        for a in angles:
            rot=im0.rotate(a,resample=Image.BICUBIC,expand=False)
            crop=rot.crop((s,s,s+256,s+256))
            crop.save(f"{DS}/positives_aug/{stem}_f{int(flip)}_r{a}.png"); naug+=1
with open(f"{DS}/manifest.csv","w",newline='') as fh:
    w=csv.DictWriter(fh,fieldnames=list(pos[0].keys())); w.writeheader(); w.writerows(pos)
print("CLEAR positives:",len(pos)," (west:",sum(1 for p in pos if p['src']=='vest'),", timis:",sum(1 for p in pos if p['src']=='timis'),")")
from collections import Counter
print("by county:",Counter(p['jud'] for p in pos))
print("augmented images:",naug,"(x%d each)"%(len(angles)*2))
