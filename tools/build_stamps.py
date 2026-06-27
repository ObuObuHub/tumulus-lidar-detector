import math,subprocess,os,csv,json
from PIL import Image, ImageDraw

SVC="https://tiles.arcgis.com/tiles/Q2Kmg0bQDn3rySgn/arcgis/rest/services/Banat_3_5_H_tif/MapServer/tile"
R=6378137.0; C=2*math.pi*R; ORIG=-20037508.342787; ORIGY=20037508.342787
Z=17; STAMP=256; HALF=STAMP//2
REPO=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PILOT=f"{REPO}/timis_pilot"
STDIR=f"{PILOT}/stamps"; TILEDIR="/tmp/banat_tiles"
os.makedirs(STDIR,exist_ok=True); os.makedirs(TILEDIR,exist_ok=True)

def merc(lon,lat):
    return R*math.radians(lon), R*math.log(math.tan(math.pi/4+math.radians(lat)/2))
res=C/(256*2**Z)  # m/px at this level

_tilecache={}
def get_tile(col,row):
    key=(col,row)
    if key in _tilecache: return _tilecache[key]
    fn=f"{TILEDIR}/{Z}_{col}_{row}.png"
    if not os.path.exists(fn):
        subprocess.run(["curl","-s","--max-time","25","-o",fn,f"{SVC}/{Z}/{row}/{col}"],check=False)
    try: im=Image.open(fn).convert('L')
    except Exception: im=None
    _tilecache[key]=im
    return im

rows=list(csv.DictReader(open(f"{REPO}/timis_tumuli.csv")))
manifest=[]
def slug(s):
    return ''.join(c if c.isalnum() else '_' for c in (s or 'NA')).strip('_')[:24]

for i,r in enumerate(rows):
    lon,lat=float(r['lon']),float(r['lat'])
    x,y=merc(lon,lat)
    gx=(x-ORIG)/res; gy=(ORIGY-y)/res            # global pixel coords at level Z
    x0=gx-HALF; y0=gy-HALF
    canvas=Image.new('L',(STAMP,STAMP),0)
    c0=int(math.floor(x0/256)); c1=int(math.floor((x0+STAMP)/256))
    r0=int(math.floor(y0/256)); r1=int(math.floor((y0+STAMP)/256))
    for col in range(c0,c1+1):
        for row in range(r0,r1+1):
            t=get_tile(col,row)
            if t is None: continue
            px=col*256-int(x0); py=row*256-int(y0)
            canvas.paste(t,(px,py))
    oid=r.get('oid'); loc=slug(r.get('loc'))
    name=f"tumul_{oid}_{loc}.png"
    canvas.save(f"{STDIR}/{name}")
    manifest.append({'oid':oid,'loc':r.get('loc'),'cod':r.get('cod'),'nume':r.get('nume'),
                     'lon':round(lon,6),'lat':round(lat,6),'stamp':f"stamps/{name}",
                     'z':Z,'ground_res_m':round(res*math.cos(math.radians(lat)),2),'stamp_m':round(STAMP*res*math.cos(math.radians(lat)))})

with open(f"{PILOT}/manifest.csv","w",newline='') as fh:
    w=csv.DictWriter(fh,fieldnames=list(manifest[0].keys())); w.writeheader(); w.writerows(manifest)

# contact sheet: grid of all stamps (downscaled) with center crosshair
n=len(manifest); cols=12; rows_n=math.ceil(n/cols); cell=132
sheet=Image.new('RGB',(cols*cell,rows_n*cell),(20,20,20)); dr=ImageDraw.Draw(sheet)
for idx,m in enumerate(manifest):
    st=Image.open(f"{PILOT}/{m['stamp']}").convert('RGB').resize((128,128))
    cx=(idx%cols)*cell+2; cy=(idx//cols)*cell+2
    sheet.paste(st,(cx,cy))
    # center crosshair (tumulus location)
    mx,my=cx+64,cy+64
    dr.line([(mx-7,my),(mx+7,my)],fill=(255,40,40),width=1); dr.line([(mx,my-7),(mx,my+7)],fill=(255,40,40),width=1)
sheet.save(f"{PILOT}/contact_sheet.png")
print("stamps:",n,"-> ",STDIR)
print("unique tiles fetched:",len(_tilecache))
print("ground res ~%.2f m/px, stamp ~%dm"%(res*math.cos(math.radians(45.8)), STAMP*res*math.cos(math.radians(45.8))))
print("contact sheet:",f"{PILOT}/contact_sheet.png")
