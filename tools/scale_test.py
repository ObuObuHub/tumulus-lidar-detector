import os,sys,math,subprocess,csv
import numpy as np
H=os.path.expanduser('~/lidar-match');CACHE="/tmp/laki3";CS=0.5;TPX=2000
APP="/Applications/QGIS-final-4_0_3.app/Contents"
ENV=dict(os.environ,DYLD_FRAMEWORK_PATH=f"{APP}/Frameworks",PROJ_DATA=f"{APP}/Resources/qgis/proj",PROJ_LIB=f"{APP}/Resources/qgis/proj",GDAL_DATA=f"{APP}/Resources/qgis/gdal")
GT=f"{APP}/MacOS/gdaltransform"
def trans(p,s,t):
    r=subprocess.run([GT,"-s_srs",s,"-t_srs",t],input=f"{p[0]} {p[1]}\n",capture_output=True,text=True,env=ENV);a,b,_=r.stdout.split();return float(a),float(b)
def dl(nk,ek):
    p=f"{CACHE}/{nk}_{ek}.npy";return np.load(p) if os.path.exists(p) else None
def boxblur(a,r):
    ii=np.zeros((a.shape[0]+1,a.shape[1]+1));ii[1:,1:]=np.cumsum(np.cumsum(a,0),1)
    H2,W2=a.shape;ys=np.arange(H2);xs=np.arange(W2)
    y0=np.clip(ys-r,0,H2);y1=np.clip(ys+r+1,0,H2);x0=np.clip(xs-r,0,W2);x1=np.clip(xs+r+1,0,W2)
    return (ii[y1][:,x1]-ii[y0][:,x1]-ii[y1][:,x0]+ii[y0][:,x0])/((y1-y0)[:,None]*(x1-x0)[None,:])

PH=int(120/CS)        # 120 m half-window -> 240px patch radius
R=int(60/CS)          # baseline blur 60 m (dome ~30-40m nu e absorbit)
CENT=int(40/CS)       # central search 40 m
SNAP=int(40/CS)       # snap apex within 40 m = window half-width

def measure(lon,lat):
    e,n=trans((lon,lat),"EPSG:4326","EPSG:3844")
    nk=int(n//1000);ek=int(e//1000)
    blk=np.full((3*TPX,3*TPX),np.nan,np.float32)
    for dn in (-1,0,1):
        for de in (-1,0,1):
            d=dl(nk+dn,ek+de)
            if d is None: continue
            blk[(1-dn)*TPX:(1-dn)*TPX+TPX,(de+1)*TPX:(de+1)*TPX+TPX]=d[:TPX,:TPX]
    cx=int((e-ek*1000)/CS)+TPX; cy=int(((nk+1)*1000-n)/CS)+TPX
    w=blk[cy-PH:cy+PH, cx-PH:cx+PH]
    if w.shape!=(2*PH,2*PH) or np.isnan(w).mean()>0.1: return None
    w=np.nan_to_num(w,nan=np.nanmedian(w))
    slrm=w-boxblur(w,R)                 # signed local relief, meters
    c0=PH-CENT;c1=PH+CENT
    cen=slrm[c0:c1,c0:c1]
    # snap apex within 10 m of given point
    s0=PH-SNAP;s1=PH+SNAP
    sn=slrm[s0:s1,s0:s1]
    ay,ax=np.unravel_index(np.argmax(sn),sn.shape); apx=s0+ax; apy=s0+ay
    h_pos=float(slrm[apy,apx])          # dome height at snapped apex (m)
    h_neg=float(cen.min())              # pit depth in central (m, negative)
    # diameter: connected-ish region above half-max around apex (area->equiv diam)
    if h_pos>0.05:
        mask=slrm[c0:c1,c0:c1]>h_pos*0.5
        area=mask.sum()*(CS*CS)
        diam=2*math.sqrt(area/math.pi)
    else:
        diam=0.0
    return dict(h_pos=h_pos,h_neg=h_neg,diam=diam)

# ---- groups ----
groups={'real':[], 'fp':[]}
# real = gold_ran + model_found from labels.csv (Dolj DTM positives, same product as FP)
for r in csv.DictReader(open(f'{H}/labeled/labels.csv')):
    if r['source'] in ('gold_ran','model_found'):
        groups['real'].append((r['source']+'_'+r['id'],float(r['lon']),float(r['lat'])))
# also the 10 eval-truth (confirmed real in Catane)
truth={11,17,18,30,43,45,50,55,57,64}
em={int(x['idx']):x for x in csv.DictReader(open('/tmp/eval_map.csv'))}
for i in sorted(truth):
    groups['real'].append((f'eval{i}',float(em[i]['lon']),float(em[i]['lat'])))
# fp = the 6 confirmed
for r in csv.DictReader(open('/tmp/fp_coords.csv')):
    groups['fp'].append((f"fp{r['idx']}",float(r['lon']),float(r['lat'])))

rows=[]
for g,items in groups.items():
    for name,lon,lat in items:
        m=measure(lon,lat)
        if m is None: continue
        rows.append((g,name,lon,lat,m['h_pos'],m['h_neg'],m['diam']))

# ---- report ----
out=open('/tmp/scale_test.csv','w');wr=csv.writer(out);wr.writerow(['group','name','lon','lat','h_pos_m','h_neg_m','diam_m'])
for r in rows: wr.writerow([r[0],r[1],f"{r[2]:.5f}",f"{r[3]:.5f}",f"{r[4]:.2f}",f"{r[5]:.2f}",f"{r[6]:.1f}"])
out.close()

def stats(g):
    hs=[r[4] for r in rows if r[0]==g]; ds=[r[6] for r in rows if r[0]==g]
    hn=[r[5] for r in rows if r[0]==g]
    return hs,ds,hn
for g in ('real','fp'):
    hs,ds,hn=stats(g)
    if not hs: print(f"{g}: (0 măsurate)"); continue
    hs=np.array(hs);ds=np.array(ds);hn=np.array(hn)
    print(f"\n=== {g.upper()} (n={len(hs)}) ===")
    print(f"  înălțime dom h_pos (m):  med={np.median(hs):.2f}  min={hs.min():.2f}  p25={np.percentile(hs,25):.2f}  p75={np.percentile(hs,75):.2f}  max={hs.max():.2f}")
    print(f"  diametru (m):            med={np.median(ds):.1f}   min={ds.min():.1f}  p25={np.percentile(ds,25):.1f}  p75={np.percentile(ds,75):.1f}  max={ds.max():.1f}")
    print(f"  adâncime groapă h_neg(m):med={np.median(hn):.2f}  min={hn.min():.2f}  (negativ mare = groapă)")

# separation analysis: best single threshold on h_pos and on diam
real_h=np.array([r[4] for r in rows if r[0]=='real']); fp_h=np.array([r[4] for r in rows if r[0]=='fp'])
real_d=np.array([r[6] for r in rows if r[0]=='real']); fp_d=np.array([r[6] for r in rows if r[0]=='fp'])
print("\n=== SEPARARE ===")
print("Cei 6 FP, individual (h_pos, diam, h_neg):")
for r in rows:
    if r[0]=='fp': print(f"  {r[1]:6s}  h={r[4]:+.2f}m  ⌀={r[6]:.0f}m  groapă={r[5]:+.2f}m")
# how many FP would a height filter h>=0.5 keep as 'mound'?
for thr in (0.3,0.5,0.7,1.0):
    real_keep=(real_h>=thr).mean()*100; fp_cut=(fp_h<thr).mean()*100
    print(f"  prag înălțime ≥{thr}m: păstrează {real_keep:.0f}% din tumuli reali, taie {fp_cut:.0f}% din cei 6 FP")
print(f"\nmin înălțime tumul real = {real_h.min():.2f}m ; max înălțime FP = {fp_h.max():.2f}m  -> {'SE SEPARĂ' if fp_h.max()<real_h.min() else 'suprapunere parțială'}")
print(f"CSV: /tmp/scale_test.csv  ({len(rows)} puncte măsurate)")
