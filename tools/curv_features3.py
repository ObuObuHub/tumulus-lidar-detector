#!/usr/bin/env python3
# curv_features3.py IN_CSV OUT_CSV — trasături de formă SCALE-ADAPTIVE (v3).
# Diferența față de v2: estimează MĂRIMEA movilei (sigma) din potrivirea Gauss, apoi calculează TOATE trăsăturile
# la SCARA ei (raze cap/SLRM ∝ sigma) → invariant la mărime. Merge pe movile mari (RO) ȘI mici (NL/erodate).
# Aceleași NUME de coloane ca v2 (prom,gfit,fwhm_m,sym,convex,rough,slrm15,slrm45,mono,relief_m) → curv_gate.py merge neschimbat.
import os,sys,subprocess,math,csv,zipfile
import numpy as np
CACHE="/tmp/laki3";CS=0.5;TPX=2000;os.makedirs(CACHE,exist_ok=True)
import pyproj
_TF={}
def _tf(s,t):
    if (s,t) not in _TF:_TF[(s,t)]=pyproj.Transformer.from_crs(s,t,always_xy=True)
    return _TF[(s,t)]
def trans_batch(pts,s,t):
    tf=_tf(s,t);out=[]
    for a,b in pts:
        x,y=tf.transform(a,b);out.append((x,y) if math.isfinite(x) and math.isfinite(y) else (None,None))
    return out
def load_one(nkm,ekm):
    NUME=f"{nkm}_{ekm}";npy=f"{CACHE}/{NUME}.npy"
    if os.path.exists(npy):
        try:return np.load(npy)
        except:pass
    z=f"{CACHE}/{NUME}.zip"
    if not os.path.exists(z):subprocess.run(["curl","-s","--max-time","120","-o",z,f"https://geoportal.ancpi.ro/laki3_mnt/zip/{NUME}.zip"],check=False)
    try: zf=zipfile.ZipFile(z)
    except:
        if os.path.exists(z):os.remove(z)
        return None
    asc=[n for n in zf.namelist() if n.lower().endswith('.asc')]
    if not asc: return None
    raw=zf.read(asc[0]).decode('latin-1').replace(',','.');lines=raw.split('\n');hdr={};i=0
    while i<len(lines):
        p=lines[i].split()
        if len(p)>=2 and p[0].lower() in ('ncols','nrows','xllcorner','yllcorner','cellsize','nodata_value'):hdr[p[0].lower()]=float(p[1]);i+=1
        else:break
    nc=int(hdr['ncols']);nr=int(hdr['nrows']);nd=hdr.get('nodata_value',-9999)
    data=np.fromstring(' '.join(lines[i:]),sep=' ',dtype=np.float32)[:nc*nr].reshape(nr,nc);data[data==nd]=np.nan;np.save(npy,data);return data
def downs(a,fc):
    Hh,Ww=a.shape;return a[:Hh//fc*fc,:Ww//fc*fc].reshape(Hh//fc,fc,Ww//fc,fc).mean((1,3))
WIN=110.0;HALF=int(WIN/CS)
def local_window(est,nord):
    e0=int((est-WIN)//1000);e1=int((est+WIN)//1000);n0=int((nord-WIN)//1000);n1=int((nord+WIN)//1000)
    xll0=e0*1000;ytop0=(n1+1)*1000;W=(e1-e0+1)*TPX;Hh=(n1-n0+1)*TPX;mos=np.full((Hh,W),np.nan,np.float32);got=0
    for nk in range(n0,n1+1):
        for ek in range(e0,e1+1):
            d=load_one(nk,ek)
            if d is None:continue
            got+=1;ox=int((ek*1000-xll0)/CS);oy=int((ytop0-(nk+1)*1000)/CS);mos[oy:oy+TPX,ox:ox+TPX]=d[:TPX,:TPX]
    if got==0:return None
    px=int((est-xll0)/CS);py=int((ytop0-nord)/CS);return mos[py-HALF:py+HALF,px-HALF:px+HALF]
def boxblur1(a,r):
    r=max(1,int(r));Hh,Ww=a.shape;ii=np.zeros((Hh+1,Ww+1),np.float64);ii[1:,1:]=np.cumsum(np.cumsum(a,0),1)
    ys=np.arange(Hh);xs=np.arange(Ww);y0=np.clip(ys-r,0,Hh);y1=np.clip(ys+r+1,0,Hh);x0=np.clip(xs-r,0,Ww);x1=np.clip(xs+r+1,0,Ww)
    A=ii[y1][:,x1]-ii[y0][:,x1]-ii[y1][:,x0]+ii[y0][:,x0];cnt=((y1-y0)[:,None]*(x1-x0)[None,:]).astype(np.float64)
    return (A/cnt).astype(np.float32)
def pos_openness(z,cy,cx,cs2,L_m,naz=8):
    # openness pozitivă la (cy,cx): media pe azimuturi a (90° - unghiul max spre orizont în L_m). Apex de dom = mare.
    z0=float(z[cy,cx]);L=max(2,int(L_m/cs2));acc=0.0
    for a in range(naz):
        th=2*math.pi*a/naz;dy=math.sin(th);dx=math.cos(th);mx=-90.0
        for r in range(1,L+1):
            yy=int(round(cy+dy*r));xx=int(round(cx+dx*r))
            if 0<=yy<z.shape[0] and 0<=xx<z.shape[1]:
                el=math.degrees(math.atan2(float(z[yy,xx])-z0,r*cs2))
                if el>mx:mx=el
        acc+=90.0-mx
    return acc/naz
def fit_sigma(rc,prof):
    best=(1e9,8.0,prof.max() if len(prof) else 0.1,0.0)
    for s in np.linspace(3,40,38):
        g=np.exp(-(rc**2)/(2*s*s));M=np.c_[g,np.ones(len(g))];sol,_,_,_=np.linalg.lstsq(M,prof,rcond=None);ss=float(((prof-M@sol)**2).sum())
        if ss<best[0]:best=(ss,s,sol[0],sol[1])
    return best  # ss, sigma, A, d
def radial(res,rad,cap,rmax,step=2):
    edges=np.arange(0,rmax+step,step);prof=[];rc=[]
    for k in range(len(edges)-1):
        m=(rad>=edges[k])&(rad<edges[k+1])&cap
        if m.sum()>=3:prof.append(float(res[m].mean()));rc.append((edges[k]+edges[k+1])/2)
    return np.array(rc),np.array(prof)
def analyze(est,nord):
    w=local_window(est,nord)
    if w is None or w.shape!=(2*HALF,2*HALF) or np.isnan(w).mean()>0.15:return None
    w=np.nan_to_num(w,nan=np.nanmedian(w));f=int(round(2.0/CS));z=downs(w,f);cs2=CS*f
    c0=z.shape[0]//2;rr=int(16/cs2);sub=z[c0-rr:c0+rr,c0-rr:c0+rr]
    off=np.unravel_index(np.argmax(sub),sub.shape);ay=c0-rr+off[0];ax=c0-rr+off[1]
    ys,xs=np.mgrid[0:z.shape[0],0:z.shape[1]].astype(np.float64);rad=np.hypot(ys-ay,xs-ax)*cs2
    # PASS 1: detrend cu fundal generos -> estimează sigma
    bg0=(rad>=30)&(rad<=52)
    if bg0.sum()<20:return None
    A0=np.c_[xs[bg0].ravel(),ys[bg0].ravel(),np.ones(bg0.sum())];c0f,_,_,_=np.linalg.lstsq(A0,z[bg0].ravel(),rcond=None)
    res0=z-(c0f[0]*xs+c0f[1]*ys+c0f[2])
    rc0,prof0=radial(res0,rad,rad<=45,45)
    if len(prof0)<4:return None
    _,sigma,_,_=fit_sigma(rc0,prof0);s=float(np.clip(sigma,3,35))
    # SCARA ADAPTIVĂ
    R=float(np.clip(2.2*s,10,48));Rin=float(np.clip(0.9*s,4,22))
    bgi=float(np.clip(R+6,R+6,52));bgo=float(np.clip(R+24,bgi+6,54))
    # PASS 2: detrend la fundal ADAPTIV (chiar în afara movilei)
    bg=(rad>=bgi)&(rad<=bgo)
    if bg.sum()<15:bg=bg0
    A=np.c_[xs[bg].ravel(),ys[bg].ravel(),np.ones(bg.sum())];coef,_,_,_=np.linalg.lstsq(A,z[bg].ravel(),rcond=None)
    res=z-(coef[0]*xs+coef[1]*ys+coef[2])
    cap=rad<=R;capin=rad<=Rin;prom=max(float(res[int(ay),int(ax)]),0.05);relief=float(z[cap].max()-np.percentile(z[cap],10))
    rc,prof=radial(res,rad,cap,R)
    if len(prof)<4:return None
    ss,sg2,Aamp,d=fit_sigma(rc,prof);sstot=float(((prof-prof.mean())**2).sum())+1e-9;gfit=1-ss/sstot;fwhm_m=float(2.3548*sg2)
    cvs=[]
    edges=np.arange(0,R+2,2)
    for k in range(len(edges)-1):
        m=(rad>=edges[k])&(rad<edges[k+1])&cap
        if m.sum()>=8:
            v=res[m];mu=v.mean()
            if abs(mu)>1e-3:cvs.append(v.std()/abs(mu))
    sym=float(1.0/(1.0+np.mean(cvs))) if cvs else 0.0
    lap=(np.roll(res,1,0)+np.roll(res,-1,0)+np.roll(res,1,1)+np.roll(res,-1,1)-4*res)/(cs2*cs2)
    convex=float(-lap[capin].mean()/prom);rough=float(lap[cap].std()/prom)
    def slpk(r_m):
        bgb=boxblur1(boxblur1(z,int(max(2,r_m/cs2))),int(max(2,r_m/cs2)));return float((z-bgb)[capin].mean()/prom)
    slrm15=slpk(max(8,1.2*s));slrm45=slpk(max(16,2.6*s))   # SLRM la scara movilei
    mono=float(np.mean(np.diff(prof)<=0)) if len(prof)>2 else 0.0
    # OPENNESS (Niculiță) — TESTATĂ 25.06, REDUNDANTĂ cu convex/sym → fără câștig la deploy (Catane 15→16); NEadoptată. pos_openness rămâne pt revisit.
    return dict(prom=prom,gfit=gfit,fwhm_m=fwhm_m,sym=sym,convex=convex,rough=rough,slrm15=slrm15,slrm45=slrm45,mono=mono,relief_m=relief)
FEATS=['prom','gfit','fwhm_m','sym','convex','rough','slrm15','slrm45','mono','relief_m']
def main():
    inp=sys.argv[1];outp=sys.argv[2];rows=list(csv.DictReader(open(inp)))
    cols={c.lower():c for c in rows[0].keys()};lonc=cols.get('lon');latc=cols.get('lat')
    pts=[(float(r[lonc]),float(r[latc])) for r in rows]
    print(f"{len(pts)} puncte; transform...",flush=True);st=trans_batch(pts,"EPSG:4326","EPSG:3844")
    extra=list(rows[0].keys())
    with open(outp,'w',newline='') as fo:
        wr=csv.writer(fo);wr.writerow(extra+FEATS);ok=0;na=0
        for i,(r,(e,n)) in enumerate(zip(rows,st)):
            a=analyze(e,n) if e is not None else None
            if a is None:na+=1;wr.writerow([r.get(c,'') for c in extra]+['NA']*len(FEATS))
            else:ok+=1;wr.writerow([r.get(c,'') for c in extra]+[f"{a[k]:.5f}" for k in FEATS])
            if (i+1)%200==0:print(f"  {i+1}/{len(pts)} (ok {ok})",flush=True)
        print(f"-> {outp} | ok {ok} | NA {na}",flush=True)
if __name__=='__main__':main()
