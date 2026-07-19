#!/usr/bin/env python3
# tumul_scan.py — SCANERUL INTEGRAT de producție (10.07.2026, „Pornește" — Andrei msg 12906).
# Leagă tot ce s-a validat azi într-o singură comandă:
#   DETECȚIE (filtru potrivit cu amprenta, lib_tumul.detect, șanțuri mascate)
#   -> VERIFICARE per candidat: cele 4 gărzi matematice (amprentă Mahalanobis + centru + contur închis
#      + credibilitate) și CNN r3 4ch multi-offset (max pe ±12m)
#   -> ieșire: CSV ranked + sumar la praguri de operare.
# Pragurile de operare implicite se aleg cu tune_operating_point.py pe etaloanele oarbe (dezvăluit)
# și se verifică pe setul PROASPĂT (eISM 48) — regula raportului detecții/FP a lui Andrei.
# Usage: tumul_scan.py W_LON E_LON S_LAT N_LAT OUT_CSV   (env: SOURCE=laki3|<fisier .npy 1m>, CNN_THR)
import os,sys,csv,math,json,importlib.util
import numpy as np
import torch,torch.nn as nn
H=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("LAKI3_CACHE","/tmp/laki3")  # override cu LAKI3_CACHE=<dir dale/arhivă locală>
dev=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
lt=importlib.util.spec_from_file_location("lt",f"{H}/tools/lib_tumul.py");LT=importlib.util.module_from_spec(lt);lt.loader.exec_module(LT)
lc=importlib.util.spec_from_file_location("lc",f"{H}/tools/lib_channels.py");LC=importlib.util.module_from_spec(lc);lc.loader.exec_module(LC)
import pyproj
_TF=pyproj.Transformer.from_crs("EPSG:4326","EPSG:3844",always_xy=True)
_TFi=pyproj.Transformer.from_crs("EPSG:3844","EPSG:4326",always_xy=True)
CHANS=['hs','slrm','slope','rough']
# Prag CNN implicit 0.80 = punctul de operare ECHILIBRAT (reglat 10.07 pe Catane+zona2 — dezvăluit —
# și VERIFICAT pe setul proaspăt eISM neatins de reglaj: 14/16=88%): Catane 19/26 @0.19FP/km²,
# zona2 7/11 @0 FP. Mod STRICT: CNN_THR=0.9 → Catane 18/26 @0.07FP/km² (bate producția pe ambele axe:
# 17/26 @0.11), zona2 5/11 @0. Tabel complet: review/tune_operating.log.
CNN_THR=float(os.environ.get('CNN_THR','0.80'))  # doar pt uneltele de analiză (comparații cu lanțul vechi); decizia NU-l mai folosește
# ── DECIZIA „DINTR-O BUCATĂ" (10.07.2026, Andrei msg 12939/12942): formula v3 noisy-OR „ochi SAU formă".
# Înlocuiește lanțul de vetouri (5 gărzi + prag CNN) cu UN scor + UN prag. Verificată: transplant seed 12
# PROASPĂT 69/150 @2 FP vs lanț 54/150 @6 FP (dominare pe ambele axe); cifre complete: PLAN_FORMULA_OPTIMA.md.
# Gărzile vechi rămân calculate ca SEMNALE (mahal, contur, etc.) — intră în formulă, nu mai au veto.
# Singura excludere dură rămasă: credibilitatea măsurătorii (teren interpolat, doar pe date >=0.9m).
FUSE_THR=float(os.environ.get('FUSE_THR','0.70'))  # 0.70 + filtru FP = punctul de operare din 16.07.2026 (decizia Andrei); vechiul regim: FUSE_THR=0.80 FP_FILTER_THR=0
VFRAC_MIN=float(os.environ.get('VFRAC_MIN','0.98'))  # garda de cusaturi NoData (fereastra 80m; 0=dezactivat)
_FZ=json.load(open(os.environ.get('FORMULA',f"{H}/assets/fuse_formula_v4.json")))
_CHS=[(ch['features'],np.array(ch['mu']),np.array(ch['sd']),np.array(ch['w']),ch['b']) for ch in _FZ['channels']]
def fuse_score(c):
    vals=dict(zlog=math.log10(max(c['z_mf'],1.0)),scale=c['scale'],mahal=min(c['mahal'],40.0),
              amp=min(c['amp'],6.0),cf=c['cf'],closed=float(bool(c['closed'])),
              asym=c['asym'] if c['asym'] is not None else 1.0,
              border=min(c['border'],6.0) if c['border'] is not None else 6.0,
              cnn=0.0 if math.isnan(c['cnn']) else c['cnn'])
    _cc=max(min(vals['cnn'],0.999),0.001);vals['cnnlogit']=math.log(_cc/(1-_cc))
    q=1.0
    for feats,mu,sd,w,b in _CHS:
        x=np.array([vals[f] for f in feats]);z=float(((x-mu)/sd)@w+b);q*=1-1/(1+math.exp(-z))
    return 1-q
# ── Filtru FP (post-formulă): P(FP) învățat pe marcajul FP al lui Andrei (Dolj v4). Validat orb pe
# Catane + sheet. Din 16.07.2026 IMPLICIT ACTIV la 0.55 (împreună cu FUSE_THR=0.70: Catane 21/26 GT,
# ~59 FP vs 18/26, 47 la vechiul regim). FP_FILTER_THR=0 → dezactivat, keep = doar formulă+gărzi.
FP_FILTER_THR=float(os.environ.get('FP_FILTER_THR','0.55'))
# SCUT SCOS (18.07.2026, decizia Andrei) = filtru GLOBAL. Scutul HIBRID din 17.07 (FP_PROTECT=0.80) se
# baza pe „filtrul taie 50% din movile în banda 0.80–0.838" — artefact de etichete incomplete (banda nu
# era parcursă integral); pe verdictele COMPLETE filtrul taie acolo 73% din FP și pierde 13% din BUMP
# (analiza_fp_dolj_20260715/recalc_banda_final_20260718.py; verificare end-to-end cu toate cele 3
# regimuri reproduse: review/verify_shield_off_20260718.log — fără scut 21/26 GT, 60 FP).
FP_PROTECT=float(os.environ.get('FP_PROTECT','2'))
# Pragul dur pe bombare (fost „dome0" <1e-5): din 18.07.2026 DOME_MIN=5e-4 — pe TOATE cele 690 de movile
# confirmate (Scan1+recuperare) niciuna nu e sub 5e-4 (podeaua reală: 5.6e-4); pe Catane efect zero
# (21/26, 60 FP identic). Cifre: analiza_fp_dolj_20260715/pattern2_numbers_20260718.md §1.
DOME_MIN=float(os.environ.get('DOME_MIN','5e-4'))
# DEPLOY = v3 (7 semnale + domeness + aniso) ca JUDECĂTOR PRINCIPAL; v1/v2/v4 în carantină
# (review/carantina_asset_json_20260716/) — v4 (+contur) regresa pe Catane, NEpromovat.
_fpmp=f"{H}/assets/fp_filter_model_v3.json"
_FPM=None
if FP_FILTER_THR>0:
    assert os.path.exists(_fpmp), f"filtru FP activ (FP_FILTER_THR={FP_FILTER_THR}) dar lipsește {_fpmp}"
    _fj=json.load(open(_fpmp)); _FPM=(_fj['feats'],np.array(_fj['mu']),np.array(_fj['sd']),np.array(_fj['w']))
# CASCADĂ v6 (18.07.2026, decizia Andrei „cel mai agresiv"): al 2-lea filtru (v3 + dome3/dome10 multi-scară
# + ring_comp/ring_unif contur, fit pe verdictele COMPLETE — dolj_filter_v5/patterns2) taie DOAR ce
# condamnă AMBELE filtre: keep cere pfp<FP_FILTER_THR ȘI pfp6<FP6_THR. v6 singur pierdea o movilă pe
# Catane (20/26) → NU înlocuiește v3, îl dublează pe prag blând. FP6_THR=0 = cascada oprită.
FP6_THR=float(os.environ.get('FP6_THR','0.80'))
_fp6p=f"{H}/assets/fp_filter_model_v6.json"
_FP6M=None
if FP_FILTER_THR>0 and FP6_THR>0:
    assert os.path.exists(_fp6p), f"cascada v6 activă (FP6_THR={FP6_THR}) dar lipsește {_fp6p}"
    _f6=json.load(open(_fp6p)); _FP6M=(_f6['feats'],np.array(_f6['mu']),np.array(_f6['sd']),np.array(_f6['w']))
_FP_NEEDS_SHAPE=any(M is not None and any(f in M[0] for f in ('domeness','aniso','ring_comp','ring_unif')) for M in (_FPM,_FP6M))
_NEED_MULTI=(_FP6M is not None and any(f in _FP6M[0] for f in ('dome3','dome10')))
def _shape_feats(fill,py,px,HW=80):
    """curbură (domeness=-kmax, aniso=kmax-kmin) + CONTUR (ring_comp=completitudine inel, ring_unif=circularitate)
    din elevația 2m. Rotund/inel închis → movilă; liniar/inel deschis → FP. Edge -> zerouri (rar)."""
    w=fill[py-HW:py+HW,px-HW:px+HW]
    if w.shape!=(2*HW,2*HW): return 0.0,0.0,0.0,0.0,0.0,0.0
    z2=LC.downs(w.astype(np.float32),4); zs=LC.boxblur1(z2,3); cc=z2.shape[0]//2; rad=12
    Zy,Zx=np.gradient(zs,2.0); Zyy,_=np.gradient(Zy,2.0); Zxy,Zxx=np.gradient(Zx,2.0)
    mn=(Zxx+Zyy)/2.0; disc=np.sqrt(np.clip(((Zxx-Zyy)/2.0)**2+Zxy**2,0,None)); kmax=mn+disc; kmin=mn-disc
    dome=-float(np.percentile(kmax[cc-rad:cc+rad+1,cc-rad:cc+rad+1],5))
    aniso=float(np.percentile((kmax-kmin)[cc-rad:cc+rad+1,cc-rad:cc+rad+1],95))
    # bombarea multi-scară pt cascada v6 (rețeta compute_curv_features_all: netezire 3m→blur1, 10m→blur5)
    dome3=dome10=0.0
    if _NEED_MULTI:
        for _bl,_nm in ((1,'d3'),(5,'d10')):
            _zs=LC.boxblur1(z2,_bl)
            _Zy,_Zx=np.gradient(_zs,2.0); _Zyy,_=np.gradient(_Zy,2.0); _Zxy,_Zxx=np.gradient(_Zx,2.0)
            _mn=(_Zxx+_Zyy)/2.0; _di=np.sqrt(np.clip(((_Zxx-_Zyy)/2.0)**2+_Zxy**2,0,None))
            _d=-float(np.percentile((_mn+_di)[cc-rad:cc+rad+1,cc-rad:cc+rad+1],5))
            if _nm=='d3': dome3=_d
            else: dome10=_d
    # contur: inel de relief in jurul centrului
    zc=LC.boxblur1(z2,2); h0=float(zc[cc,cc]); A=24; best=None
    for r in range(3,11):
        rel=[]
        for a in range(A):
            th=2*math.pi*a/A; yy=int(round(cc+r*math.sin(th))); xx=int(round(cc+r*math.cos(th)))
            if 0<=yy<z2.shape[0] and 0<=xx<z2.shape[1]: rel.append(h0-float(zc[yy,xx]))
        if len(rel)<A: continue
        rel=np.array(rel)
        if best is None or rel.mean()>best[0]: best=(rel.mean(),rel)
    if best is None: return dome,aniso,0.0,0.0,dome3,dome10
    rel=best[1]; comp=float((rel>0.10).mean()); unif=min(float(max(0.0,rel.min())/rel.mean()) if rel.mean()>1e-6 else 0.0,3.0)
    return dome,aniso,comp,unif,dome3,dome10
def fp_score(c,M=None):
    """P(FP) din semnalele brute (+curbură/contur dacă modelul le cere). M=modelul (implicit v3 = _FPM);
    0.0 dacă filtrul respectiv e dezactivat."""
    if M is None: M=_FPM
    if M is None: return 0.0
    feats,mu,sd,w=M
    cc=max(min(0.0 if math.isnan(c['cnn']) else c['cnn'],0.999),0.001)
    v=dict(zlog=math.log10(max(c['z_mf'],1.0)),scale=c['scale'],mahal=min(c['mahal'],40.0),
           amplog=math.log10(max(c['amp'],0.01)),cf=c['cf'],closed=float(bool(c['closed'])),
           cnnlogit=math.log(cc/(1-cc)),domeness=c.get('domeness',0.0),aniso=c.get('aniso',0.0),
           ring_comp=c.get('ring_comp',0.0),ring_unif=c.get('ring_unif',0.0),
           dome3=c.get('dome3',0.0),dome10=c.get('dome10',0.0))
    x=np.array([v[f] for f in feats]); zz=w[0]+float(((x-mu)/sd)@w[1:])
    return 1.0/(1.0+math.exp(-max(min(zz,30.0),-30.0)))
class Net(nn.Module):
    def __init__(s,nch):
        super().__init__();s.c=nn.Sequential(nn.Conv2d(nch,16,3,2,1),nn.ReLU(),nn.Conv2d(16,32,3,2,1),nn.ReLU(),nn.Conv2d(32,64,3,2,1),nn.ReLU(),nn.AdaptiveAvgPool2d(1));s.f=nn.Linear(64,1)
    def forward(s,x): return s.f(s.c(x).flatten(1)).squeeze(1)
_net=None
MODEL_PATH=f'{H}/multichannel_cnn.pt' if os.path.exists(f'{H}/multichannel_cnn.pt') else f'{H}/models/multi_r3_treat.pt'
def net():
    global _net
    if _net is None:
        _net=Net(len(CHANS)).to(dev);_net.load_state_dict(torch.load(MODEL_PATH,map_location=dev,weights_only=True));_net.eval()
    return _net
def scan(dem,native_cs):
    """Nucleul: DEM (NaN permis) -> listă candidați dict(y2,x2,z_mf,mahal,cnn,verdict,why,asym,border).
    y2,x2 = px în grila 2m a SLRM-ului."""
    med=float(np.nanmedian(dem));fill=np.where(np.isfinite(dem),dem,med)
    S=LT.slrm2(fill,native_cs)
    dets=LT.detect(S)
    f=max(1,int(round(2.0/native_cs)))
    out=[]
    batch=[];meta=[]
    for y,x,z,sc in dets:
        pr=LT.profile(S,y,x)
        if pr is None: continue
        p,amp,cf=pr if len(pr)==3 else (None,0,0)
        # credibilitate: doar pe date >=0.9m (MNT); fereastra nativă
        mr=None
        if native_cs>=0.9:
            py,px=y*f,x*f
            w=fill[max(0,py-60):py+60,max(0,px-60):px+60]
            mr=LT.microrough(w) if w.shape==(120,120) else None
        ok,why=LT.verdict(p,amp,cf,mr,native_cs,S,y,x)  # verdictul vechilor gărzi = doar informativ (câmpul 'verdict')
        d=LT.mahal(p) if p is not None else 99.0
        closed,asym,border=LT.blob_closed(S,y,x)
        interp=(mr is not None and native_cs>=0.9 and mr<LT.MICROROUGH_MIN)  # credibilitate: excludere dură
        py,px=y*f,x*f;HW=int(80/native_cs/2)
        wv=dem[max(0,py-HW):py+HW,max(0,px-HW):px+HW]
        vfr=float(np.isfinite(wv).mean()) if wv.size else 0.0  # cusătură NoData (10.07: 23% din lista Dolj, cost zero)
        rec=dict(y2=y,x2=x,z_mf=z,mahal=d,verdict=ok,why=why,asym=asym,border=border,cnn=float('nan'),
                 scale=sc,amp=amp,cf=cf,closed=closed,interp=interp,vfrac=vfr)
        out.append(rec)
        # CNN pentru TOATE propunerile (decizia e a formulei, nu a unui veto); multi-offset ±12m
        for dy in (-int(12/native_cs),0,int(12/native_cs)):
            for dx in (-int(12/native_cs),0,int(12/native_cs)):
                w=fill[py+dy-HW:py+dy+HW,px+dx-HW:px+dx+HW]
                if w.shape!=(2*HW,2*HW): continue
                s=LC.stamp_multi(w,native_cs,CHANS)
                if s is None: continue
                batch.append(s);meta.append(len(out)-1)
    if batch:
        X=torch.tensor(np.array(batch,np.uint8))
        with torch.no_grad():
            for i in range(0,len(X),512):
                v=torch.sigmoid(net()(X[i:i+512].float().to(dev)/255.)).cpu().numpy()
                for j,val in enumerate(v):
                    k=meta[i+j]
                    if math.isnan(out[k]['cnn']) or val>out[k]['cnn']: out[k]['cnn']=float(val)
    for c in out:
        c['fuse']=fuse_score(c)
        if _FP_NEEDS_SHAPE and FP_FILTER_THR>0:
            c['domeness'],c['aniso'],c['ring_comp'],c['ring_unif'],c['dome3'],c['dome10']=_shape_feats(fill,c['y2']*f,c['x2']*f)  # curbură+contur (doar când activ)
        c['pfp']=fp_score(c)
        c['pfp6']=fp_score(c,_FP6M)
        # Bombare sub DOME_MIN (plat/concav, NICIODATĂ dom) = FP fără excepție. Istoric: regula lui Andrei
        # 16.07 la <1e-5 (0 movile reale pe 748 etichetate); 18.07 ridicat la 5e-4 pe verdictele COMPLETE
        # (0/690 movile sub 5e-4, podeaua la 5.6e-4; Catane neatins) → tăiat necondiționat când filtrul e activ.
        _dome0 = (FP_FILTER_THR>0 and c.get('domeness',1.0)<DOME_MIN)
        # Filtrul: v3 decide (pfp), iar cascada v6 (pfp6) taie DOAR ce condamnă amândouă (FP6_THR=0 = oprită).
        _fpok = (FP_FILTER_THR<=0 or c['fuse']>=FP_PROTECT or
                 (c['pfp']<FP_FILTER_THR and (FP6_THR<=0 or c['pfp6']<FP6_THR)))
        c['keep']=(c['fuse']>=FUSE_THR) and not c['interp'] and c['vfrac']>=VFRAC_MIN and not _dome0 and _fpok
    return out,S
def scan_laki3(W_LON,E_LON,S_LAT,N_LAT):
    CACHE=os.environ["LAKI3_CACHE"]
    cor=[_TF.transform(a,b) for a,b in [(W_LON,S_LAT),(E_LON,S_LAT),(W_LON,N_LAT),(E_LON,N_LAT)]]
    es=[e for e,n in cor];ns=[n for e,n in cor];MARG=400
    e0=int((min(es)-MARG)//1000);e1=int((max(es)+MARG)//1000);n0=int((min(ns)-MARG)//1000);n1=int((max(ns)+MARG)//1000)
    xll=e0*1000;ytop=(n1+1)*1000
    mos=np.full(((n1-n0+1)*2000,(e1-e0+1)*2000),np.nan,np.float32);nt=0
    for nk in range(n0,n1+1):
        for ek in range(e0,e1+1):
            p=f"{CACHE}/{nk}_{ek}.npy"
            if not os.path.exists(p): continue
            d=np.load(p);nt+=1
            ox=int((ek*1000-xll)/0.5);oy=int((ytop-(nk+1)*1000)/0.5);mos[oy:oy+2000,ox:ox+2000]=d[:2000,:2000]
    area=float(np.isfinite(mos).sum())*0.25/1e6
    print(f"mozaic: {nt} dale, ~{area:.0f} km²",flush=True)
    return mos,xll,ytop,area
def main():
    W_LON,E_LON,S_LAT,N_LAT=map(float,sys.argv[1:5]);OUT=sys.argv[5]
    mos,xll,ytop,area=scan_laki3(W_LON,E_LON,S_LAT,N_LAT)
    cands,S=scan(mos,0.5)
    keep=[c for c in cands if c['keep']]
    with open(OUT,'w',newline='') as fo:
        w=csv.writer(fo);w.writerow(['lon','lat','fuse','z_mf','mahal','cnn','asym','border'])
        for c in sorted(keep,key=lambda c:-(c['fuse'])):
            e=xll+c['x2']*2.0;n=ytop-c['y2']*2.0;lo,la=_TFi.transform(e,n)
            w.writerow([f"{lo:.6f}",f"{la:.6f}",f"{c['fuse']:.3f}",f"{c['z_mf']:.0f}",f"{c['mahal']:.2f}",
                        f"{c['cnn']:.3f}" if not math.isnan(c['cnn']) else '',
                        f"{c['asym']:.3f}" if c['asym'] is not None else '',f"{c['border']:.2f}" if c['border'] is not None else ''])
    print(f"propuneri: {len(cands)} | FINALE (formulă>={FUSE_THR}): {len(keep)} ({len(keep)/max(area,1e-9):.2f}/km²) -> {OUT}",flush=True)
if __name__=='__main__': main()
