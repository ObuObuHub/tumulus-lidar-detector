#!/usr/bin/env python3
# lib_tumul.py — NUCLEUL MATEMATIC UNIFICAT al detecției de tumuli (10.07.2026, directiva lui Andrei:
# „îmbunătățim logic, nu cu patch-uri"). Înlocuiește peticăria: enumerator+pscore+bg-norm+mască+gauss+MF.
#
# PRINCIPIU: un candidat e movilă dacă (1) relieful lui se potrivește STATISTIC cu amprenta movilelor
# confirmate și (2) măsurătoarea e credibilă (teren real, nu artefact de interpolare).
#
# TREI COMPONENTE:
#   1. DETECȚIE  = filtru potrivit cu AMPRENTA ca șablon (template radial mediu, la mai multe scări),
#                  pe SLRM cu șanțurile mascate la nivel de câmp (fix validat: FP de inel-de-șanț).
#   2. VERIFICARE = distanța Mahalanobis față de amprentă (matematica; câștigătoarea evaluării 10.07)
#                  + probabilitatea CNN (învățarea) — combinate de apelant.
#   3. CREDIBILITATE = micro-textura terenului (teren interpolat = neted nefiresc la 1m).
#
# DISCIPLINA CONSTANTELOR: fiecare prag e derivat din cele 96 de profile confirmate (percentile) sau
# din fizica datelor, cu proveniența scrisă lângă el. Zero numere potrivite din ochi pe datele de test.
#
# Validat de validate_core.py (echivalență cu lanțul vechi pe Catane/zona2/Jijia) înainte de adopție.
import os,math
import numpy as np

H=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_FP_PATH=f"{H}/assets/fingerprint_ro_v2.npz" if os.path.exists(f"{H}/assets/fingerprint_ro_v2.npz") else f"{H}/review/fingerprint_ro_v2.npz"
_FP=np.load(_FP_PATH)  # amprenta = DOAR statistici de formă (profile normalizate, medie, covarianță) — zero coordonate
BINS=_FP['bins']                      # inele 2m, 0-40m
RADS=(BINS[:-1]+BINS[1:])/2.0
PROF_MEAN=_FP['mean']                 # profilul radial mediu (vârf=1, fundal=0), n=96 confirmate
MU=_FP['mu'];COV_INV=_FP['cov_inv']   # Mahalanobis pe binurile 1: (bin 0 = 1 prin construcție)
GATE_T=float(_FP['gate_T'])           # prag Mahalanobis: LOO p99×1.15 pe cele 96 (fără date de test)

# ── constante derivate, cu proveniență ───────────────────────────────────────
DITCH_LVL=-0.25       # m SLRM; sub asta = șanț -> mascat la câmp înainte de MF. Proveniență: movilele
                      # confirmate au 0% pixeli sub -0.25 în inelul 1-3σ (diagnoza 10.07); șanțurile 20-30%.
CENTER_MIN=0.30       # m; centrul candidatului MF trebuie să fie peste câmp. Proveniență: minimul
                      # centru-câmp al movilelor GT zona2 = +0.36m (diagnoza 10.07), cu marjă.
AMP_MIN=0.15          # m; amplitudine minimă a profilului pt. verificare (sub = fără formă măsurabilă).
                      # Proveniență: fingerprint_build — sub 0.15m profilul normalizat devine zgomot.
MICROROUGH_MIN=0.02   # m; micro-textura reală a terenului la 1m (fațete de interpolare ~0.000x;
                      # arătura reală 0.02-0.10). Proveniență: distribuțiile Jijia 10.07 — canarii 0.05-0.075,
                      # fațetele sub 0.001; pragul lasă marja 2.5× sub cel mai slab canar.
MF_SCALES=(0.6,1.0,1.6)  # șablonul amprentei scalat la 0.6×/1×/1.6× (acoperă R50 4.5-22.5m = p2-p98
                      # al razelor confirmate) — nu σ arbitrare.
ZMIN_CONF=13.5        # prag detecție MF (z normalizat la zgomotul câmpului). Proveniență: p5×0.9 al
                      # răspunsului la cele 106 movile confirmate (calibrare 10.07; med 60, p95 206).
                      # NU quantilă a scenei (prima variantă — picată la validare: 8/26 Catane).

def boxblur1(a,r):
    Hh,Ww=a.shape;ii=np.zeros((Hh+1,Ww+1),np.float64);ii[1:,1:]=np.cumsum(np.cumsum(a,0),1)
    ys=np.arange(Hh);xs=np.arange(Ww);y0=np.clip(ys-r,0,Hh);y1=np.clip(ys+r+1,0,Hh);x0=np.clip(xs-r,0,Ww);x1=np.clip(xs+r+1,0,Ww)
    A=ii[y1][:,x1]-ii[y0][:,x1]-ii[y1][:,x0]+ii[y0][:,x0];cnt=((y1-y0)[:,None]*(x1-x0)[None,:]).astype(np.float64)
    return (A/cnt).astype(np.float32)
def downs(a,fc):
    Hh,Ww=a.shape;return a[:Hh//fc*fc,:Ww//fc*fc].reshape(Hh//fc,fc,Ww//fc,fc).mean((1,3))

def slrm2(dem,native_cs):
    """SLRM la 2m efectiv (rețeta validată: elev − boxblur³(30m)); NaN umplut cu mediană de apelant."""
    f=max(1,int(round(2.0/native_cs)))
    z=downs(dem,f);br=int(30/(native_cs*f))
    return z-boxblur1(boxblur1(boxblur1(z,br),br),br)

def _templates():
    """Șabloane 2D zero-mean din PROFILUL MEDIU confirmat, la scările MF_SCALES (px de 2m)."""
    out=[]
    for sc in MF_SCALES:
        Rm=40.0*sc;R=int(Rm/2)
        ys,xs=np.mgrid[-R:R+1,-R:R+1].astype(float);rad=np.hypot(xs,ys)*2.0/sc
        k=np.interp(rad,RADS,PROF_MEAN,right=0.0)
        k-=k.mean();k/=math.sqrt(float((k*k).sum()))+1e-12
        out.append((sc,k))
    return out
_TPL=_templates()

def detect(S,step=2,zmin=None,exclude_margin=25):
    """DETECȚIE: filtru potrivit cu amprenta pe SLRM ditch-masked. S = SLRM 2m (numpy).
    Întoarce listă (y,x,zscore,scale) — maxime locale ale răspunsului normalizat la zgomotul câmpului.
    zmin implicit: ZMIN_CONF (calibrat pe movilele confirmate, nu pe scenă)."""
    Sm=np.where(S<DITCH_LVL,0.0,S)
    noise=float(np.std(Sm[np.abs(Sm)<1.0]))+1e-9  # zgomotul câmpului (fără forme mari)
    best=np.full(S.shape,-9.,np.float32);bsc=np.zeros(S.shape,np.float32)
    for sc,k in _TPL:
        R=k.shape[0]//2
        # corelare prin FFT (unealtă unică, nu bucle)
        from numpy.fft import rfft2,irfft2
        pad=np.zeros_like(Sm);pad[:k.shape[0],:k.shape[1]]=k
        resp=irfft2(rfft2(Sm)*np.conj(rfft2(pad)),s=Sm.shape)
        resp=np.roll(resp,(R,R),(0,1)).astype(np.float32)/noise
        m=resp>best
        best[m]=resp[m];bsc[m]=sc
    if zmin is None: zmin=ZMIN_CONF
    # maxime locale la separare 40m (20px)
    r=10
    b=np.pad(best,r,constant_values=-np.inf)
    for ax in (0,1):
        c=b.copy()
        for d in range(1,r+1):
            c=np.maximum(c,np.roll(b,d,ax));c=np.maximum(c,np.roll(b,-d,ax))
        b=c
    mx=b[r:-r,r:-r]
    pk=(best>=mx)&(best>=zmin)
    pk[:exclude_margin,:]=False;pk[-exclude_margin:,:]=False;pk[:,:exclude_margin]=False;pk[:,-exclude_margin:]=False
    ys,xs=np.nonzero(pk)
    return [(int(y),int(x),float(best[y,x]),float(bsc[y,x])) for y,x in zip(ys,xs)]

def profile(S,y,x):
    """Profil radial normalizat la (y,x) pe SLRM 2m, recentrat pe maximul local ≤10px.
    -> (profil_normalizat, amplitudine_m, centru_minus_câmp_m) sau None la margine."""
    P=S[max(0,y-10):y+11,max(0,x-10):x+11]
    if P.size==0: return None
    iy,ix=np.unravel_index(np.argmax(P),P.shape);cy,cx=max(0,y-10)+iy,max(0,x-10)+ix
    if not (20<=cy<S.shape[0]-20 and 20<=cx<S.shape[1]-20): return None
    W=S[cy-20:cy+21,cx-20:cx+21]
    ys,xs=np.mgrid[0:41,0:41].astype(float);rad=np.hypot(xs-20,ys-20)*2.0
    prof=np.array([float(W[(rad>=BINS[k])&(rad<BINS[k+1])].mean()) for k in range(len(BINS)-1)])
    ref=float(np.mean(prof[-3:]));amp=float(prof[0]-ref)
    cen_field=float(W[rad<=8].mean())-ref
    if amp<AMP_MIN: return None,amp,cen_field
    return (prof-ref)/amp,amp,cen_field

def mahal(p):
    """VERIFICARE matematică: distanța Mahalanobis a profilului normalizat față de amprentă."""
    d=p[1:]-MU
    return float(math.sqrt(max(0.0,d@COV_INV@d)))

def blob_closed(S,y,x):
    """VERIFICARE topologică (dermatoscopie, ideea lui Andrei 10.07: leziunea benignă are contur ÎNCHIS):
    blobul conectat la centru la half-max trebuie să se ÎNCHIDĂ în ±50m. Proveniență fereastră: R50 p98
    al confirmatelor = 22.5m ⇒ diametru max ~45m ⇒ 50m acoperă orice movilă confirmată. Half-max = 0.5
    (definițional, analogul conturului ABCD). Validare 10.07: închis la 65/65 TP; NEînchis la 61/79 FP
    reziduali care treceau amprenta (profilul radial mediază unghiular — orb la forme deschise).
    -> (închis?, asimetrie, neregularitate_margine) sau (False,None,None)."""
    from collections import deque
    P=S[max(0,y-10):y+11,max(0,x-10):x+11]
    if P.size==0: return False,None,None
    iy,ix=np.unravel_index(np.argmax(P),P.shape);cy,cx=max(0,y-10)+iy,max(0,x-10)+ix
    if not (25<=cy<S.shape[0]-25 and 25<=cx<S.shape[1]-25): return False,None,None
    W=S[cy-25:cy+26,cx-25:cx+26]
    ys2,xs2=np.mgrid[0:51,0:51].astype(float);rad=np.hypot(xs2-25,ys2-25)*2.0
    ref=float(np.median(W[rad>40]));amp=float(W[25,25]-ref)
    if amp<AMP_MIN: return False,None,None
    Wn=(W-ref)/amp
    M=np.zeros_like(Wn,bool);dq=deque([(25,25)]);M[25,25]=True
    while dq:
        yy,xx=dq.popleft()
        for dy,dx in ((1,0),(-1,0),(0,1),(0,-1)):
            ny,nx=yy+dy,xx+dx
            if 0<=ny<51 and 0<=nx<51 and not M[ny,nx] and Wn[ny,nx]>=0.5:
                M[ny,nx]=True;dq.append((ny,nx))
    A=int(M.sum())
    if A<5 or M[0,:].any() or M[-1,:].any() or M[:,0].any() or M[:,-1].any(): return False,None,None
    Pm=int((M[1:,:]!=M[:-1,:]).sum())+int((M[:,1:]!=M[:,:-1]).sum())
    border=Pm*Pm/(4*math.pi*A)
    ys3,xs3=np.nonzero(M);ys3=ys3.astype(float);xs3=xs3.astype(float)
    my,mx_=ys3.mean(),xs3.mean()
    cyy=((ys3-my)**2).mean();cxx=((xs3-mx_)**2).mean();cxy=((ys3-my)*(xs3-mx_)).mean()
    th=0.5*math.atan2(2*cxy,cxx-cyy+1e-12)
    def refl(theta):
        c,s=math.cos(theta),math.sin(theta)
        d=(-s*(xs3-mx_)+c*(ys3-my))
        py=np.rint(ys3-2*d*c).astype(int);px=np.rint(xs3-2*d*(-s)).astype(int)
        ok=(py>=0)&(py<51)&(px>=0)&(px<51)
        return 1.0-M[py[ok],px[ok]].sum()/len(ys3)
    asym=(refl(th)+refl(th+math.pi/2))/2.0
    # asym/border: doar metrici informative (măsurate 10.07: NU separă FP reziduali — aceia pică pe închidere)
    return True,float(asym),float(border)

def microrough(dem_1m_window,r_px=3):
    """CREDIBILITATE: micro-textura reală (std al rezidualului high-pass) în centru 40m.
    dem_1m_window = fereastră DEM la rezoluția nativă ≤1m (la 0.5m se dă direct; testul e pe px nativi)."""
    hp=dem_1m_window-boxblur1(dem_1m_window,r_px)
    Hh,Ww=hp.shape
    c=hp[Hh//2-20:Hh//2+20,Ww//2-20:Ww//2+20]
    return float(c.std())

def verdict(p,amp,cen_field,mr,native_cs,S=None,y=None,x=None):
    """Decizia unificată. mr=None => date fără artefacte de interpolare (LAKI3 0.5m) — testul se sare.
    S,y,x (opțional dar recomandat) => se aplică și criteriul topologic al conturului închis."""
    if p is None or amp<AMP_MIN: return False,'fără formă măsurabilă'
    if cen_field<CENTER_MIN: return False,f'centru sub câmp+{CENTER_MIN}'
    if mr is not None and native_cs>=0.9 and mr<MICROROUGH_MIN: return False,'teren interpolat'
    d=mahal(p)
    if d>GATE_T: return False,f'amprentă d={d:.1f}>{GATE_T:.1f}'
    if S is not None:
        closed,asym,border=blob_closed(S,y,x)
        if not closed: return False,'contur neînchis (>50m)'
    return True,f'd={d:.2f}'
