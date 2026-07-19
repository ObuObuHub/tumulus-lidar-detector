#!/usr/bin/env python3
# lib_channels.py — SURSĂ UNICĂ de adevăr pt canalele multi-canal (Yang-style, focus formă, FĂRĂ DEM brut).
# Toate canalele calculate din DEM la rezoluție efectivă 2m, apoi stretch128 -> uint8 (C,128,128).
# Canale: hs (hillshade multidir) · slrm (relief local = proxy curbură) · slope (pantă) · rough (rugozitate).
# Rețeta hs/slrm/stretch128/homog/boxblur = IDENTICĂ cut_2ch.py (paritate cu ce s-a validat deja).
# Folosit de: cut RO (arhiva T7), cut DK (WCS), benchmark multi-canal. NU include DEM brut (altitudinea
# absolută = scurgere de domeniu între țări; păstrăm doar derivate de formă, invariante la sursă).
import math
import numpy as np
from PIL import Image,ImageFilter

CHAN_ORDER=['hs','slrm','slope','rough','slrm15','curv']  # ordinea canonică; CHANS selectează subset în ordinea asta
# slrm15 (10.07.2026, direcția lui Andrei „întărim filtrul SLRM"): SLRM cu rază 15m — accentuează
# movilele MICI (golul de recall documentat). Canal ADITIV la final: subseturile 4ch existente = neschimbate.
# curv (15.07.2026, direcția lui Andrei „îmbunătățim canalul de curbură"): CURBURĂ PRINCIPALĂ adevărată
# (nu proxy ca slrm). domeness = -kmax al elevației netezite ~6m: dom ROTUND = convex în TOATE direcțiile
# (-kmax mare); creastă LINIARĂ = convex pe o direcție (-kmax ~0). Atacă modul de eșec din GradCAM (maluri
# liniare confundate cu movile). Poartă trecută: AUC domeness real-vs-FP 0,70 (0,79 pe subsetul CNN≥0,9).
# ADITIV la final: subseturile 4ch existente = NESCHIMBATE (producția cere doar hs,slrm,slope,rough).

def hs(dem,cs,azs=(315,45,135,225,270,0),alt=35):
    gy,gx=np.gradient(dem,cs);sl=np.arctan(np.hypot(gx,gy));asp=np.arctan2(-gy,gx);o=np.zeros_like(dem);ar=math.radians(alt)
    for az in azs: azr=math.radians(360-az+90);o+=np.clip(np.sin(ar)*np.cos(sl)+np.cos(ar)*np.sin(sl)*np.cos(azr-asp),0,1)
    return o/len(azs)
def downs(a,fc):
    Hh,Ww=a.shape;return a[:Hh//fc*fc,:Ww//fc*fc].reshape(Hh//fc,fc,Ww//fc,fc).mean((1,3))
def homog(a):
    a2=np.asarray(Image.fromarray(a).filter(ImageFilter.GaussianBlur(0.8)),np.uint8);cdf=np.bincount(a2.ravel(),minlength=256).astype(np.float64).cumsum()
    return (cdf[a2]/cdf[-1]*255).astype(np.uint8) if cdf[-1]>0 else a2
def boxblur1(a,r):
    Hh,Ww=a.shape;ii=np.zeros((Hh+1,Ww+1),np.float64);ii[1:,1:]=np.cumsum(np.cumsum(a,0),1)
    ys=np.arange(Hh);xs=np.arange(Ww);y0=np.clip(ys-r,0,Hh);y1=np.clip(ys+r+1,0,Hh);x0=np.clip(xs-r,0,Ww);x1=np.clip(xs+r+1,0,Ww)
    A=ii[y1][:,x1]-ii[y0][:,x1]-ii[y1][:,x0]+ii[y0][:,x0];cnt=((y1-y0)[:,None]*(x1-x0)[None,:]).astype(np.float64)
    return (A/cnt).astype(np.float32)
def stretch128(field):
    lo,hi=np.percentile(field,2),np.percentile(field,98)
    if hi-lo<1e-9: return None
    u=np.clip((field-lo)/(hi-lo)*255,0,255).astype('uint8')
    return homog(np.asarray(Image.fromarray(u).resize((128,128)),np.uint8))

def _slope(z,cs):
    gy,gx=np.gradient(z,cs);return np.hypot(gx,gy)  # tangenta pantei (adimensional)
def _rough(z,cs):
    # rugozitate = deviația standard locală a elevației într-o fereastră ~10m (rezidual față de medie locală)
    r=max(1,int(5/cs))
    m=boxblur1(z.astype(np.float32),r); m2=boxblur1((z*z).astype(np.float32),r)
    return np.sqrt(np.clip(m2-m*m,0,None))
def _curv(z,cs):
    # domeness = -kmax (curbura principală maximă) a elevației netezite la scara movilei (~6m).
    # dom rotund -> -kmax mare (convex în toate direcțiile); creastă liniară -> -kmax ~0. Invariant la iluminare.
    zs=boxblur1(z.astype(np.float32),max(1,int(6/cs)))
    Zy,Zx=np.gradient(zs,cs); Zyy,_=np.gradient(Zy,cs); Zxy,Zxx=np.gradient(Zx,cs)
    mean=(Zxx+Zyy)/2.0; disc=np.sqrt(np.clip(((Zxx-Zyy)/2.0)**2+Zxy**2,0,None))
    return -(mean+disc)

def compute_all(dem,native_cs,chans=None):
    """DEM (float, cu NaN deja umplut) + rezoluția nativă (m/px) -> dict canal->uint8(128,128).
    Downsample la 2m efectiv (ca RO 0.5m->2m și DK 0.4m->2m). `chans`=None -> toate;
    altfel calculează DOAR canalele cerute (lazy — consumatorii 4ch nu plătesc pt slrm15)."""
    f=max(1,int(round(2.0/native_cs)))
    z=downs(dem,f); cs2=native_cs*f
    br=int(30/cs2)
    want=set(chans) if chans else set(CHAN_ORDER)
    out={}
    if 'hs' in want: out['hs']=stretch128(hs(z,cs2))
    if 'slrm' in want: out['slrm']=stretch128(z-boxblur1(boxblur1(boxblur1(z,br),br),br))
    if 'slope' in want: out['slope']=stretch128(_slope(z,cs2))
    if 'rough' in want: out['rough']=stretch128(_rough(z,cs2))
    if 'slrm15' in want:
        b15=max(1,int(15/cs2))
        out['slrm15']=stretch128(z-boxblur1(boxblur1(boxblur1(z,b15),b15),b15))
    if 'curv' in want: out['curv']=stretch128(_curv(z,cs2))
    return out

def stamp_multi(dem,native_cs,chans):
    """-> uint8 (C,128,128) în ordinea `chans` (listă din CHAN_ORDER), sau None dacă vreun canal e degenerat."""
    allc=compute_all(dem,native_cs,chans)
    sel=[allc[c] for c in chans]
    if any(s is None for s in sel): return None
    return np.stack(sel)
