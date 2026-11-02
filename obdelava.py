import cv2
import numpy as np
import random

def uredi_tocke(tocke):
    """Razvrsti 4 točke v vrstni red: zgoraj-levo, zgoraj-desno, spodaj-desno, spodaj-levo"""
    tocke = tocke.reshape((4, 2))
    nove_tocke = np.zeros((4, 2), dtype=np.float32)
    
    vsota = tocke.sum(axis=1)
    nove_tocke[0] = tocke[np.argmin(vsota)] # Zgoraj levo
    nove_tocke[2] = tocke[np.argmax(vsota)] # Spodaj desno
    
    razlika = np.diff(tocke, axis=1)
    nove_tocke[1] = tocke[np.argmin(razlika)] # Zgoraj desno
    nove_tocke[3] = tocke[np.argmax(razlika)] # Spodaj levo
    
    return nove_tocke

def analiziraj_sliko(slika_rgb):
    """Brezšumna različica: enojna preslikava brez artefaktov in izboljšano glajenje teksture."""
    
    # --- KORAK 1: DETEKCIJA LISTIČA ---
    hsv_orig = cv2.cvtColor(slika_rgb, cv2.COLOR_RGB2HSV)
    spodnja_rumena = np.array([15, 80, 80])
    zgornja_rumena = np.array([45, 255, 255])
    maska_rumene = cv2.inRange(hsv_orig, spodnja_rumena, zgornja_rumena)
    
    jedro = np.ones((5, 5), np.uint8)
    maska_rumene = cv2.morphologyEx(maska_rumene, cv2.MORPH_OPEN, jedro)
    maska_rumene = cv2.morphologyEx(maska_rumene, cv2.MORPH_CLOSE, jedro)
    
    konture, _ = cv2.findContours(maska_rumene, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not konture:
        return "NAPAKA", "Napaka: Na sliki ni rumenega lističa.", None, None
        
    najvecja_kontura = max(konture, key=cv2.contourArea)
    maska_papirja_celotna = np.zeros(slika_rgb.shape[:2], dtype=np.uint8)
    cv2.drawContours(maska_papirja_celotna, [najvecja_kontura], -1, 255, -1)
    
    # --- KORAK 2: DIREKTNO RAVNANJE IN SKALIRANJE (Brez dvojne interpolacije!) ---
    pravokotnik = cv2.minAreaRect(najvecja_kontura)
    tocke_pravokotnika = cv2.boxPoints(pravokotnik)
    tocke_pravokotnika = np.int32(tocke_pravokotnika)
    
    urejene_tocke = uredi_tocke(tocke_pravokotnika)
    (tl, tr, br, bl) = urejene_tocke
    
    sirina_px_orig = max(int(np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))), 
                         int(np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))))
    visina_px_orig = max(int(np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))), 
                         int(np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))))
                         
    pike_na_mm_cilj = 60.0
    daljsa_stranica_mm = 76.0
    ciljna_dolzina_px = int(daljsa_stranica_mm * pike_na_mm_cilj)
    
    if visina_px_orig > sirina_px_orig:
        ciljna_visina_px = ciljna_dolzina_px
        razmerje = sirina_px_orig / visina_px_orig if visina_px_orig > 0 else 1
        ciljna_sirina_px = int(ciljna_visina_px * razmerje)
    else:
        ciljna_sirina_px = ciljna_dolzina_px
        razmerje = visina_px_orig / sirina_px_orig if sirina_px_orig > 0 else 1
        ciljna_visina_px = int(ciljna_sirina_px * razmerje)
        
    ciljne_tocke = np.array([
        [0, 0],
        [ciljna_sirina_px - 1, 0],
        [ciljna_sirina_px - 1, ciljna_visina_px - 1],
        [0, ciljna_visina_px - 1]
    ], dtype="float32")
    
    # Preslikava in skaliranje se izvedeta sočasno z mehkim INTER_LINEAR, da ni ostrih artefaktov
    matrika_preslikave = cv2.getPerspectiveTransform(urejene_tocke, ciljne_tocke)
    slika_skalirana = cv2.warpPerspective(slika_rgb, matrika_preslikave, (ciljna_sirina_px, ciljna_visina_px), flags=cv2.INTER_LINEAR)
    maska_skalirana = cv2.warpPerspective(maska_papirja_celotna, matrika_preslikave, (ciljna_sirina_px, ciljna_visina_px), flags=cv2.INTER_NEAREST)
    _, maska_skalirana = cv2.threshold(maska_skalirana, 127, 255, cv2.THRESH_BINARY)
    
    # Prisilna vodoravna orientacija
    if ciljna_visina_px > ciljna_sirina_px:
        slika_skalirana = cv2.rotate(slika_skalirana, cv2.ROTATE_90_CLOCKWISE)
        maska_skalirana = cv2.rotate(maska_skalirana, cv2.ROTATE_90_CLOCKWISE)
        temp = ciljna_sirina_px
        ciljna_sirina_px = ciljna_visina_px
        ciljna_visina_px = temp
        
    um_na_piko = 1000 / pike_na_mm_cilj
    um2_na_piko = um_na_piko ** 2
    
    # --- KORAK 3: DETEKCIJA S ČIŠČENJEM TEKSTURE PAPIRJA ---
    siva = cv2.cvtColor(slika_skalirana, cv2.COLOR_RGB2GRAY)
    
    # Povečan Median filter z(3 na 5), ki popolnoma izniči drobno papirnato teksturo
    siva_zglad = cv2.medianBlur(siva, 5) 
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    siva_clahe = clahe.apply(siva_zglad)
    
    # Zvišan C parameter (iz 4 na 5), da algoritem postane odpornejši na sence
    maska_kontrasta = cv2.adaptiveThreshold(
        siva_clahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 15, 5 
    )
    maska_temnih = cv2.bitwise_and(maska_kontrasta, cv2.bitwise_not(cv2.inRange(siva_clahe, 185, 255)))
    
    hsv_izrez = cv2.cvtColor(slika_skalirana, cv2.COLOR_RGB2HSV)
    spodnja_modra = np.array([85, 30, 30])
    zgornja_modra = np.array([170, 255, 255])
    maska_hsv = cv2.inRange(hsv_izrez, spodnja_modra, zgornja_modra)
    
    maska_kapljic = cv2.bitwise_or(maska_temnih, maska_hsv)
    maska_kapljic = cv2.bitwise_and(maska_kapljic, maska_skalirana)
    
    # Čiščenje
    jedro_zapiranje = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    maska_sklenjena = cv2.morphologyEx(maska_kapljic, cv2.MORPH_CLOSE, jedro_zapiranje)
    
    konture_mask
