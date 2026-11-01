import cv2
import numpy as np

def analiziraj_sliko(slika_rgb):
    # --- 1. ISKANJE RUMENEGA LISTIČA ---
    hsv = cv2.cvtColor(slika_rgb, cv2.COLOR_RGB2HSV)
    spodnja_rumena = np.array([15, 80, 80])
    zgornja_rumena = np.array([45, 255, 255])
    maska_rumene = cv2.inRange(hsv, spodnja_rumena, zgornja_rumena)
    
    # Odstranitev drobnih smeti okoli robu
    jedro = np.ones((5, 5), np.uint8)
    maska_rumene = cv2.morphologyEx(maska_rumene, cv2.MORPH_OPEN, jedro)
    maska_rumene = cv2.morphologyEx(maska_rumene, cv2.MORPH_CLOSE, jedro)
    
    konture, _ = cv2.findContours(maska_rumene, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not konture:
        return "NAPAKA", "Napaka: Na sliki ni rumenega lističa.", None, None, None
        
    najvecja_kontura = max(konture, key=cv2.contourArea)
    
    # --- 2. PREPROST IZREZ (KOT S ŠKARJAMI NA BELO OZADJE) ---
    # Dobimo navaden kvadratni okvir okoli lističa, brez vrtenja in popačenj!
    x, y, w, h = cv2.boundingRect(najvecja_kontura)
    
    # Izrežemo sliko znotraj tega okvirja
    slika_izrez = slika_rgb[y:y+h, x:x+w].copy()
    
    # Naredimo masko točne oblike lističa in jo tudi izrežemo
    maska_naravna = np.zeros(slika_rgb.shape[:2], dtype=np.uint8)
    cv2.drawContours(maska_naravna, [najvecja_kontura], -1, 255, -1)
    maska_izrez = maska_naravna[y:y+h, x:x+w].copy()
    
    # Vse, kar je izven neravne konture lističa (ozadje), postane belo
    slika_izrez[maska_izrez == 0] = (255, 255, 255)
    
    # --- 3. ZAVRTIMO V LEŽEČ POLOŽAJ ---
    # Če je listič na sliki bolj pokončen, ga obrnemo ležeče
    if h > w:
        slika_izrez = cv2.rotate(slika_izrez, cv2.ROTATE_90_CLOCKWISE)
        maska_izrez = cv2.rotate(maska_izrez, cv2.ROTATE_90_CLOCKWISE)
        # Posodobimo širino in višino po rotaciji
        h, w = slika_izrez.shape[:2] 
        
    # --- 4. SKALIRANJE NA 76mm ---
    # Zdaj vemo, da je listič ležeč, zato je daljša stranica (širina 'w') = 76 mm
    pike_na_mm_cilj = 60.0
    ciljna_sirina_px = int(76.0 * pike_na_mm_cilj)
    
    # Višino izračunamo strogo sorazmerno, da ohranimo naravno sliko
    razmerje = ciljna_sirina_px / w
    ciljna_visina_px = int(h * razmerje)
    
    slika_skalirana = cv2.resize(slika_izrez, (ciljna_sirina_px, ciljna_visina_px), interpolation=cv2.INTER_CUBIC)
    maska_skalirana = cv2.
