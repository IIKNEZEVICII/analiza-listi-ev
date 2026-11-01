import cv2
import numpy as np

def analiziraj_sliko(slika_rgb):
    """Funkcija za obdelavo in analizo vodoobčutljivega lističa"""
    
    # --- KORAK 1: PREPROST IN NARAVEN IZREZ (BREZ DEFORMACIJ) ---
    hsv_orig = cv2.cvtColor(slika_rgb, cv2.COLOR_RGB2HSV)
    spodnja_rumena = np.array([15, 80, 80])
    zgornja_rumena = np.array([45, 255, 255])
    maska_rumene = cv2.inRange(hsv_orig, spodnja_rumena, zgornja_rumena)
    
    jedro = np.ones((5, 5), np.uint8)
    maska_rumene = cv2.morphologyEx(maska_rumene, cv2.MORPH_OPEN, jedro)
    maska_rumene = cv2.morphologyEx(maska_rumene, cv2.MORPH_CLOSE, jedro)
    
    konture, _ = cv2.findContours(maska_rumene, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not konture:
        return "NAPAKA", "Napaka: Na sliki ni rumenega lističa.", None, None, None
        
    najvecja_kontura = max(konture, key=cv2.contourArea)
    
    maska_papirja_celotna = np.zeros(slika_rgb.shape[:2], dtype=np.uint8)
    cv2.drawContours(maska_papirja_celotna, [najvecja_kontura], -1, 255, -1)
    
    x, y, w, h = cv2.boundingRect(najvecja_kontura)
    
    slika_izrez = slika_rgb[y:y+h, x:x+w].copy()
    maska_izrez = maska_papirja_celotna[y:y+h, x:x+w].copy()
    
    if h > w:
        slika_izrez = cv2.rotate(slika_izrez, cv2.ROTATE_90_CLOCKWISE)
        maska_izrez = cv2.rotate(maska_izrez, cv2.ROTATE_90_CLOCKWISE)
        temp = w
        w = h
        h = temp
        
    # --- SKALIRANJE ---
    pike_na_mm_cilj = 60.0
    visina_mm = 76.0
    ciljna_visina_px = int(visina_mm * pike_na_mm_cilj)
    
    razmerje = ciljna_visina_px / h
    ciljna_sirina_px = int(w * razmerje)
    
    slika_skalirana = cv2.resize(slika_izrez, (ciljna_sirina_px, ciljna_visina_px), interpolation=cv2.INTER_CUBIC)
    maska_skalirana = cv2.resize(maska_izrez, (ciljna_sirina_px, ciljna_visina_px), interpolation=cv2.INTER_NEAREST)
    
    um_na_piko = 1000 / pike_na_mm_cilj
    um2_na_piko = um_na_piko ** 2
    
    # --- KORAK 2: TROSLOJNA DETEKCIJA KAPLJIC ---
    siva = cv2.cvtColor(slika_skalirana, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    siva_clahe = clahe.apply(siva)
    
    # Maska 1: Globalni prag (najboljši za VELIKE in trdno obarvane kapljice)
    _, maska_globalna = cv2.threshold(siva_clahe, 135, 255, cv2.THRESH_BINARY_INV)
    
    # Maska 2: Adaptivni prag s povečanim območjem iskanja (blok 31 namesto 17), 
    # da ne ignorira velikih kapljic, ujame pa drobne in blede robove.
    maska_adapt = cv2.adaptiveThreshold(
        siva_clahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 31, 4
    )
    # Zavarujemo se pred svetlim šumom
    maska_adapt = cv2.bitwise_and(maska_adapt, cv2.bitwise_not(cv2.inRange(siva_clahe, 185, 255)))
    
    # Maska 3: HSV barvni spekter (za izrazito modre odtise)
    hsv_izrez = cv2.cvtColor(slika_skalirana, cv2.COLOR_RGB2HSV)
    spodnja_modra = np.array([75, 15, 20])
    zgornja_modra = np.array([170, 255, 255])
    maska_hsv = cv2.inRange(hsv_izrez, spodnja_modra, zgornja_modra)
    
    # ZDRUŽITEV VSEH TREH MASK: Kar ena zgreši, druga popravi!
    maska_kapljic = cv2.bitwise_or(maska_globalna, maska_adapt)
    maska_kapljic = cv2.bitwise_or(maska_kapljic, maska_hsv)
    
    # Obrežemo iskanje z zaščitnim robom navznoter, da ignoriramo mizo
    maska_za_iskanje = cv2.erode(maska_skalirana, np.ones((3, 3), np.uint8))
    maska_kapljic = cv2.bitwise_and(maska_kapljic, maska_za_iskanje)
    
    # --- KORAK 3: ZAPOLNJEVANJE IN ČIŠČENJE ---
    # Uporabimo večje jedro za zapiranje (5x5), ki bo popolnoma zapolnilo morebitne 
    # luknje znotraj VELIKIH kapljic in jih povezalo v celoto.
    jedro_zapiranje = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    maska_sklenjena = cv2.morphologyEx(maska_kapljic, cv2.MORPH_CLOSE, jedro_zapiranje)
    
    konture_maske, _ = cv2.findContours(maska_sklenjena, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    maska_zapolnjena = np.zeros_like(maska_sklenjena)
    for kontura in konture_maske:
        lupina = cv2.convexHull(kontura)
        cv2.drawContours(maska_zapolnjena, [lupina], -1, 255, -1)
        
    # Nežno čiščenje mikroskopske umazanije papirja
    jedro_odpiranje = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    maska_cista = cv2.morphologyEx(maska_zapolnjena, cv2.MORPH_OPEN, jedro_odpiranje)
    
    # --- KORAK 4: IZRAČUNI ---
    koncne_konture, _ = cv2.findContours(maska_cista, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    stevilo_odtisov = 0
    maska_filtrirana = np.zeros_like(maska_cista)
    
    for kontura in koncne_konture:
        povrsina_px = cv2.contourArea(kontura)
        povrsina_um2 = povrsina_px * um2_na_piko
        
        if povrsina_um2 > 0:
            premer_um = 2 * np.sqrt(povrsina_um2 / np.pi)
            
            # Štejemo samo kapljice >= 50 µm
            if premer_um >= 50:
                stevilo_odtisov += 1
                cv2.drawContours(maska_filtrirana, [kontura], -1, 255, -1)
                
    # Natančen izračun
    piksli_papirja = cv2.countNonZero(maska_za_iskanje)
    piksli_kapljic = cv2.countNonZero(maska_filtrirana)
    
    pokritost_odstotek = (piksli_kapljic / piksli_papirja) * 100 if piksli_papirja > 0 else 0
    
    # Prikaz - pobelimo območje zunaj lističa
    slika_prikaz = cv2.bitwise_and(slika_skalirana, slika_skalirana, mask=maska_skalirana)
    
    return "USPEH", slika_prikaz, maska_filtrirana, stevilo_odtisov, pokritost_odstotek
