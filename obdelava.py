import cv2
import numpy as np

def analiziraj_sliko(slika_rgb):
    """Funkcija za obdelavo in analizo vodoobčutljivega lističa"""
    
    # --- KORAK 1: PREPROST IN NARAVEN IZREZ (BREZ DEFORMACIJ) ---
    hsv_orig = cv2.cvtColor(slika_rgb, cv2.COLOR_RGB2HSV)
    spodnja_rumena = np.array([15, 80, 80])
    zgornja_rumena = np.array([45, 255, 255])
    maska_rumene = cv2.inRange(hsv_orig, spodnja_rumena, zgornja_rumena)
    
    # Blago čiščenje robu maske
    jedro = np.ones((5, 5), np.uint8)
    maska_rumene = cv2.morphologyEx(maska_rumene, cv2.MORPH_OPEN, jedro)
    maska_rumene = cv2.morphologyEx(maska_rumene, cv2.MORPH_CLOSE, jedro)
    
    konture, _ = cv2.findContours(maska_rumene, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not konture:
        return "NAPAKA", "Napaka: Na sliki ni rumenega lističa.", None, None, None
        
    najvecja_kontura = max(konture, key=cv2.contourArea)
    
    # Ustvarimo natančno masko lističa po njegovi DEJANSKI konturi (da kasneje ignoriramo ozadje)
    maska_papirja_celotna = np.zeros(slika_rgb.shape[:2], dtype=np.uint8)
    cv2.drawContours(maska_papirja_celotna, [najvecja_kontura], -1, 255, -1)
    
    # Uporabimo RAVNI obdajajoči pravokotnik (brez vsiljevanja kotov ali preslikav!)
    x, y, w, h = cv2.boundingRect(najvecja_kontura)
    
    # Sliko in masko izrežemo kot z navadnimi škarjami - ohranimo 100% originalna razmerja
    slika_izrez = slika_rgb[y:y+h, x:x+w].copy()
    maska_izrez = maska_papirja_celotna[y:y+h, x:x+w].copy()
    
    # Poskrbimo, da je listič vedno ležeč za nadaljnjo obdelavo
    if h > w:
        slika_izrez = cv2.rotate(slika_izrez, cv2.ROTATE_90_CLOCKWISE)
        maska_izrez = cv2.rotate(maska_izrez, cv2.ROTATE_90_CLOCKWISE)
        # Zamenjamo w in h spremenljivki
        temp = w
        w = h
        h = temp
        
    # --- SKALIRANJE (Fiksna višina 76mm, širina se prilagaja) ---
    pike_na_mm_cilj = 60.0
    visina_mm = 76.0
    ciljna_visina_px = int(visina_mm * pike_na_mm_cilj)
    
    # Širino izračunamo strogo proporcionalno
    razmerje = ciljna_visina_px / h
    ciljna_sirina_px = int(w * razmerje)
    
    slika_skalirana = cv2.resize(slika_izrez, (ciljna_sirina_px, ciljna_visina_px), interpolation=cv2.INTER_CUBIC)
    maska_skalirana = cv2.resize(maska_izrez, (ciljna_sirina_px, ciljna_visina_px), interpolation=cv2.INTER_NEAREST)
    
    um_na_piko = 1000 / pike_na_mm_cilj
    um2_na_piko = um_na_piko ** 2
    
    # --- KORAK 2: DETEKCIJA KAPLJIC ---
    siva = cv2.cvtColor(slika_skalirana, cv2.COLOR_RGB2GRAY)
    
    # Povečanje lokalnega kontrasta (CLAHE) za izpostavitev bledih kapljic
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    siva_clahe = clahe.apply(siva)
    
    # Natančnejši adaptivni prag (manjši blok, manjša C konstanta za večjo občutljivost)
    maska_kontrasta = cv2.adaptiveThreshold(
        siva_clahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 15, 4
    )
    
    # Ohranjanje le tistih pik, ki so dejansko temnejše (filtracija šuma na svetlih delih)
    maska_temnih = cv2.bitwise_and(maska_kontrasta, cv2.bitwise_not(cv2.inRange(siva_clahe, 185, 255)))
    
    hsv_izrez = cv2.cvtColor(slika_skalirana, cv2.COLOR_RGB2HSV)
    # Sužena HSV definicija specifično za modre/vijolične WSP odtise
    spodnja_modra = np.array([85, 30, 30])
    zgornja_modra = np.array([170, 255, 255])
    maska_hsv = cv2.inRange(hsv_izrez, spodnja_modra, zgornja_modra)
    
    maska_kapljic = cv2.bitwise_or(maska_temnih, maska_hsv)
    
    # KLJUČNO: Omejimo detekcijo strogo samo na rumeni del (masko konture lističa), da ne štejemo mize v kotih!
    maska_kapljic = cv2.bitwise_and(maska_kapljic, maska_skalirana)
    
    # --- KORAK 3: ČIŠČENJE ---
    # Uporaba manjšega jedra (3x3 namesto 5x5), da se ohranijo drobne kapljice
    jedro_zapiranje = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    maska_sklenjena = cv2.morphologyEx(maska_kapljic, cv2.MORPH_CLOSE, jedro_zapiranje)
    
    konture_maske, _ = cv2.findContours(maska_sklenjena, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    maska_zapolnjena = np.zeros_like(maska_sklenjena)
    for kontura in konture_maske:
        lupina = cv2.convexHull(kontura)
        cv2.drawContours(maska_zapolnjena, [lupina], -1, 255, -1)
        
    jedro_odpiranje = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    maska_cista = cv2.morphologyEx(maska_zapolnjena, cv2.MORPH_OPEN, jedro_odpiranje)
    
    # --- KORAK 4: NOVI IZRAČUNI ---
    koncne_konture, _ = cv2.findContours(maska_cista, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    stevilo_odtisov = 0
    maska_filtrirana = np.zeros_like(maska_cista)
    
    for kontura in koncne_konture:
        povrsina_px = cv2.contourArea(kontura)
        povrsina_um2 = povrsina_px * um2_na_piko
        
        if povrsina_um2 > 0:
            premer_um = 2 * np.sqrt(povrsina_um2 / np.pi)
            
            # Detektiramo samo kapljice nad 50 mikrometrov
            if premer_um >= 50:
                stevilo_odtisov += 1
                cv2.drawContours(maska_filtrirana, [kontura], -1, 255, -1)
                
    # Natančen izračun pokritosti: piksli kapljic / piksli DEJANSKEGA lističa (brez ozadja) * 100
    piksli_papirja = cv2.countNonZero(maska_skalirana)
    piksli_kapljic = cv2.countNonZero(maska_filtrirana)
    
    pokritost_odstotek = (piksli_kapljic / piksli_papirja) * 100 if piksli_papirja > 0 else 0
    
    # Za lepši prikaz v aplikaciji zatemnimo morebitno mizo v kotih izreza
    slika_prikaz = cv2.bitwise_and(slika_skalirana, slika_skalirana, mask=maska_skalirana)
    
    return "USPEH", slika_prikaz, maska_filtrirana, stevilo_odtisov, pokritost_odstotek
