import cv2
import numpy as np

def analiziraj_sliko(slika_rgb):
    # --- 1. ISKANJE KONTURE ---
    hsv = cv2.cvtColor(slika_rgb, cv2.COLOR_RGB2HSV)
    # Razpon, ki ujame tudi svetlejše/temnejše dele rumenega lističa
    spodnja_rumena = np.array([10, 50, 50])
    zgornja_rumena = np.array([50, 255, 255])
    maska_rumene = cv2.inRange(hsv, spodnja_rumena, zgornja_rumena)
    
    # Zapolnitev lukenj (kjer je veliko kapljic) in čiščenje smeti
    maska_rumene = cv2.morphologyEx(maska_rumene, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    maska_rumene = cv2.morphologyEx(maska_rumene, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    
    konture, _ = cv2.findContours(maska_rumene, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not konture:
        return "NAPAKA", "Napaka: Na sliki ni rumenega lističa.", None, None, None
        
    najvecja_kontura = max(konture, key=cv2.contourArea)
    
    # --- 2. NATANČEN IZREZ BREZ DEFORMACIJ ---
    x, y, w, h = cv2.boundingRect(najvecja_kontura)
    slika_izrez = slika_rgb[y:y+h, x:x+w].copy()
    
    # Ustvarimo masko za točno (neravno) obliko lističa
    maska_izrez = np.zeros((h, w), dtype=np.uint8)
    zamaknjena_kontura = najvecja_kontura - [x, y]
    cv2.drawContours(maska_izrez, [zamaknjena_kontura], -1, 255, -1)
    
    # Ozadje zunaj lističa pobarvamo na čisto belo
    slika_izrez[maska_izrez == 0] = (255, 255, 255)
    
    # Obračanje v ležeč format za preglednost v aplikaciji
    if h > w:
        slika_izrez = cv2.rotate(slika_izrez, cv2.ROTATE_90_CLOCKWISE)
        maska_izrez = cv2.rotate(maska_izrez, cv2.ROTATE_90_CLOCKWISE)
        h, w = slika_izrez.shape[:2]
        
    # --- 3. PRAVILNO SKALIRANJE NA 76 mm ---
    # S pomočjo minAreaRect dobimo dejansko dolžino papirja (ne glede na nagib na sliki)
    rect = cv2.minAreaRect(najvecja_kontura)
    dejanska_dolzina_px = max(rect[1][0], rect[1][1])
    
    pike_na_mm = 60.0
    ciljna_dolzina_px = 76.0 * pike_na_mm
    
    # Izračunamo skupni faktor povečave in skaliramo sliko (ohranimo vsa naravna razmerja!)
    faktor = ciljna_dolzina_px / dejanska_dolzina_px
    nova_sirina = int(w * faktor)
    nova_visina = int(h * faktor)
    
    slika_skalirana = cv2.resize(slika_izrez, (nova_sirina, nova_visina), interpolation=cv2.INTER_CUBIC)
    maska_skalirana = cv2.resize(maska_izrez, (nova_sirina, nova_visina), interpolation=cv2.INTER_NEAREST)
    
    um2_na_piko = (1000 / pike_na_mm) ** 2
    
    # --- 4. DETEKCIJA KAPLJIC ---
    siva = cv2.cvtColor(slika_skalirana, cv2.COLOR_RGB2GRAY)
    maska_kontrasta = cv2.adaptiveThreshold(siva, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 7)
    maska_temnih = cv2.bitwise_and(maska_kontrasta, cv2.bitwise_not(cv2.inRange(siva, 175, 255)))
    
    hsv_sk = cv2.cvtColor(slika_skalirana, cv2.COLOR_RGB2HSV)
    maska_hsv = cv2.inRange(hsv_sk, np.array([75, 12, 12]), np.array([165, 255, 255]))
    
    maska_kapljic = cv2.bitwise_or(maska_temnih, maska_hsv)
    
    # KLJUČNI POPRAVEK: Masko lističa skrčimo za 25 pikslov (cca 0.2 mm), da odrežemo ostri rob, 
    # ki ga algoritem zmotno prepozna kot kapljice!
    maska_notranjosti = cv2.erode(maska_skalirana, np.ones((25, 25), np.uint8))
    maska_kapljic = cv2.bitwise_and(maska_kapljic, maska_notranjosti)
    
    # --- 5. ČIŠČENJE IN IZRAČUNI ---
    maska_kapljic = cv2.morphologyEx(maska_kapljic, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    
    konture_kapljic, _ = cv2.findContours(maska_kapljic, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    maska_zapolnjena = np.zeros_like(maska_kapljic)
    for k in konture_kapljic:
        cv2.drawContours(maska_zapolnjena, [cv2.convexHull(k)], -1, 255, -1)
        
    maska_cista = cv2.morphologyEx(maska_zapolnjena, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    
    koncne_konture, _ = cv2.findContours(maska_cista, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    stevilo_odtisov = 0
    maska_filtrirana = np.zeros_like(maska_cista)
    
    for k in koncne_konture:
        povrsina_px = cv2.contourArea(k)
        povrsina_um2 = povrsina_px * um2_na_piko
        
        # Filtiramo samo odtise >= 50 µm
        if povrsina_um2 > 0:
            premer_um = 2 * np.sqrt(povrsina_um2 / np.pi)
            if premer_um >= 50:
                stevilo_odtisov += 1
                cv2.drawContours(maska_filtrirana, [k], -1, 255, -1)
                
    # Izračunamo natančen odstotek na podlagi pikslov lističa (brez ozadja in brez problematičnega zunanjega roba)
    piksli_papirja = cv2.countNonZero(maska_notranjosti)
    piksli_kapljic = cv2.countNonZero(maska_filtrirana)
    
    pokritost_odstotek = (piksli_kapljic / piksli_papirja) * 100 if piksli_papirja > 0 else 0
    
    return "USPEH", slika_skalirana, maska_filtrirana, stevilo_odtisov, pokritost_odstotek
