import cv2
import numpy as np

def analiziraj_sliko(slika_rgb):
    """Funkcija za obdelavo in analizo vodoobčutljivega lističa"""
    
    # --- KORAK 1: AVTOMATSKI IZREZ IN MASKA ---
    hsv_orig = cv2.cvtColor(slika_rgb, cv2.COLOR_RGB2HSV) # HSV (Hue, Saturation, Value)
    spodnja_rumena = np.array([15, 80, 80])
    zgornja_rumena = np.array([45, 255, 255])
    maska_rumene = cv2.inRange(hsv_orig, spodnja_rumena, zgornja_rumena)
    
    # Zmanjšanje šuma na originalni maski rumenega lističa za lepši rob
    jedro = np.ones((5,5), np.uint8)
    maska_rumene = cv2.morphologyEx(maska_rumene, cv2.MORPH_OPEN, jedro)
    maska_rumene = cv2.morphologyEx(maska_rumene, cv2.MORPH_CLOSE, jedro)
    
    konture, _ = cv2.findContours(maska_rumene, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not konture:
        return "NAPAKA", "Napaka: Na sliki ni rumenega lističa.", None, None, None
        
    najvecja_kontura = max(konture, key=cv2.contourArea)
    
    # Narišemo točno masko nepravilne oblike papirja (da izločimo ozadje pri natrganih robovih)
    maska_papirja_celotna = np.zeros(slika_rgb.shape[:2], dtype=np.uint8)
    cv2.drawContours(maska_papirja_celotna, [najvecja_kontura], -1, 255, -1)
    
    # Izračun optimalnega pravokotnika
    rect = cv2.minAreaRect(najvecja_kontura)
    box = cv2.boxPoints(rect)
    tocke = np.float32(box)
    
    # Dolžini stranic iz minAreaRect
    d1 = np.linalg.norm(tocke[0] - tocke[1])
    d2 = np.linalg.norm(tocke[1] - tocke[2])
    
    # Preslikava z ohranjanjem zaporedja točk (popolnoma prepreči deformacije in "mečkanje")
    ciljne_tocke = np.float32([
        [0, d2 - 1],
        [0, 0],
        [d1 - 1, 0],
        [d1 - 1, d2 - 1]
    ])
    
    matrika = cv2.getPerspectiveTransform(tocke, ciljne_tocke)
    slika_osnovna = cv2.warpPerspective(slika_rgb, matrika, (int(d1), int(d2)))
    maska_osnovna = cv2.warpPerspective(maska_papirja_celotna, matrika, (int(d1), int(d2)))
    
    # Postavimo sliko pokončno, da je daljša stranica vedno višina
    h_px, w_px = slika_osnovna.shape[:2]
    if h_px < w_px:
        slika_osnovna = cv2.rotate(slika_osnovna, cv2.ROTATE_90_CLOCKWISE)
        maska_osnovna = cv2.rotate(maska_osnovna, cv2.ROTATE_90_CLOCKWISE)
        h_px, w_px = slika_osnovna.shape[:2]
        
    # --- KORAK 2: SKALIRANJE NA 76mm ---
    pike_na_mm_cilj = 60.0
    ciljna_visina = int(76.0 * pike_na_mm_cilj) 
    razmerje = ciljna_visina / h_px
    ciljna_sirina = int(w_px * razmerje) # Širina se sorazmerno prilagodi
    
    slika_izrez = cv2.resize(slika_osnovna, (ciljna_sirina, ciljna_visina), interpolation=cv2.INTER_CUBIC)
    maska_izrez = cv2.resize(maska_osnovna, (ciljna_sirina, ciljna_visina), interpolation=cv2.INTER_NEAREST)
    
    # Obrnemo ležeče za preglednejši končni prikaz v aplikaciji
    slika_izrez = cv2.rotate(slika_izrez, cv2.ROTATE_90_CLOCKWISE)
    maska_izrez = cv2.rotate(maska_izrez, cv2.ROTATE_90_CLOCKWISE)
    
    # Odstranitev robov, da se znebimo artefaktov preslikave
    slika_izrez = slika_izrez[5:-5, 5:-5]
    maska_izrez = maska_izrez[5:-5, 5:-5]
    
    um_na_piko = 1000 / pike_na_mm_cilj
    um2_na_piko = um_na_piko ** 2
    
    # --- KORAK 3: DETEKCIJA KAPLJIC ---
    siva = cv2.cvtColor(slika_izrez, cv2.COLOR_RGB2GRAY)
    maska_kontrasta = cv2.adaptiveThreshold(
        siva, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 21, 7
    )
    maska_temnih = cv2.bitwise_and(maska_kontrasta, cv2.bitwise_not(cv2.inRange(siva, 175, 255)))
    
    hsv_izrez = cv2.cvtColor(slika_izrez, cv2.COLOR_RGB2HSV)
    spodnja_modra = np.array([75, 12, 12])
    zgornja_modra = np.array([165, 255, 255])
    maska_hsv_kapljice = cv2.inRange(hsv_izrez, spodnja_modra, zgornja_modra)
    
    maska_kapljic = cv2.bitwise_or(maska_temnih, maska_hsv_kapljice)
    
    # KLJUČNO: Omejimo iskanje kapljic strogo le na dejansko površino lističa, da preprečimo branje ozadja!
    maska_kapljic = cv2.bitwise_and(maska_kapljic, maska_izrez)
    
    # Čiščenje šuma v maski kapljic
    jedro_zapiranje = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    maska_sklenjena = cv2.morphologyEx(maska_kapljic, cv2.MORPH_CLOSE, jedro_zapiranje)
    
    konture_maske, _ = cv2.findContours(maska_sklenjena, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    maska_zapolnjena = np.zeros_like(maska_sklenjena)
    for kontura in konture_maske:
        lupina = cv2.convexHull(kontura)
        cv2.drawContours(maska_zapolnjena, [lupina], -1, 255, -1)
            
    jedro_odpiranje = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
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
            if premer_um >= 50: # Detekcija le nad 50 µm
                stevilo_odtisov += 1
                cv2.drawContours(maska_filtrirana, [kontura], -1, 255, -1)
                
    # Natančen izračun pokritosti (št. pikslov kapljic deli s številom pikslov DEJANSKEGA rumenega papirja)
    piksli_papirja = cv2.countNonZero(maska_izrez)
    piksli_kapljic = cv2.countNonZero(maska_filtrirana)
    
    pokritost_odstotek = (piksli_kapljic / piksli_papirja) * 100 if piksli_papirja > 0 else 0
    
    return "USPEH", slika_izrez, maska_filtrirana, stevilo_odtisov, pokritost_odstotek
