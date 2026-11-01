import cv2
import numpy as np

def analiziraj_sliko(slika_rgb):
    """Funkcija za obdelavo in analizo vodoobčutljivega lističa"""
    
    # --- KORAK 1: AVTOMATSKI IZREZ ---
    hsv_orig = cv2.cvtColor(slika_rgb, cv2.COLOR_RGB2HSV)
    spodnja_rumena = np.array([15, 80, 80])
    zgornja_rumena = np.array([45, 255, 255])
    maska_rumene = cv2.inRange(hsv_orig, spodnja_rumena, zgornja_rumena)
    
    konture, _ = cv2.findContours(maska_rumene, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not konture:
        return "NAPAKA", "Napaka: Na sliki ni rumenega lističa.", None, None, None
        
    najvecja_kontura = max(konture, key=cv2.contourArea)
    
    # UPORABA minAreaRect ZA NEPRAVILNE OBLIKE
    # Obdaja tudi natrgane in nepravilne robove v optimalen pravokotnik
    rect = cv2.minAreaRect(najvecja_kontura)
    box = cv2.boxPoints(rect)
    tocke = np.intp(box)
    
    # Razvrščanje točk (zgoraj-levo, zgoraj-desno, spodaj-desno, spodaj-levo)
    urejene_tocke = np.zeros((4, 2), dtype="float32")
    vsota = tocke.sum(axis=1)
    urejene_tocke[0] = tocke[np.argmin(vsota)]
    urejene_tocke[2] = tocke[np.argmax(vsota)]
    razlika = np.diff(tocke, axis=1)
    urejene_tocke[1] = tocke[np.argmin(razlika)]
    urejene_tocke[3] = tocke[np.argmax(razlika)]
    
    sirina_zgoraj_px = np.linalg.norm(urejene_tocke[0] - urejene_tocke[1])
    visina_levo_px = np.linalg.norm(urejene_tocke[0] - urejene_tocke[3])
    
    # Referenca je fiksna le za daljšo stranico = 76 mm
    pike_na_mm_cilj = 60.0
    
    if visina_levo_px > sirina_zgoraj_px:
        visina_nova = int(76.0 * pike_na_mm_cilj)
        sirina_nova = int((sirina_zgoraj_px / visina_levo_px) * visina_nova)
    else:
        sirina_nova = int(76.0 * pike_na_mm_cilj)
        visina_nova = int((visina_levo_px / sirina_zgoraj_px) * sirina_nova)
        
    ciljne_tocke = np.float32([
        [0, 0],
        [sirina_nova - 1, 0],
        [sirina_nova - 1, visina_nova - 1],
        [0, visina_nova - 1]
    ])
    
    matrika = cv2.getPerspectiveTransform(urejene_tocke, ciljne_tocke)
    slika_izrez = cv2.warpPerspective(slika_rgb, matrika, (sirina_nova, visina_nova))
    slika_izrez = slika_izrez[3:-3, 3:-3] # Odstranimo robove zaradi artefaktov transformacije
    
    # Vedno ležeča postavitev za lažjo analizo in prikaz
    if visina_levo_px > sirina_zgoraj_px:
        slika_izrez = cv2.rotate(slika_izrez, cv2.ROTATE_90_CLOCKWISE)
        
    um_na_piko = 1000 / pike_na_mm_cilj
    um2_na_piko = um_na_piko ** 2
    
    # --- KORAK 2: DETEKCIJA KAPLJIC ---
    siva = cv2.cvtColor(slika_izrez, cv2.COLOR_RGB2GRAY)
    maska_kontrasta = cv2.adaptiveThreshold(
        siva, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 21, 7
    )
    maska_temnih = cv2.bitwise_and(maska_kontrasta, cv2.bitwise_not(cv2.inRange(siva, 175, 255)))
    
    hsv_izrez = cv2.cvtColor(slika_izrez, cv2.COLOR_RGB2HSV)
    spodnja_modra = np.array([75, 12, 12])
    zgornja_modra = np.array([165, 255, 255])
    maska_hsv = cv2.inRange(hsv_izrez, spodnja_modra, zgornja_modra)
    
    maska_kapljic = cv2.bitwise_or(maska_temnih, maska_hsv)
    
    # --- KORAK 3: ČIŠČENJE IN KONVEKSNA LUPINA ---
    jedro_zapiranje = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
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
            
            # POGOJ 50 um: shranimo le dovolj velike kapljice
            if premer_um >= 50:
                stevilo_odtisov += 1
                cv2.drawContours(maska_filtrirana, [kontura], -1, 255, -1)
                
    # Izračun % pokritosti na podlagi pikslov filtrirane maske in celotnega obrezanega lističa
    skupno_pikslov_listica = slika_izrez.shape[0] * slika_izrez.shape[1]
    piksli_kapljic = cv2.countNonZero(maska_filtrirana)
    
    pokritost_odstotek = (piksli_kapljic / skupno_pikslov_listica) * 100 if skupno_pikslov_listica > 0 else 0
    
    return "USPEH", slika_izrez, maska_filtrirana, stevilo_odtisov, pokritost_odstotek
