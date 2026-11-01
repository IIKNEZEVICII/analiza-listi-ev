import cv2
import numpy as np

def uredi_tacke(pts):
    """Pravilno sortiranje 4 tačke kako se slika ne bi ukrstila ili prevrnula"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)] # Gore-levo
    rect[2] = pts[np.argmax(s)] # Dole-desno
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] # Gore-desno
    rect[3] = pts[np.argmax(diff)] # Dole-levo
    return rect

def analiziraj_sliko(slika_rgb):
    # --- KORAK 1: PREPOZNAVANJE OBLIKA I IZREZ BEZ DISTORZIJE ---
    hsv_orig = cv2.cvtColor(slika_rgb, cv2.COLOR_RGB2HSV)
    spodnja_rumena = np.array([15, 80, 80])
    zgornja_rumena = np.array([45, 255, 255])
    maska_rumene = cv2.inRange(hsv_orig, spodnja_rumena, zgornja_rumena)
    
    # Blago čišćenje maske za stabilniju spoljnu ivicu
    jedro = np.ones((5, 5), np.uint8)
    maska_rumene = cv2.morphologyEx(maska_rumene, cv2.MORPH_OPEN, jedro)
    
    konture, _ = cv2.findContours(maska_rumene, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not konture:
        return "NAPAKA", "Na slici nema žutog papira.", None, None, None
        
    najvecja_kontura = max(konture, key=cv2.contourArea)
    
    # Korišćenjem minAreaRect dobijamo savršen matematički pravougaonik 
    # Ovo apsolutno sprečava bilo kakvo rastezanje i deformaciju unutar slike!
    rect = cv2.minAreaRect(najvecja_kontura)
    box = cv2.boxPoints(rect)
    box = np.float32(box)
    
    urejene_tocke = uredi_tacke(box)
    
    # Računamo širinu i visinu isečka zadržavajući prirodne proporcije sa slike
    sirina_px = int(max(np.linalg.norm(urejene_tocke[0] - urejene_tocke[1]),
                        np.linalg.norm(urejene_tocke[2] - urejene_tocke[3])))
    visina_px = int(max(np.linalg.norm(urejene_tocke[0] - urejene_tocke[3]),
                        np.linalg.norm(urejene_tocke[1] - urejene_tocke[2])))
    
    ciljne_tocke = np.float32([
        [0, 0],
        [sirina_px - 1, 0],
        [sirina_px - 1, visina_px - 1],
        [0, visina_px - 1]
    ])
    
    matrika = cv2.getPerspectiveTransform(urejene_tocke, ciljne_tocke)
    slika_izrez = cv2.warpPerspective(slika_rgb, matrika, (sirina_px, visina_px))
    
    # Osiguravamo da je list uvek u uspravnom položaju radi lakšeg računanja
    if visina_px < sirina_px:
        slika_izrez = cv2.rotate(slika_izrez, cv2.ROTATE_90_CLOCKWISE)
        
    # Skraćujemo oko 2% sa svake strane da bismo uklonili oštećene/pocepane ivice i eventualnu pozadinu
    h, w = slika_izrez.shape[:2]
    rob_x, rob_y = int(w * 0.02), int(h * 0.02)
    slika_izrez = slika_izrez[rob_y:-rob_y, rob_x:-rob_x]
    
    # --- KORAK 2: PRIMENA PRAVILA ZA 76mm ---
    faktor_krcenja = 0.985
    referencna_visina_mm = 76.0 * faktor_krcenja
    
    # Pošto nismo deformisali sliku, sada širinu možemo izračunati strogo proporcionalno!
    h_nova, w_nova = slika_izrez.shape[:2]
    sirina_mm = referencna_visina_mm * (w_nova / h_nova)
    
    pike_na_mm_cilj = 60.0
    koncna_visina = int(referencna_visina_mm * pike_na_mm_cilj)
    koncna_sirina = int(sirina_mm * pike_na_mm_cilj)
    
    slika_izrez = cv2.resize(slika_izrez, (koncna_sirina, koncna_visina), interpolation=cv2.INTER_CUBIC)
    
    um_na_piko = 1000 / pike_na_mm_cilj
    um2_na_piko = um_na_piko ** 2
    
    # --- KORAK 3: DETEKCIJA KAPLJICA (Ostaje ista) ---
    siva = cv2.cvtColor(slika_izrez, cv2.COLOR_RGB2GRAY)
    maska_kontrasta = cv2.adaptiveThreshold(siva, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 7)
    maska_temnih = cv2.bitwise_and(maska_kontrasta, cv2.bitwise_not(cv2.inRange(siva, 175, 255)))
    
    hsv_izrez = cv2.cvtColor(slika_izrez, cv2.COLOR_RGB2HSV)
    maska_hsv = cv2.inRange(hsv_izrez, np.array([75, 12, 12]), np.array([165, 255, 255]))
    
    maska_kapljic = cv2.bitwise_or(maska_temnih, maska_hsv)
    
    jedro_zapiranje = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    maska_sklenjena = cv2.morphologyEx(maska_kapljic, cv2.MORPH_CLOSE, jedro_zapiranje)
    
    konture_maske, _ = cv2.findContours(maska_sklenjena, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    maska_zapolnjena = np.zeros_like(maska_sklenjena)
    for kontura in konture_maske:
        cv2.drawContours(maska_zapolnjena, [cv2.convexHull(kontura)], -1, 255, -1)
        
    maska_cista = cv2.morphologyEx(maska_zapolnjena, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    
    # --- KORAK 4: IZRAČUNAVANJE ---
    koncne_konture, _ = cv2.findContours(maska_cista, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    stevilo_odtisov = 0
    maska_filtrirana = np.zeros_like(maska_cista)
    
    for kontura in koncne_konture:
        povrsina_um2 = cv2.contourArea(kontura) * um2_na_piko
        if povrsina_um2 > 0:
            premer_um = 2 * np.sqrt(povrsina_um2 / np.pi)
            if premer_um >= 50:
                stevilo_odtisov += 1
                cv2.drawContours(maska_filtrirana, [kontura], -1, 255, -1)
                
    skupno_pikslov = slika_izrez.shape[0] * slika_izrez.shape[1]
    pokritost_odstotek = (cv2.countNonZero(maska_filtrirana) / skupno_pikslov) * 100 if skupno_pikslov > 0 else 0
    
    # Na kraju rotiramo rezultat za lepši vizuelni prikaz u tvojoj aplikaciji
    slika_izrez_prikaz = cv2.rotate(slika_izrez, cv2.ROTATE_90_COUNTERCLOCKWISE)
    maska_filtrirana_prikaz = cv2.rotate(maska_filtrirana, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    return "USPEH", slika_izrez_prikaz, maska_filtrirana_prikaz, stevilo_odtisov, pokritost_odstotek
