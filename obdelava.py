import cv2
import numpy as np
import random

def analiziraj_sliko(slika_rgb):
    """Izboljšana funkcija: konveksna ovojnica za razdrobljen listič, točno merilo in odprava šuma."""
    
    # --- KORAK 1: NARAVEN IZREZ IN SKALIRANJE ---
    hsv_orig = cv2.cvtColor(slika_rgb, cv2.COLOR_RGB2HSV)
    spodnja_rumena = np.array([15, 80, 80])
    zgornja_rumena = np.array([45, 255, 255])
    maska_rumene = cv2.inRange(hsv_orig, spodnja_rumena, zgornja_rumena)
    
    jedro = np.ones((5, 5), np.uint8)
    maska_rumene = cv2.morphologyEx(maska_rumene, cv2.MORPH_OPEN, jedro)
    maska_rumene = cv2.morphologyEx(maska_rumene, cv2.MORPH_CLOSE, jedro)
    
    konture, _ = cv2.findContours(maska_rumene, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not konture:
        return "NAPAKA", "Napaka: Na sliki ni rumenega lističa.", None
        
    # POPRAVEK: Združevanje razdrobljenih rumenih območij zaradi gostih kapljic
    # 1. Izločimo drobne smeti (šum) in ohranimo le pomembnejše dele lističa (površina > 500)
    velike_konture = [k for k in konture if cv2.contourArea(k) > 500]
    
    # Če slučajno noben del ni večji od 500, vzamemo pač tistega največjega
    if not velike_konture:
        velike_konture = [max(konture, key=cv2.contourArea)]
        
    # 2. Vse točke zložimo v en niz
    vse_tocke = np.vstack(velike_konture)
    
    # 3. Narišemo konveksno ovojnico (elastiko okrog vseh točk), ki zapolni notranje luknje in prekinitve
    ovojnica = cv2.convexHull(vse_tocke)
    
    maska_papirja_celotna = np.zeros(slika_rgb.shape[:2], dtype=np.uint8)
    cv2.drawContours(maska_papirja_celotna, [ovojnica], -1, 255, -1)
    
    x, y, w, h = cv2.boundingRect(ovojnica)
    slika_izrez = slika_rgb[y:y+h, x:x+w].copy()
    maska_izrez = maska_papirja_celotna[y:y+h, x:x+w].copy()
    
    # GARANCIJA MERILA: Daljša stranica je na standardnem WSP lističu vedno 76 mm.
    pike_na_mm_cilj = 60.0
    daljsa_stranica_mm = 76.0
    
    if h > w:
        ciljna_visina_px = int(daljsa_stranica_mm * pike_na_mm_cilj)
        razmerje = ciljna_visina_px / h
        ciljna_sirina_px = int(w * razmerje)
    else:
        ciljna_sirina_px = int(daljsa_stranica_mm * pike_na_mm_cilj)
        razmerje = ciljna_sirina_px / w
        ciljna_visina_px = int(h * razmerje)
        
    slika_skalirana = cv2.resize(slika_izrez, (ciljna_sirina_px, ciljna_visina_px), interpolation=cv2.INTER_CUBIC)
    maska_skalirana = cv2.resize(maska_izrez, (ciljna_sirina_px, ciljna_visina_px), interpolation=cv2.INTER_NEAREST)
    
    um_na_piko = 1000 / pike_na_mm_cilj
    um2_na_piko = um_na_piko ** 2
    
    # --- KORAK 2: ČISTA DETEKCIJA KAPLJIC ---
    slika_blur = cv2.GaussianBlur(slika_skalirana, (3, 3), 0)
    
    hsv = cv2.cvtColor(slika_blur, cv2.COLOR_RGB2HSV)
    spodnja_modra = np.array([85, 40, 40])
    zgornja_modra = np.array([170, 255, 255])
    maska_modre = cv2.inRange(hsv, spodnja_modra, zgornja_modra)
    
    siva = cv2.cvtColor(slika_blur, cv2.COLOR_RGB2GRAY)
    _, maska_crne = cv2.threshold(siva, 100, 255, cv2.THRESH_BINARY_INV)
    
    maska_kapljic = cv2.bitwise_or(maska_modre, maska_crne)
    maska_kapljic = cv2.bitwise_and(maska_kapljic, maska_skalirana)
    
    # --- KORAK 3: ODSTRANJEVANJE MANJŠIH OD 50 µm ---
    jedro_zapiranje = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    maska_cista = cv2.morphologyEx(maska_kapljic, cv2.MORPH_CLOSE, jedro_zapiranje)
    
    koncne_konture, _ = cv2.findContours(maska_cista, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    maska_filtrirana = np.zeros_like(maska_cista)
    
    min_povrsina_um2 = np.pi * (25 ** 2) 
    
    for kontura in koncne_konture:
        povrsina_px = cv2.contourArea(kontura)
        povrsina_um2 = povrsina_px * um2_na_piko
        
        if povrsina_um2 >= min_povrsina_um2:
            cv2.drawContours(maska_filtrirana, [kontura], -1, 255, -1)
                
    slika_prikaz = cv2.bitwise_and(slika_skalirana, slika_skalirana, mask=maska_skalirana)
    
    # --- KORAK 4: IZREZ TREH 1cm x 1cm KVADRATKOV ---
    velikost_kvadrata_px = int(10 * pike_na_mm_cilj) 
    rob_px = int(1.5 * pike_na_mm_cilj) 
    
    if ciljna_sirina_px < velikost_kvadrata_px + 2*rob_px or ciljna_visina_px < velikost_kvadrata_px + 2*rob_px:
        rob_px = 0
        if ciljna_sirina_px < velikost_kvadrata_px or ciljna_visina_px < velikost_kvadrata_px:
            return "NAPAKA", "Listič je premajhen za izrez 1x1 cm kvadrata.", None
        
    rezultati_kvadratkov = []
    shranjene_koordinate = [] 
    slika_z_oznakami = slika_prikaz.copy()
    
    poskusi = 0
    najdeni = 0
    
    while najdeni < 3 and poskusi < 500: 
        x_rand = random.randint(rob_px, ciljna_sirina_px - velikost_kvadrata_px - rob_px)
        y_rand = random.randint(rob_px, ciljna_visina_px - velikost_kvadrata_px - rob_px)
        
        prekrivanje = False
        for (sx, sy) in shranjene_koordinate:
            if abs(x_rand - sx) < velikost_kvadrata_px and abs(y_rand - sy) < velikost_kvadrata_px:
                prekrivanje = True
                break
                
        if prekrivanje:
            poskusi += 1
            continue
            
        test_maska_papirja = maska_skalirana[y_rand:y_rand+velikost_kvadrata_px, x_rand:x_rand+velikost_kvadrata_px]
        povrsina_papirja_v_kvadratku = cv2.countNonZero(test_maska_papirja)
        
        # Test povrsine se preverja po narisanem Convex Hull, tako da luknje zaradi kapljic niso več ovira
        if povrsina_papirja_v_kvadratku > (velikost_kvadrata_px * velikost_kvadrata_px * 0.95):
            shranjene_koordinate.append((x_rand, y_rand))
            
            cv2.rectangle(slika_z_oznakami, (x_rand, y_rand), (x_rand+velikost_kvadrata_px, y_rand+velikost_kvadrata_px), (0, 255, 0), 6)
            
            izrez_rgb = slika_prikaz[y_rand:y_rand+velikost_kvadrata_px, x_rand:x_rand+velikost_kvadrata_px]
            izrez_maska = maska_filtrirana[y_rand:y_rand+velikost_kvadrata_px, x_rand:x_rand+velikost_kvadrata_px]
            
            konture_izrez, _ = cv2.findContours(izrez_maska, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            st_kapljic = len(konture_izrez)
            
            piksli_kapljic = cv2.countNonZero(izrez_maska)
            pokritost_odstotek = (piksli_kapljic / povrsina_papirja_v_kvadratku) * 100
            
            rezultati_kvadratkov.append({
                "rgb": izrez_rgb,
                "maska": izrez_maska,
                "st_kapljic": st_kapljic,
                "pokritost": pokritost_odstotek
            })
            najdeni += 1
            
        poskusi += 1
        
    if najdeni < 3:
        return "NAPAKA", "Na lističu zaradi robov in prekrivanja ni bilo mogoče najti 3 ločenih 1x1 cm območij.", None
        
    return "USPEH", slika_z_oznakami, rezultati_kvadratkov
