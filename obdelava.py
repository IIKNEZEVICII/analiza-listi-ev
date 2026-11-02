import cv2
import numpy as np
import random

def analiziraj_sliko(slika_rgb):
    """Vrnitev na originalno in natančno detekcijo, a brez združevanja kapljic + odebeljene vizualije."""
    
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
        
    najvecja_kontura = max(konture, key=cv2.contourArea)
    maska_papirja_celotna = np.zeros(slika_rgb.shape[:2], dtype=np.uint8)
    cv2.drawContours(maska_papirja_celotna, [najvecja_kontura], -1, 255, -1)
    
    x, y, w, h = cv2.boundingRect(najvecja_kontura)
    slika_izrez = slika_rgb[y:y+h, x:x+w].copy()
    maska_izrez = maska_papirja_celotna[y:y+h, x:x+w].copy()
    
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
    
    # --- KORAK 2: ORIGINALNA ZANESLJIVA DETEKCIJA KAPLJIC ---
    siva = cv2.cvtColor(slika_skalirana, cv2.COLOR_RGB2GRAY)
    
    # Zelo rahel filter (3x3), ki zgladi zrnato teksturo rumenega papirja, pravih kapljic pa ne poškoduje
    siva_zglad = cv2.medianBlur(siva, 3) 
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    siva_clahe = clahe.apply(siva_zglad)
    
    # Vrnjeno na tvoj natančnejši adaptivni prag iz originalne kode
    maska_kontrasta = cv2.adaptiveThreshold(
        siva_clahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 15, 4
    )
    
    maska_temnih = cv2.bitwise_and(maska_kontrasta, cv2.bitwise_not(cv2.inRange(siva_clahe, 185, 255)))
    
    hsv_izrez = cv2.cvtColor(slika_skalirana, cv2.COLOR_RGB2HSV)
    spodnja_modra = np.array([85, 30, 30])
    zgornja_modra = np.array([170, 255, 255])
    maska_hsv = cv2.inRange(hsv_izrez, spodnja_modra, zgornja_modra)
    
    maska_kapljic = cv2.bitwise_or(maska_temnih, maska_hsv)
    maska_kapljic = cv2.bitwise_and(maska_kapljic, maska_skalirana)
    
    # --- KORAK 3: ČIŠČENJE BREZ ZDRUŽEVANJA KAPLJIC ---
    jedro_zapiranje = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    maska_sklenjena = cv2.morphologyEx(maska_kapljic, cv2.MORPH_CLOSE, jedro_zapiranje)
    
    # Tukaj smo odstranili "Convex Hull". Uporabimo samo topološko zapolnjevanje (RETR_CCOMP), da ohranimo ločene kapljice.
    konture_maske, hierarhija = cv2.findContours(maska_sklenjena, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    maska_zapolnjena = maska_sklenjena.copy()
    if hierarhija is not None:
        for i in range(len(konture_maske)):
            if hierarhija[0][i][3] != -1:
                cv2.drawContours(maska_zapolnjena, [konture_maske[i]], -1, 255, -1)
                
    jedro_odpiranje = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    maska_cista = cv2.morphologyEx(maska_zapolnjena, cv2.MORPH_OPEN, jedro_odpiranje)
    
    koncne_konture, _ = cv2.findContours(maska_cista, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    maska_filtrirana = np.zeros_like(maska_cista)
    
    for kontura in koncne_konture:
        povrsina_px = cv2.contourArea(kontura)
        povrsina_um2 = povrsina_px * um2_na_piko
        if povrsina_um2 > 0:
            premer_um = 2 * np.sqrt(povrsina_um2 / np.pi)
            if premer_um >= 50:
                cv2.drawContours(maska_filtrirana, [kontura], -1, 255, -1)
                
    slika_prikaz = cv2.bitwise_and(slika_skalirana, slika_skalirana, mask=maska_skalirana)
    
    # --- KORAK 4: IZREZ TREH 1x1 cm KVADRATKOV ---
    velikost_kvadrata_px = int(10 * pike_na_mm_cilj) 
    rob_px = int(2.0 * pike_na_mm_cilj) 
    
    if ciljna_sirina_px < velikost_kvadrata_px + 2*rob_px or ciljna_visina_px < velikost_kvadrata_px + 2*rob_px:
        rob_px = 0
        if ciljna_sirina_px < velikost_kvadrata_px or ciljna_visina_px < velikost_kvadrata_px:
            return "NAPAKA", "Listič je premajhen za izrez 1x1 cm kvadrata.", None
        
    rezultati_kvadratkov = []
    shranjene_koordinate = [] 
    slika_z_oznakami = slika_prikaz.copy()
    
    poskusi = 0
    najdeni = 0
    
    while najdeni < 3 and poskusi < 1000:
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
        
        if povrsina_papirja_v_kvadratku >= (velikost_kvadrata_px * velikost_kvadrata_px * 0.99):
            shranjene_koordinate.append((x_rand, y_rand)) 
            
            # Debelejši okvir (debelina povečana na 15)
            cv2.rectangle(slika_z_oznakami, (x_rand, y_rand), (x_rand+velikost_kvadrata_px, y_rand+velikost_kvadrata_px), (255, 0, 0), 15)
            
            # Povečane številke (Scale povečan na 4, debelina teksta na 10, zamik prilagojen)
            oznaka = str(najdeni + 1)
            cv2.putText(slika_z_oznakami, oznaka, (x_rand + 30, y_rand + 110), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 0, 0), 10, cv2.LINE_AA)
            
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
        return "NAPAKA", "Na lističu zaradi preprečevanja robov in prekrivanja ni bilo mogoče najti 3 ločenih 1x1 cm območij.", None
        
    return "USPEH", slika_z_oznakami, rezultati_kvadratkov
