import cv2
import numpy as np
import random

def analiziraj_sliko(slika_rgb):
    """Funkcija za obdelavo WSP lističa in izrez 3 naključnih 1x1cm kvadratkov z odpravljenim šumom in merilom"""
    
    # --- KORAK 1: PREPROST IN NARAVEN IZREZ ---
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
    
    # Ohranimo listič vodoravno (širina mora biti večja od višine)
    if h > w:
        slika_izrez = cv2.rotate(slika_izrez, cv2.ROTATE_90_CLOCKWISE)
        maska_izrez = cv2.rotate(maska_izrez, cv2.ROTATE_90_CLOCKWISE)
        temp = w
        w = h
        h = temp
        
    # --- POPRAVEK SKALIRANJA (Listič je dolg 76 mm po ŠIRINI) ---
    pike_na_mm_cilj = 60.0
    sirina_mm = 76.0
    ciljna_sirina_px = int(sirina_mm * pike_na_mm_cilj)
    
    # Pravilno izračunamo višino glede na prilagojeno širino
    razmerje = ciljna_sirina_px / w
    ciljna_visina_px = int(h * razmerje)
    
    slika_skalirana = cv2.resize(slika_izrez, (ciljna_sirina_px, ciljna_visina_px), interpolation=cv2.INTER_CUBIC)
    maska_skalirana = cv2.resize(maska_izrez, (ciljna_sirina_px, ciljna_visina_px), interpolation=cv2.INTER_NEAREST)
    
    um_na_piko = 1000 / pike_na_mm_cilj
    um2_na_piko = um_na_piko ** 2
    
    # --- KORAK 2: DETEKCIJA KAPLJIC Z ODPRAVLJANJEM ŠUMA ---
    # Uporaba mediane zameglitve močno zmanjša zaznavanje teksture papirja
    slika_blur = cv2.medianBlur(slika_skalirana, 5)
    
    siva = cv2.cvtColor(slika_blur, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    siva_clahe = clahe.apply(siva)
    
    # Prilagojen adaptivni prag (manj občutljiv na drobne sence)
    maska_kontrasta = cv2.adaptiveThreshold(
        siva_clahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 21, 6
    )
    
    maska_temnih = cv2.bitwise_and(maska_kontrasta, cv2.bitwise_not(cv2.inRange(siva_clahe, 185, 255)))
    
    hsv_izrez = cv2.cvtColor(slika_blur, cv2.COLOR_RGB2HSV)
    spodnja_modra = np.array([90, 40, 40])
    zgornja_modra = np.array([170, 255, 255])
    maska_hsv = cv2.inRange(hsv_izrez, spodnja_modra, zgornja_modra)
    
    maska_kapljic = cv2.bitwise_or(maska_temnih, maska_hsv)
    maska_kapljic = cv2.bitwise_and(maska_kapljic, maska_skalirana)
    
    # --- KORAK 3: ČIŠČENJE BREZ ZDRUŽEVANJA ---
    jedro_odpiranje = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    maska_cista = cv2.morphologyEx(maska_kapljic, cv2.MORPH_OPEN, jedro_odpiranje)
    
    # Inteligentno polnjenje lukenj (RETR_CCOMP) namesto Convex Hull, da ohranimo kapljice ločene
    konture_maske, hierarhija = cv2.findContours(maska_cista, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    maska_zapolnjena = maska_cista.copy()
    
    if hierarhija is not None:
        for i in range(len(konture_maske)):
            # Če ima kontura starša (kar pomeni, da je luknja znotraj kapljice), jo zapolnimo z belo
            if hierarhija[0][i][3] != -1:
                cv2.drawContours(maska_zapolnjena, [konture_maske[i]], -1, 255, -1)
                
    koncne_konture, _ = cv2.findContours(maska_zapolnjena, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    maska_filtrirana = np.zeros_like(maska_zapolnjena)
    
    # Filtracija > 50 µm
    for kontura in koncne_konture:
        povrsina_px = cv2.contourArea(kontura)
        povrsina_um2 = povrsina_px * um2_na_piko
        if povrsina_um2 > 0:
            premer_um = 2 * np.sqrt(povrsina_um2 / np.pi)
            if premer_um >= 50:
                cv2.drawContours(maska_filtrirana, [kontura], -1, 255, -1)
                
    slika_prikaz = cv2.bitwise_and(slika_skalirana, slika_skalirana, mask=maska_skalirana)
    
    # --- KORAK 4: PAMETEN IZREZ TREH 1cm x 1cm KVADRATKOV ---
    velikost_kvadrata_px = int(10 * pike_na_mm_cilj) # Natančno 600 px za 1 cm
    
    if ciljna_sirina_px < velikost_kvadrata_px or ciljna_visina_px < velikost_kvadrata_px:
        return "NAPAKA", "Listič je premajhen za izrez 1x1 cm kvadrata.", None
        
    rezultati_kvadratkov = []
    slika_z_oznakami = slika_prikaz.copy()
    
    poskusi = 0
    najdeni = 0
    
    while najdeni < 3 and poskusi < 100:
        x_rand = random.randint(0, ciljna_sirina_px - velikost_kvadrata_px)
        y_rand = random.randint(0, ciljna_visina_px - velikost_kvadrata_px)
        
        test_maska_papirja = maska_skalirana[y_rand:y_rand+velikost_kvadrata_px, x_rand:x_rand+velikost_kvadrata_px]
        povrsina_papirja_v_kvadratku = cv2.countNonZero(test_maska_papirja)
        
        if povrsina_papirja_v_kvadratku > (velikost_kvadrata_px * velikost_kvadrata_px * 0.9):
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
        return "NAPAKA", "Na lističu ni bilo mogoče najti treh celih 1x1 cm območij.", None
        
    return "USPEH", slika_z_oznakami, rezultati_kvadratkov
