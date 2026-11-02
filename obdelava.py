import cv2
import numpy as np
import random

def uredi_tocke(tocke):
    """Razvrsti 4 točke v vrstni red: zgoraj-levo, zgoraj-desno, spodaj-desno, spodaj-levo"""
    tocke = tocke.reshape((4, 2))
    nove_tocke = np.zeros((4, 2), dtype=np.float32)
    
    vsota = tocke.sum(axis=1)
    nove_tocke[0] = tocke[np.argmin(vsota)] # Zgoraj levo
    nove_tocke[2] = tocke[np.argmax(vsota)] # Spodaj desno
    
    razlika = np.diff(tocke, axis=1)
    nove_tocke[1] = tocke[np.argmin(razlika)] # Zgoraj desno
    nove_tocke[3] = tocke[np.argmax(razlika)] # Spodaj levo
    
    return nove_tocke

def analiziraj_sliko(slika_rgb):
    """Vrhunska analiza: perspektivno ravnanje, globalna analiza in >50µm cenzura."""
    
    # --- KORAK 1: DETEKCIJA LISTIČA ---
    hsv_orig = cv2.cvtColor(slika_rgb, cv2.COLOR_RGB2HSV)
    spodnja_rumena = np.array([15, 80, 80])
    zgornja_rumena = np.array([45, 255, 255])
    maska_rumene = cv2.inRange(hsv_orig, spodnja_rumena, zgornja_rumena)
    
    jedro = np.ones((5, 5), np.uint8)
    maska_rumene = cv2.morphologyEx(maska_rumene, cv2.MORPH_OPEN, jedro)
    maska_rumene = cv2.morphologyEx(maska_rumene, cv2.MORPH_CLOSE, jedro)
    
    konture, _ = cv2.findContours(maska_rumene, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not konture:
        return "NAPAKA", "Napaka: Na sliki ni rumenega lističa.", None, None
        
    najvecja_kontura = max(konture, key=cv2.contourArea)
    maska_papirja_celotna = np.zeros(slika_rgb.shape[:2], dtype=np.uint8)
    cv2.drawContours(maska_papirja_celotna, [najvecja_kontura], -1, 255, -1)
    
    # --- KORAK 2: MATEMATIČNO RAVNANJE (Perspektivna preslikava) ---
    # Poiščemo minimalni pravokotnik (tudi če je pod kotom)
    pravokotnik = cv2.minAreaRect(najvecja_kontura)
    tocke_pravokotnika = cv2.boxPoints(pravokotnik)
    tocke_pravokotnika = np.int32(tocke_pravokotnika)
    
    urejene_tocke = uredi_tocke(tocke_pravokotnika)
    (tl, tr, br, bl) = urejene_tocke
    
    # Izračun natančne širine in višine nagnjenega papirja
    sirinaA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    sirinaB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    sirina_px = max(int(sirinaA), int(sirinaB))
    
    visinaA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    visinaB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    visina_px = max(int(visinaA), int(visinaB))
    
    ciljne_tocke = np.array([
        [0, 0],
        [sirina_px - 1, 0],
        [sirina_px - 1, visina_px - 1],
        [0, visina_px - 1]
    ], dtype="float32")
    
    # Transformacija same slike in maske, da sta popolnoma ravni
    matrika_preslikave = cv2.getPerspectiveTransform(urejene_tocke, ciljne_tocke)
    slika_zravnana = cv2.warpPerspective(slika_rgb, matrika_preslikave, (sirina_px, visina_px))
    maska_zravnana = cv2.warpPerspective(maska_papirja_celotna, matrika_preslikave, (sirina_px, visina_px))
    
    # Zagotovimo vodoravno orientacijo
    if visina_px > sirina_px:
        slika_zravnana = cv2.rotate(slika_zravnana, cv2.ROTATE_90_CLOCKWISE)
        maska_zravnana = cv2.rotate(maska_zravnana, cv2.ROTATE_90_CLOCKWISE)
        temp = sirina_px
        sirina_px = visina_px
        visina_px = temp
        
    pike_na_mm_cilj = 60.0
    daljsa_stranica_mm = 76.0
    ciljna_sirina_px = int(daljsa_stranica_mm * pike_na_mm_cilj)
    razmerje = ciljna_sirina_px / sirina_px
    ciljna_visina_px = int(visina_px * razmerje)
    
    slika_skalirana = cv2.resize(slika_zravnana, (ciljna_sirina_px, ciljna_visina_px), interpolation=cv2.INTER_CUBIC)
    maska_skalirana = cv2.resize(maska_zravnana, (ciljna_sirina_px, ciljna_visina_px), interpolation=cv2.INTER_NEAREST)
    _, maska_skalirana = cv2.threshold(maska_skalirana, 127, 255, cv2.THRESH_BINARY)
    
    um_na_piko = 1000 / pike_na_mm_cilj
    um2_na_piko = um_na_piko ** 2
    
    # --- KORAK 3: DETEKCIJA ---
    siva = cv2.cvtColor(slika_skalirana, cv2.COLOR_RGB2GRAY)
    siva_zglad = cv2.medianBlur(siva, 3) 
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    siva_clahe = clahe.apply(siva_zglad)
    
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
    
    # Čiščenje
    jedro_zapiranje = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    maska_sklenjena = cv2.morphologyEx(maska_kapljic, cv2.MORPH_CLOSE, jedro_zapiranje)
    
    konture_maske, hierarhija = cv2.findContours(maska_sklenjena, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    maska_zapolnjena = maska_sklenjena.copy()
    if hierarhija is not None:
        for i in range(len(konture_maske)):
            if hierarhija[0][i][3] != -1:
                cv2.drawContours(maska_zapolnjena, [konture_maske[i]], -1, 255, -1)
                
    jedro_odpiranje = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    maska_cista = cv2.morphologyEx(maska_zapolnjena, cv2.MORPH_OPEN, jedro_odpiranje)
    
    # --- KORAK 4: GLOBALNA ANALIZA IN FILTRACIJA (>50µm) ---
    koncne_konture, _ = cv2.findContours(maska_cista, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    maska_filtrirana = np.zeros_like(maska_cista)
    
    st_kapljic_global = 0
    for kontura in koncne_konture:
        povrsina_px = cv2.contourArea(kontura)
        povrsina_um2 = povrsina_px * um2_na_piko
        if povrsina_um2 > 0:
            premer_um = 2 * np.sqrt(povrsina_um2 / np.pi)
            
            # STROGO: Kapljice pod 50 µm sploh ne narišemo in ne preštejemo
            if premer_um >= 50:
                cv2.drawContours(maska_filtrirana, [kontura], -1, 255, -1)
                st_kapljic_global += 1
                
    piksli_kapljic_global = cv2.countNonZero(maska_filtrirana)
    povrsina_papirja_global = cv2.countNonZero(maska_skalirana)
    pokritost_global = (piksli_kapljic_global / povrsina_papirja_global) * 100 if povrsina_papirja_global > 0 else 0
    
    globalni_rezultati = {
        "st_kapljic": st_kapljic_global,
        "pokritost": pokritost_global
    }
                
    slika_prikaz = cv2.bitwise_and(slika_skalirana, slika_skalirana, mask=maska_skalirana)
    
    # --- KORAK 5: IZREZ IN OŠTEVILČENJE TREH 1x1 cm KVADRATKOV ---
    velikost_kvadrata_px = int(10 * pike_na_mm_cilj) 
    rob_px = int(2.0 * pike_na_mm_cilj) 
    
    if ciljna_sirina_px < velikost_kvadrata_px + 2*rob_px or ciljna_visina_px < velikost_kvadrata_px + 2*rob_px:
        rob_px = 0
        if ciljna_sirina_px < velikost_kvadrata_px or ciljna_visina_px < velikost_kvadrata_px:
            return "NAPAKA", "Listič je premajhen za izrez 1x1 cm kvadrata.", None, None
        
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
            
            cv2.rectangle(slika_z_oznakami, (x_rand, y_rand), (x_rand+velikost_kvadrata_px, y_rand+velikost_kvadrata_px), (255, 0, 0), 15)
            
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
        return "NAPAKA", "Zaradi prekrivanja ni bilo mogoče najti 3 območij.", None, None
        
    return "USPEH", slika_z_oznakami, rezultati_kvadratkov, globalni_rezultati
