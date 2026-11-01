import cv2
import numpy as np

def analiziraj_sliko(slika_rgb):
    """Funkcija za obdelavo in analizo vodoobčutljivega lističa"""
    
    # =================================================================
    # KORAK 1: PREPROST IN NARAVEN IZREZ (IZ KODE OBDELAVA 1)
    # =================================================================
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
    
    # Ustvarimo masko točne oblike lističa po njegovi dejanski konturi
    maska_papirja_celotna = np.zeros(slika_rgb.shape[:2], dtype=np.uint8)
    cv2.drawContours(maska_papirja_celotna, [najvecja_kontura], -1, 255, -1)
    
    # RAVNI obdajajoči pravokotnik (Brez popačenj!)
    x, y, w, h = cv2.boundingRect(najvecja_kontura)
    
    # Izrežemo sliko in masko - tukaj namerno NE pobarvamo ozadja na belo!
    slika_izrez = slika_rgb[y:y+h, x:x+w].copy()
    maska_izrez = maska_papirja_celotna[y:y+h, x:x+w].copy()
    
    # Vedno ležeč položaj (daljša stran je širina)
    if h > w:
        slika_izrez = cv2.rotate(slika_izrez, cv2.ROTATE_90_CLOCKWISE)
        maska_izrez = cv2.rotate(maska_izrez, cv2.ROTATE_90_CLOCKWISE)
        h, w = slika_izrez.shape[:2]
        
    # --- SKALIRANJE (Fiksna dolžina 76mm na daljši stranici) ---
    pike_na_mm_cilj = 60.0
    ciljna_sirina_px = int(76.0 * pike_na_mm_cilj)
    
    # Višino izračunamo strogo proporcionalno
    razmerje = ciljna_sirina_px / w
    ciljna_visina_px = int(h * razmerje)
    
    slika_skalirana = cv2.resize(slika_izrez, (ciljna_sirina_px, ciljna_visina_px), interpolation=cv2.INTER_CUBIC)
    maska_skalirana = cv2.resize(maska_izrez, (ciljna_sirina_px, ciljna_visina_px), interpolation=cv2.INTER_NEAREST)
    
    um_na_piko = 1000 / pike_na_mm_cilj
    um2_na_piko = um_na_piko ** 2
    
    # =================================================================
    # KORAK 2: ORIGINALNA DETEKCIJA KAPLJIC IN ČIŠČENJE
    # =================================================================
    siva = cv2.cvtColor(slika_skalirana, cv2.COLOR_RGB2GRAY)
    
    # Tvoj originalni adaptivni prag
    maska_kontrasta = cv2.adaptiveThreshold(
        siva, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 21, 7
    )
    maska_temnih = cv2.bitwise_and(maska_kontrasta, cv2.bitwise_not(cv2.inRange(siva, 175, 255)))
    
    hsv_izrez = cv2.cvtColor(slika_skalirana, cv2.COLOR_RGB2HSV)
    spodnja_modra = np.array([75, 12, 12])
    zgornja_modra = np.array([165, 255, 255])
    maska_hsv = cv2.inRange(hsv_izrez, spodnja_modra, zgornja_modra)
    
    maska_kapljic = cv2.bitwise_or(maska_temnih, maska_hsv)
    
    # KLJUČNO ZA POPRAVEK: Masko lističa skrčimo za nekaj pikslov navznoter (cv2.erode), 
    # da se povsem izognemo temnemu in problematičnemu robu, nato pa odrežemo ozadje.
    rob_jedro = np.ones((7, 7), np.uint8)
    maska_notranjosti = cv2.erode(maska_skalirana, rob_jedro)
    maska_kapljic = cv2.bitwise_and(maska_kapljic, maska_notranjosti)
    
    # Tvoje originalno čiščenje
    jedro_zapiranje = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    maska_sklenjena = cv2.morphologyEx(maska_kapljic, cv2.MORPH_CLOSE, jedro_zapiranje)
    
    konture_maske, _ = cv2.findContours(maska_sklenjena, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    maska_zapolnjena = np.zeros_like(maska_sklenjena)
    for kontura in konture_maske:
        lupina = cv2.convexHull(kontura)
        cv2.drawContours(maska_zapolnjena, [lupina], -1, 255, -1)
        
    jedro_odpiranje = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    maska_cista = cv2.morphologyEx(maska_zapolnjena, cv2.MORPH_OPEN, jedro_odpiranje)
    
    # =================================================================
    # KORAK 3: IZRAČUNI (% pokritosti in št. odtisov)
    # =================================================================
    koncne_konture, _ = cv2.findContours(maska_cista, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    stevilo_odtisov = 0
    maska_filtrirana = np.zeros_like(maska_cista)
    
    for kontura in koncne_konture:
        povrsina_px = cv2.contourArea(kontura)
        povrsina_um2 = povrsina_px * um2_na_piko
        
        if povrsina_um2 > 0:
            premer_um = 2 * np.sqrt(povrsina_um2 / np.pi)
            
            # Shranimo samo nad 50 mikrometrov
            if premer_um >= 50:
                stevilo_odtisov += 1
                cv2.drawContours(maska_filtrirana, [kontura], -1, 255, -1)
                
    # Pokritost izračunana natančno na piksel lističa
    piksli_papirja = cv2.countNonZero(maska_skalirana)
    piksli_kapljic = cv2.countNonZero(maska_filtrirana)
    
    pokritost_odstotek = (piksli_kapljic / piksli_papirja) * 100 if piksli_papirja > 0 else 0
    
    # Sedaj, šele na koncu, pobarvamo mizo v kotih izreza na belo samo za lep prikaz v aplikaciji
    slika_prikaz = slika_skalirana.copy()
    slika_prikaz[maska_skalirana == 0] = (255, 255, 255)
    
    return "USPEH", slika_prikaz, maska_filtrirana, stevilo_odtisov, pokritost_odstotek
