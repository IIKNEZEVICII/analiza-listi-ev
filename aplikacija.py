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
    obseg = cv2.arcLength(najvecja_kontura, True)
    
    oglisca = None
    for faktor in np.arange(0.01, 0.1, 0.005):
        temp_oglisca = cv2.approxPolyDP(najvecja_kontura, faktor * obseg, True)
        if len(temp_oglisca) == 4:
            oglisca = temp_oglisca
            break
            
    if oglisca is None or len(oglisca) != 4:
        return "NAPAKA", "Napaka: Prepoznava robov lističa ni uspela. Poskusite z drugo sliko.", None, None, None
        
    # Ravnanje lističa
    tocke = oglisca.reshape(4, 2)
    urejene_tocke = np.zeros((4, 2), dtype="float32")
    vsota = tocke.sum(axis=1)
    urejene_tocke[0] = tocke[np.argmin(vsota)]
    urejene_tocke[2] = tocke[np.argmax(vsota)]
    razlika = np.diff(tocke, axis=1)
    urejene_tocke[1] = tocke[np.argmin(razlika)]
    urejene_tocke[3] = tocke[np.argmax(razlika)]
    
    sirina_zgoraj = np.linalg.norm(urejene_tocke[0] - urejene_tocke[1])
    visina_levo = np.linalg.norm(urejene_tocke[0] - urejene_tocke[3])
    
    if visina_levo > sirina_zgoraj:
        sirina_mm, visina_mm = 26.0, 76.0
    else:
        sirina_mm, visina_mm = 76.0, 26.0
        
    pike_na_mm_cilj = 60.0
    sirina_nova = int(sirina_mm * pike_na_mm_cilj)
    visina_nova = int(visina_mm * pike_na_mm_cilj)
    
    ciljne_tocke = np.float32([
        [0, 0],
        [sirina_nova - 1, 0],
        [sirina_nova - 1, visina_nova - 1],
        [0, visina_nova - 1]
    ])
    
    matrika = cv2.getPerspectiveTransform(urejene_tocke, ciljne_tocke)
    slika_izrez = cv2.warpPerspective(slika_rgb, matrika, (sirina_nova, visina_nova))
    slika_izrez = slika_izrez[3:-3, 3:-3] # Odstranimo robove zaradi artefaktov transformacije
    
    # --- NOVO: VEDNO LEŽEČA POSTAVITEV ---
    # Če je bil listič prepoznan kot pokončen (višina > širina), 
    # ga zavrtimo za 90 stopinj v smeri urinega kazalca, da bo vedno ležeč.
    if visina_levo > sirina_zgoraj:
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
