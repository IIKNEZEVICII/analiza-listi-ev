import cv2
import numpy as np

def analiziraj_sliko(slika_rgb):
    # 1. Iskanje lističa
    hsv = cv2.cvtColor(slika_rgb, cv2.COLOR_RGB2HSV)
    maska = cv2.inRange(hsv, np.array([10, 50, 50]), np.array([50, 255, 255]))
    
    konture, _ = cv2.findContours(maska, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not konture: 
        return "NAPAKA", "Ni lističa.", None, None, None
    
    najvecja = max(konture, key=cv2.contourArea)
    
    # 2. NAJBOLJ PREPROST IZREZ (Brez deformacij in rotacij)
    x, y, w, h = cv2.boundingRect(najvecja)
    
    # Izrežemo sliko natančno po tem okvirju
    izrez = slika_rgb[y:y+h, x:x+w].copy()
    
    # Če je listič pokončen, ga obrnemo, da je širina daljša stranica
    if h > w:
        izrez = cv2.rotate(izrez, cv2.ROTATE_90_CLOCKWISE)
        h, w = izrez.shape[:2]
        
    # 3. Skaliranje (Gledamo samo višino 76 mm)
    ciljna_sirina = int(76.0 * 60.0) # 4560 pikslov pri 60 dpi
    razmerje = ciljna_sirina / w
    ciljna_visina = int(h * razmerje)
    
    izrez_skaliran = cv2.resize(izrez, (ciljna_sirina, ciljna_visina), interpolation=cv2.INTER_CUBIC)
    
    # 4. Detekcija kapljic
    siva = cv2.cvtColor(izrez_skaliran, cv2.COLOR_RGB2GRAY)
    maska_kontrasta = cv2.adaptiveThreshold(siva, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 7)
    
    maska_kapljic = cv2.bitwise_and(maska_kontrasta, cv2.bitwise_not(cv2.inRange(siva, 175, 255)))
    
    koncne_konture, _ = cv2.findContours(maska_kapljic, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    maska_filtrirana = np.zeros_like(maska_kapljic)
    
    st_odtisov = 0
    um2_na_piko = (1000 / 60.0) ** 2
    
    for k in koncne_konture:
        povrsina = cv2.contourArea(k) * um2_na_piko
        if povrsina > 0:
            premer = 2 * np.sqrt(povrsina / np.pi)
            if premer >= 50:
                st_odtisov += 1
                cv2.drawContours(maska_filtrirana, [k], -1, 255, -1)
                
    piksli_kapljic = cv2.countNonZero(maska_filtrirana)
    piksli_vsi = izrez_skaliran.shape[0] * izrez_skaliran.shape[1]
    pokritost = (piksli_kapljic / piksli_vsi) * 100
    
    return "USPEH", izrez_skaliran, maska_filtrirana, st_odtisov, pokritost
