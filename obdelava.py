import cv2
import numpy as np

def analiziraj_sliko(slika_rgb):
    # --- 1. ISKANJE KONTURE ---
    hsv = cv2.cvtColor(slika_rgb, cv2.COLOR_RGB2HSV)
    # Razpon, ki ujame tudi svetlejše/temnejše dele rumenega lističa
    spodnja_rumena = np.array([10, 50, 50])
    zgornja_rumena = np.array([50, 255, 255])
    maska_rumene = cv2.inRange(hsv, spodnja_rumena, zgornja_rumena)
    
    # Zapolnitev lukenj (kjer je veliko kapljic) in čiščenje smeti
    maska_rumene = cv2.morphologyEx(maska_rumene, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    maska_rumene = cv2.morphologyEx(maska_rumene, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    
    konture, _ = cv2.findContours(maska_rumene, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not konture:
        return "NAPAKA", "Napaka: Na sliki ni rumenega lističa.", None, None, None
        
    najvecja_kontura = max(konture, key=cv2.contourArea)
    
    # --- 2. NATANČEN IZREZ BREZ DEFORMACIJ ---
    x, y, w, h = cv2.boundingRect(najvecja_kontura)
    slika_izrez = slika_rgb[y:y+h, x:x+w].copy()
    
    # Ustvarimo masko za točno (neravno) obliko lističa
    maska_izrez = np.zeros((h, w), dtype=np.uint8)
    zamaknjena_kontura = najvecja_kontura - [x, y]
    cv2.drawContours(maska_izrez, [zamaknjena_kontura], -1, 255, -1)
    
    # Ozadje zunaj lističa pobarvamo na čisto belo
    slika_izrez[maska_izrez == 0] = (255, 255, 255)
    
    # Obračanje v ležeč format za preglednost v aplikaciji
    if h > w:
        slika_izrez = cv2.rotate(slika_izrez, cv2.ROTATE_90_CLOCKWISE)
        maska_izrez = cv2.rotate(maska_izrez, cv2.ROTATE_90_CLOCKWISE)
        h, w = slika_izrez.shape[:2]
        
    # --- 3. PRAVILNO SKALIRANJE NA 76 mm ---
    # S pomočjo minAreaRect dobimo dejansko dolžino papirja (ne glede na nagib na sliki)
    rect = cv2.minAreaRect(najvecja_kontura)
    dejanska_dolzina_px = max(rect[1][0], rect[1][1])
    
    pike_na_mm = 60.0
    ciljna_dolzina_px = 76.0 * pike_na_mm
    
    # Izračunamo skupni faktor povečave in skaliramo sliko (ohranimo vsa naravna razmerja!)
