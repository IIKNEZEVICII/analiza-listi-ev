import cv2
import numpy as np

def analiziraj_sliko(slika_rgb):
    """Funkcija za obdelavo in analizo vodoobčutljivega lističa"""
    
    # Pretvorba v HSV (Hue, Saturation, Value) prostor
    hsv = cv2.cvtColor(slika_rgb, cv2.COLOR_RGB2HSV)
    maska_rumene = cv2.inRange(hsv, np.array([10, 50, 50]), np.array([50, 255, 255]))
    
    # Zapolnimo luknje v maski lističa
    maska_rumene = cv2.morphologyEx(maska_rumene, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)))
    
    konture, _ = cv2.findContours(maska_rumene, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not konture:
        return "NAPAKA", "Napaka: Na sliki ni rumenega lističa.", None, None, None
        
    najvecja = max(konture, key=cv2.contourArea)
    
    # --- KORAK 1: FIZIČNA ROTACIJA SLIKE (Brez kakršnihkoli preslikav ali deformacij) ---
    rect = cv2.minAreaRect(najvecja)
    box = cv2.boxPoints(rect)
    
    # Izračunamo točen kot daljše stranice lističa
    d1 = np.linalg.norm(box[0] - box[1])
    d2 = np.linalg.norm(box[1] - box[2])
    
    if d1 > d2:
        dy = box[1][1] - box[0][1]
        dx = box[1][0] - box[0][0]
    else:
        dy = box[2][1] - box[1][1]
        dx = box[2][0] - box[1][0]
        
    kot_rotacije = np.degrees(np.arctan2(dy, dx))
    (cx, cy) = rect[0]
    
    # Zavrtimo celotno originalno sliko okoli njenega centra, da se listič zravna v vodoraven položaj
    h_orig, w_orig = slika_rgb.shape[:2]
    matrika_rotacije = cv2.getRotationMatrix2D((cx, cy), kot_rotacije, 1.0)
    
    slika_zravnana = cv2.warpAffine(slika_rgb, matrika_rotacije, (w_orig, h_orig), flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255))
    
    # Narišemo masko točne oblike lističa in jo prav tako zavrtimo
    maska_celotna = np.zeros((h_orig, w_orig), dtype=np.uint8)
    cv2.drawContours(maska_celotna, [najvecja], -1, 255, -1)
    maska_zravnana = cv2.warpAffine(maska_celotna, matrika_rotacije, (w_orig, h_orig), flags=cv2.INTER_NEAREST)
    
    # --- KORAK 2: ŠKARJAST IZREZ (Bounding Box) ---
    # Ko je slika zravnana, vzamemo samo naravni obdajajoči pravokotnik
    x, y, w, h = cv2.boundingRect(maska_zravnana)
    
    slika_izrez = slika_zravnana[y:y+h, x:x+w].copy()
    maska_izrez = maska_zravnana[y:y+h, x:x+w].copy()
    
    # Vse, kar na sliki ni listič (ozadje izven konture), pobarvamo belo, da ne moti iskanja kapljic
    slika_izrez[maska_izrez == 0] = (255, 255, 255)
    
    # --- KORAK 3: SKALIRANJE NA 76 mm ---
    # Zaradi rotacije zgoraj je listič sedaj vedno ležeč (širina w je daljša stranica, torej 76 mm)
    pike_na_mm_cilj = 60.0
    ciljna_sirina = int(76.0 * pike_na_mm_cilj)
    razmerje = ciljna_sirina / w
    ciljna_visina = int(h * razmerje)
    
    slika_koncna = cv2.resize(slika_izrez, (ciljna_sirina, ciljna_visina), interpolation=cv2.INTER_CUBIC)
    maska_koncna = cv2.resize(maska_izrez, (ciljna_sirina, ciljna_visina), interpolation=cv2.INTER_NEAREST)
    
    um_na_piko = 1000 /
