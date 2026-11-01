import cv2
import numpy as np

def izrezi_po_konturi(slika_rgb):
    """
    Funkcija, ki poišče rumen listič in ga izreže točno po njegovi 
    dejanski (neravni) konturi, brez kakršnihkoli popačenj ali raztegovanj.
    """
    # 1. Pretvorba v HSV in iskanje rumene barve
    hsv = cv2.cvtColor(slika_rgb, cv2.COLOR_RGB2HSV)
    spodnja_rumena = np.array([15, 80, 80])
    zgornja_rumena = np.array([45, 255, 255])
    maska_rumene = cv2.inRange(hsv, spodnja_rumena, zgornja_rumena)
    
    # 2. Blago čiščenje maske (odstrani šum na robovih)
    jedro = np.ones((5, 5), np.uint8)
    maska_rumene = cv2.morphologyEx(maska_rumene, cv2.MORPH_OPEN, jedro)
    maska_rumene = cv2.morphologyEx(maska_rumene, cv2.MORPH_CLOSE, jedro)
    
    # 3. Iskanje kontur
    konture, _ = cv2.findContours(maska_rumene, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not konture:
        print("Rumen listič ni bil najden.")
        return None
        
    # Vzamemo največjo konturo (naš listič)
    najvecja_kontura = max(konture, key=cv2.contourArea)
    
    # 4. Ustvarjanje maske TOČNE oblike lističa
    maska_naravna = np.zeros(slika_rgb.shape[:2], dtype=np.uint8)
    cv2.drawContours(maska_naravna, [najvecja_kontura], -1, 255, -1)
    
    # 5. Raven izrez (Bounding box) - ohrani 100% naravna razmerja slike
    x, y, w, h = cv2.boundingRect(najvecja_kontura)
    
    slika_izrez = slika_rgb[y:y+h, x:x+w].copy()
    maska_izrez = maska_naravna[y:y+h, x:x+w].copy()
    
    # 6. Odstranitev ozadja (mize) - vse, kar je zunaj neravne konture, postane belo
    slika_izrez[maska_izrez == 0] = (255, 255, 255)
    
    # 7. Postavitev lističa v ležeč položaj (če je slikan pokončno)
    if h > w:
        slika_izrez = cv2.rotate(slika_izrez, cv2.ROTATE_90_CLOCKWISE)
        
    return slika_izrez

# --- TESTIRANJE KODE ---
# Kodo lahko preizkusiš lokalno tako, da odkomentiraš spodnje vrstice:
"""
if __name__ == "__main__":
    # Naloži sliko (pazi, da je pot do slike pravilna)
    slika_bgr = cv2.imread("slika_listica.jpg")
    slika_rgb = cv2.cvtColor(slika_bgr, cv2.COLOR_BGR2RGB)
    
    izrezan_listic = izrezi_po_konturi(slika_rgb)
    
    if izrezan_listic is not None:
        # OpenCV za prikaz uporablja BGR
        izrezan_prikaz = cv2.cvtColor(izrezan_listic, cv2.COLOR_RGB2BGR)
        cv2.imshow("Cist izrez", izrezan_prikaz)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
"""
