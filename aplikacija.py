import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image # PIL (Python Imaging Library) za branje naloženih slik

# Nastavitev širokega zaslona za lepši prikaz grafov
st.set_page_config(page_title="Analiza Škropljenja", layout="wide")

def analiziraj_sliko(slika_rgb):
    """Funkcija, ki vsebuje najin celoten algoritem (koraki 2 do 5)"""
    
    # --- KORAK 2: AVTOMATSKI IZREZ ---
    hsv_orig = cv2.cvtColor(slika_rgb, cv2.COLOR_RGB2HSV)
    spodnja_rumena = np.array([15, 80, 80])
    zgornja_rumena = np.array([45, 255, 255])
    maska_rumene = cv2.inRange(hsv_orig, spodnja_rumena, zgornja_rumena)

    konture, _ = cv2.findContours(maska_rumene, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not konture:
        return None, "Napaka: Na sliki ni rumenega lističa."
        
    najvecja_kontura = max(konture, key=cv2.contourArea)
    obseg = cv2.arcLength(najvecja_kontura, True)

    oglisca = None
    for faktor in np.arange(0.01, 0.1, 0.005):
        temp_oglisca = cv2.approxPolyDP(najvecja_kontura, faktor * obseg, True)
        if len(temp_oglisca) == 4:
            oglisca = temp_oglisca
            break

    if oglisca is None or len(oglisca) != 4:
        return None, "Napaka: Prepoznava 4 robov lističa ni uspela. Poskusite z drugo sliko."

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
    slika_izrez = slika_izrez[3:-3, 3:-3]
    um_na_piko = 1000 / pike_na_mm_cilj

    # --- KORAK 3: DETEKCIJA KAPLJIC ---
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

    # --- KORAK 4: ZAPOLNJEVANJE LUKENJ IN ČIŠČENJE (Konveksna lupina) ---
    jedro_zapiranje = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    maska_sklenjena = cv2.morphologyEx(maska_kapljic, cv2.MORPH_CLOSE, jedro_zapiranje)

    konture_maske, _ = cv2.findContours(maska_sklenjena, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    maska_zapolnjena = np.zeros_like(maska_sklenjena)
    
    # Konveksna lupina (Convex Hull) za zapolnitev vseh lis
    for kontura in konture_maske:
        lupina = cv2.convexHull(kontura)
        cv2.drawContours(maska_zapolnjena, [lupina], -1, 255, -1)
    
    jedro_odpiranje = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    maska_cista = cv2.morphologyEx(maska_zapolnjena, cv2.MORPH_OPEN, jedro_odpiranje)

    # --- KORAK 5: IZRAČUN POKRITOSTI IN PREMERA ---
    koncne_konture, _ = cv2.findContours(maska_cista, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    stevilo_kapljic = len(koncne_konture)
    povrsina_vseh_kapljic_px = 0
    premeri_um = []
    um2_na_piko = um_na_piko ** 2

    for kontura in koncne_konture:
        povrsina_px = cv2.contourArea(kontura)
        povrsina_vseh_kapljic_px += povrsina_px
        povrsina_um2 = povrsina_px * um2_na_piko
        if povrsina_um2 > 0:
            premer_um = 2 * np.sqrt(povrsina_um2 / np.pi)
            premeri_um.append(premer_um)

    povrsina_listica_px = slika_izrez.shape[0] * slika_izrez.shape[1]
    odstotek_pokritosti = (povrsina_vseh_kapljic_px / povrsina_listica_px) * 100

    # Priprava slike za prikaz
    zoom_y_start, zoom_y_end = int(slika_izrez.shape[0]*0.2), int(slika_izrez.shape[0]*0.5)
    zoom_x_start, zoom_x_end = int(slika_izrez.shape[1]*0.2), int(slika_izrez.shape[1]*0.8)
    zoom_original = slika_izrez[zoom_y_start:zoom_y_end, zoom_x_start:zoom_x_end]
    zoom_maska = maska_cista[zoom_y_start:zoom_y_end, zoom_x_start:zoom_x_end]
    
    prikaz_zooma = zoom_original.copy()
    prikaz_zooma[zoom_maska > 0] = [255, 0, 0] 

    # Sestavimo grafikon
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6), gridspec_kw={'width_ratios': [1, 1, 1.5]})
    ax1.imshow(slika_izrez)
    ax1.set_title('Originalni izrez')
    ax1.axis('off')
    
    ax2.imshow(maska_cista, cmap='gray')
    ax2.set_title('Detektirana maska')
    ax2.axis('off')
    
    ax3.imshow(prikaz_zooma)
    ax3.set_title('Bližnji posnetek (rdeča = detektirano)')
    ax3.axis('off')
    plt.tight_layout()

    # Sestavimo histogram
    fig_hist, ax_hist = plt.subplots(figsize=(10, 4))
    if stevilo_kapljic > 0:
        ax_hist.hist(premeri_um, bins=40, color='#1f77b4', edgecolor='black', alpha=0.7)
        ax_hist.set_title('Porazdelitev velikosti kapljic')
        ax_hist.set_xlabel('Premer kapljice (mikrometer - $\mu$m)')
        ax_hist.set_ylabel('Število kapljic')
        ax_hist.grid(axis='y', linestyle='--', alpha=0.7)

    return fig, fig_hist, stevilo_kapljic, odstotek_pokritosti, premeri_um


# ==================== STREAMLIT UPORABNIŠKI VMESNIK ====================
st.title("Aplikacija za analizo vodno občutljivih lističev")
st.markdown("Naložite sliko lističa. Algoritem bo samodejno prepoznal robove, ga zravnal in analiziral kapljice škropiva (minimalno ~50 $\mu$m).")

nalozena_slika = st.file_uploader("Naloži sliko lističa (slikaj s kamero ali izberi iz galerije)", type=['jpg', 'jpeg', 'png'])

if nalozena_slika is not None:
    # Pretvorba slike iz brskalnika v obliko primerno za OpenCV
    image = Image.open(nalozena_slika)
    slika_rgb = np.array(image)
    
    st.write("Slike se obdeluje. Prosimo, počakajte...")
    
    # Zaženemo najino funkcijo
    rezultat = analiziraj_sliko(slika_rgb)
    
    if len(rezultat) == 2: # Zgodila se je napaka (vrnila je samo (None, napaka))
        st.error(rezultat[1])
    else:
        fig, fig_hist, st_kapljic, pokritost, premeri = rezultat
        
        st.success("Analiza je uspešno zaključena!")
        
        # Prikaz glavnih številk v lepih "metrikah" (UI element)
        col1, col2, col3 = st.columns(3)
        col1.metric(label="Število kapljic", value=st_kapljic)
        col2.metric(label="Odstotek pokritosti", value=f"{pokritost:.2f} %")
        if premeri:
            col3.metric(label="Povprečni premer", value=f"{np.mean(premeri):.0f} µm")
        else:
            col3.metric(label="Povprečni premer", value="0 µm")
            
        st.write("---")
        
        # Prikaz obdelanih slik in grafikonov
        st.subheader("Vizualizacija detekcije")
        st.pyplot(fig)
        
        st.subheader("Porazdelitev kapljic")
        st.pyplot(fig_hist)