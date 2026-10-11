import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Uvozimo našo logiko iz prve datoteke
from obdelava import analiziraj_sliko

# --- 1. MODERNIZIRANE NASTAVITVE STRANI ---
st.set_page_config(
    page_title="Analiza Škropljenja",
    layout="centered"
)

# Skrijemo privzeti meni, nogo in dodamo CSS za kartice
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

.metric-card {
    background-color: rgba(128,128,128,0.05);
    border: 1px solid rgba(128,128,128,0.15);
    padding: 16px;
    border-radius: 12px;
    text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.02);
}
</style>
""", unsafe_allow_html=True)

# --- 2. FUNKCIJA S PREDPOMNENJEM (CACHE) ---
# Ta dekorator Streamlitu pove, naj si rezultat za posamezno sliko zapomni.
# Brez tega bi se ob vsakem premiku drsnika koda izvajala od začetka.
@st.cache_data
def procesiraj_sliko(slika_array):
    return analiziraj_sliko(slika_array)

# --- 3. UPORABNIŠKI VMESNIK ---
st.markdown("### Analiza Škropljenja")
st.caption("Hitra in natančna terenska analiza vodoobčutljivih lističev.")

nalozena_slika = st.file_uploader(
    "Naložite sliko lističa",
    type=['jpg', 'jpeg', 'png'],
    label_visibility="collapsed"
)

if nalozena_slika is not None:
    # Pretvorba slike v RGB (Red Green Blue) format, ki ga razume OpenCV
    image = Image.open(nalozena_slika)
    slika_rgb = np.array(image)
    
    with st.spinner("Obdelujem podatke (izločam kapljice < 50 µm)..."):
        # Klic ločene logike iz obdelava.py
        status, izrez, maska, st_kapljic, pokritost = procesiraj_sliko(slika_rgb)
        
    if status == "NAPAKA":
        # V primeru napake je v spremenljivki 'izrez' shranjeno opozorilo
        st.error(izrez) 
    else:
        st.markdown("#### Rezultati analize")
        
        # Prikaz rezultatov z jasnimi in čistimi metrikami
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Število kapljic", f"{st_kapljic}")
        with col2:
            st.metric("Pokritost", f"{pokritost:.1f} kap/cm²")
            
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("#### Vizualni pregled detekcije")
        
        # --- 4. DRSNIK IN PREKRIVANJE SLIKE ---
        prosojnost = st.slider(
            "Izberite raven prikaza črno-bele maske (0 = original, 1 = maska):", 
            min_value=0.0, max_value=1.0, value=0.5, step=0.05
        )
        
        # Pretvorba 1-kanalne črno-bele maske v 3-kanalno sliko, da se lahko prekriva z barvno
        maska_rgb = cv2.cvtColor(maska, cv2.COLOR_GRAY2RGB)
        
        # Matematično združevanje slik glede na nastavljeno prosojnost
        prikazna_slika = cv2.addWeighted(izrez, 1.0 - prosojnost, maska_rgb, prosojnost, 0)
        
        # Izris končne slike
        st.image(prikazna_slika, use_container_width=True, channels="RGB")
        
else:
    st.info("Tapnite zgoraj za izbiro slike ali zajem fotografije s telefonovo kamero.")
