import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Uvozimo našo logiko iz prve datoteke
from obdelava import analiziraj_sliko

# --- 1. MODERNIZIRANE NASTAVITVE IN SEJNO STANJE ---
st.set_page_config(
    page_title="Analiza škropljenja",
    layout="centered"
)

# Inicializacija sejnega stanja (Session State) za shranjevanje rezultatov več lističev
if 'zgodovina' not in st.session_state:
    st.session_state.zgodovina = []
    st.session_state.stevec = 1

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

@st.cache_data
def procesiraj_sliko(slika_array):
    return analiziraj_sliko(slika_array)

# Funkcija za pripravo datoteke za izvoz
def ustvari_csv(podatki):
    glava = "Vzorec,Stevilo_kapljic,Pokritost_kap_na_cm2\n"
    vrstice = [f"{p['Vzorec']},{p['Število kapljic']},{p['Pokritost (kap/cm²)']}" for p in podatki]
    return (glava + "\n".join(vrstice)).encode('utf-8')

# --- 2. UPORABNIŠKI VMESNIK ---
st.markdown("### 🌾 Analiza škropljenja")
st.caption("Hitra in natančna terenska analiza vodoobčutljivih lističev.")

# Uporabimo samo klasičen uploader, ki na telefonu avtomatsko sproži OS meni za kamero/galerijo
nalozena_slika = st.file_uploader(
    "Naložite sliko ali odprite kamero", 
    type=['jpg', 'jpeg', 'png'], 
    label_visibility="collapsed"
)

if nalozena_slika is not None:
    image = Image.open(nalozena_slika)
    slika_rgb = np.array(image)
    
    with st.spinner("Obdelujem podatke (izločam kapljice < 50 µm)..."):
        status, izrez, maska, st_kapljic, pokritost = procesiraj_sliko(slika_rgb)
        
    if status == "NAPAKA":
        st.error(izrez) 
    else:
    st.markdown("#### Rezultati analize")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Število kapljic", f"{st_kapljic}")
        with col2:
            st.metric("Pokritost", f"{pokritost:.1f} kap/cm²")
