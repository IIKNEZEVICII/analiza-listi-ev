import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Uvozimo našo logiko iz prve datoteke
from obdelava import analiziraj_sliko

# --- 1. MODERNIZIRANE NASTAVITVE IN SEJNO STANJE ---
st.set_page_config(
    page_title="Analiza Škropljenja",
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

# Funkcija za pripravo datoteke za izvoz (Excel / CSV)
def ustvari_csv(podatki):
    glava = "Vzorec,Stevilo_kapljic,Pokritost_kap_na_cm2\n"
    vrstice = [f"{p['Vzorec']},{p['Število kapljic']},{p['Pokritost (kap/cm²)']}" for p in podatki]
    return (glava + "\n".join(vrstice)).encode('utf-8')

# --- 2. UPORABNIŠKI VMESNIK ---
st.markdown("### 🌱 Analiza Škropljenja")
st.caption("Hitra in natančna terenska analiza vodoobčutljivih lističev.")

# Izbira načina vnosa (Kamera ali Galerija)
nacin_vnosa = st.radio("Izberite način vnosa slike:", ["📁 Galerija", "📷 Kamera"], horizontal=True)

nalozena_slika = None
if nacin_vnosa == "📁 Galerija":
    nalozena_slika = st.file_uploader("Naložite sliko lističa", type=['jpg', 'jpeg', 'png'], label_visibility="collapsed")
else:
    nalozena_slika = st.camera_input("Posnemite sliko lističa", label_visibility="collapsed")

if nalozena_slika is not None:
    image = Image.open(nalozena_slika)
    slika_rgb = np.array(image)
    
    with st.spinner("Obdelujem podatke (izločam kapljice < 50 µm)..."):
        status, izrez, maska, st_kapljic, pokritost = procesiraj_sliko(slika_rgb)
        
    if status == "NAPAKA":
        st.error(izrez) 
    else:
        # Prikaz in shranjevanje rezultatov
        st.markdown("#### Rezultati analize")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Število kapljic", f"{st_kapljic}")
        with col2:
            st.metric("Pokritost", f"{pokritost:.1f} kap/cm²")
            
        # Gumb za dodajanje lističa v zgodovino
        if st.button("💾 Shrani ta rezultat v tabelo"):
            st.session_state.zgodovina.append({
                "Vzorec": f"Listič {st.session_state.stevec}",
                "Število kapljic": st_kapljic,
                "Pokritost (kap/cm²)": round(pokritost, 1)
            })
            st.session_state.stevec += 1
            st.success(f"Uspešno shranjeno! (Skupaj shranjenih: {len(st.session_state.zgodovina)})")

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("#### Vizualni pregled detekcije")
        
        prosojnost = st.slider(
            "Izberite raven prikaza črno-bele maske (0 = original, 1 = maska):", 
            min_value=0.0, max_value=1.0, value=0.5, step=0.05
        )
        
        maska_rgb = cv2.cvtColor(maska, cv2.COLOR_GRAY2RGB)
        prikazna_slika = cv2.addWeighted(izrez, 1.0 - prosojnost, maska_rgb, prosojnost, 0)
        st.image(prikazna_slika, use_container_width=True, channels="RGB")
        
# --- 3. PRIKAZ TABELE IN IZVOZ PODATKOV ---
if st.session_state.zgodovina:
    st.markdown("---")
    st.markdown("#### 📋 Shranjeni rezultati (Tabela)")
    
    # Prikažemo vgrajeno Streamlit tabelo
    st.dataframe(st.session_state.zgodovina, use_container_width=True)
    
    # Gumb za prenos podatkov na napravo
    csv_podatki = ustvari_csv(st.session_state.zgodovina)
    st.download_button(
        label="⬇️ Prenesi rezultate (.csv)",
        data=csv_podatki,
        file_name="analiza_skropljenja_rezultati.csv",
        mime="text/csv"
    )
