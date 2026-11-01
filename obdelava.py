import streamlit as st
import cv2
import numpy as np
from PIL import Image
from obdelava import analiziraj_sliko

st.set_page_config(page_title="Analiza škropljenja", layout="centered")

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

def ustvari_csv(podatki):
    glava = "Vzorec,Stevilo_odtisov,Pokritost_odstotek\n"
    vrstice = [f"{p['Vzorec']},{p['Število odtisov']},{p['Pokritost (%)']}" for p in podatki]
    return (glava + "\n".join(vrstice)).encode('utf-8')

st.markdown("### 🌾 Analiza škropljenja")
st.caption("Aplikacija za hiter pregled kakovosti škropljenja.")

nalozena_slika = st.file_uploader(
    "Naložite sliko ali odprite kamero", 
    type=['jpg', 'jpeg', 'png'], 
    label_visibility="collapsed"
)

if nalozena_slika is not None:
    image = Image.open(nalozena_slika)
    slika_rgb = np.array(image)
    
    with st.spinner("Obdelujem podatke (izločam kapljice < 50 µm)..."):
        status, izrez, maska, stevilo_odtisov, pokritost = procesiraj_sliko(slika_rgb)
        
    if status == "NAPAKA":
        st.error(izrez) 
    else:
        st.markdown("#### Rezultati analize")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Število odtisov", f"{stevilo_odtisov}")
        with col2:
            st.metric("Pokritost", f"{pokritost:.2f} %")
            
        if st.button("💾 Shrani ta rezultat v tabelo"):
            st.session_state.zgodovina.append({
                "Vzorec": f"Listič {st.session_state.stevec}",
                "Število odtisov": stevilo_odtisov,
                "Pokritost (%)": round(pokritost, 2)
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
        
if st.session_state.zgodovina:
    st.markdown("---")
    st.markdown("#### 📋 Shranjeni rezultati (Tabela)")
    st.dataframe(st.session_state.zgodovina, use_container_width=True)
    csv_podatki = ustvari_csv(st.session_state.zgodovina)
    st.download_button(
        label="⬇️ Prenesi rezultate (.csv)",
        data=csv_podatki,
        file_name="analiza_skropljenja_rezultati.csv",
        mime="text/csv"
    )

if nalozena_slika is None:
    st.info("Za analizo naložite obstoječo sliko ali naredite novo.")
