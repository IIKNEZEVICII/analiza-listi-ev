import streamlit as st
import cv2
import numpy as np
from PIL import Image

from obdelava import analiziraj_sliko

st.set_page_config(
    page_title="Analiza škropljenja (1cm x 1cm)",
    layout="centered"
)

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

# Spremenjeno ime prisili nov cache v oblaku, da naloži novo detekcijo in debelejše okvirje!
@st.cache_data
def procesiraj_sliko_v5(slika_array):
    return analiziraj_sliko(slika_array)

def ustvari_csv(podatki):
    glava = "Vzorec,Povprecje_kapljic_na_cm2,Povprecna_pokritost_procent\n"
    vrstice = [f"{p['Vzorec']},{p['Povp. število kapljic (na cm²)']},{p['Povp. pokritost (%)']}" for p in podatki]
    return (glava + "\n".join(vrstice)).encode('utf-8')

st.markdown("### 🌾 Analiza škropljenja (3x 1cm²)")
st.caption("Aplikacija naključno izbere tri ločene kvadrate velikosti 1x1 cm in izračuna povprečje.")

nalozena_slika = st.file_uploader(
    "Naložite sliko ali odprite kamero", 
    type=['jpg', 'jpeg', 'png'], 
    label_visibility="collapsed"
)

if nalozena_slika is not None:
    image = Image.open(nalozena_slika)
    slika_rgb = np.array(image)
    
    with st.spinner("Analiziram listič in iščem ločene 1x1 cm vzorce..."):
        status, slika_oznake, rezultati_kvadratkov = procesiraj_sliko_v5(slika_rgb)
        
    if status == "NAPAKA":
        st.error(slika_oznake) 
    else:
        st.markdown("#### Detektiran listič z označenimi vzorci")
        st.image(slika_oznake, use_container_width=True, channels="RGB")
        
        povp_kapljic = sum(k['st_kapljic'] for k in rezultati_kvadratkov) / 3
        povp_pokritost = sum(k['pokritost'] for k in rezultati_kvadratkov) / 3
        
        st.markdown("#### Skupno povprečje izrezov")
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric("Povp. število kapljic (na 1 cm²)", f"{povp_kapljic:.0f}")
        with col_m2:
            st.metric("Povp. pokritost", f"{povp_pokritost:.2f} %")
            
        if st.button("💾 Shrani povprečje v tabelo"):
            st.session_state.zgodovina.append({
                "Vzorec": f"Listič {st.session_state.stevec}",
                "Povp. število kapljic (na cm²)": round(povp_kapljic),
                "Povp. pokritost (%)": round(povp_pokritost, 2)
            })
            st.session_state.stevec += 1
            st.success(f"Shranjeno! (Skupaj: {len(st.session_state.zgodovina)})")

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("#### Posamezni ločeni 1x1 cm izrezi")
        
        prosojnost = st.slider(
            "Izberite raven prikaza maske na vzorcih (0 = original, 1 = maska):", 
            min_value=0.0, max_value=1.0, value=0.4, step=0.1
        )
        
        stolpci = st.columns(3)
        for i, (stolpec, kvadrat) in enumerate(zip(stolpci, rezultati_kvadratkov)):
            with stolpec:
                st.write(f"**Vzorec {i+1}**")
                
                maska_rgb = cv2.cvtColor(kvadrat['maska'], cv2.COLOR_GRAY2RGB)
                prikaz = cv2.addWeighted(kvadrat['rgb'], 1.0 - prosojnost, maska_rgb, prosojnost, 0)
                
                st.image(prikaz, use_container_width=True, channels="RGB")
                st.write(f"Kapljic: **{kvadrat['st_kapljic']}**")
                st.write(f"Pokritost: **{kvadrat['pokritost']:.2f}%**")

if st.session_state.zgodovina:
    st.markdown("---")
    st.markdown("#### 📋 Shranjeni rezultati (Tabela)")
    st.dataframe(st.session_state.zgodovina, use_container_width=True)
    
    csv_podatki = ustvari_csv(st.session_state.zgodovina)
    st.download_button(
        label="⬇️ Prenesi rezultate (.csv)",
        data=csv_podatki,
        file_name="analiza_skropljenja_1x1cm.csv",
        mime="text/csv"
    )
