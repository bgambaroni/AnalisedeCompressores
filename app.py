import streamlit as st
import tempfile
import dsp_core as dsp
import matplotlib.pyplot as plt


st.set_page_config(page_title="Analisador de Compressores", layout="centered")

st.title("🎛️ Análise Comparativa de Compressores")

st.markdown("### 🔴 Referência (REF)")
ref_file = st.file_uploader(
    "Escolha o áudio de referência",
    type=["wav"],
    accept_multiple_files=False
)

st.markdown("### 🟣 Outros Compressores")
test_files = st.file_uploader(
    "Escolha os áudios para comparar",
    type=["wav"],
    accept_multiple_files=True
)

if st.button("▶ Analisar"):
    if ref_file is None or len(test_files) == 0:
        st.error("Carregue a referência e ao menos um áudio para comparação.")
    else:
        with st.spinner("Analisando..."):

            # --- Referência ---
            with tempfile.NamedTemporaryFile(delete=False) as f:
                f.write(ref_file.read())
                ref_path = f.name

            ref = dsp.rms_normalize(dsp.load_audio(ref_path))
            ref_cent, ref_roll = dsp.spectral_features(ref)
            ref_cf = dsp.crest_factor(ref)

            st.success("Referência carregada!")

            # --- Loop TEST ---
            for tf in test_files:
                with tempfile.NamedTemporaryFile(delete=False) as f:
                    f.write(tf.read())
                    test_path = f.name

                test = dsp.rms_normalize(dsp.load_audio(test_path))
                ref_m, test_m = dsp.match_length(ref, test)

                # Métricas
                test_cent, test_roll = dsp.spectral_features(test_m)
                test_cf = dsp.crest_factor(test_m)
                dist = dsp.spectral_distance(ref_m, test_m)

                st.markdown("---")
                st.subheader(tf.name)

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("### 🔴 Referência")
                    st.write(f"Centroid: {ref_cent:.1f} Hz")
                    st.write(f"Rolloff: {ref_roll:.1f} Hz")
                    st.write(f"Crest Factor: {ref_cf:.2f}")

                with col2:
                    st.markdown("### 🟣 Teste")
                    st.write(f"Centroid: {test_cent:.1f} Hz")
                    st.write(f"Rolloff: {test_roll:.1f} Hz")
                    st.write(f"Crest Factor: {test_cf:.2f}")

                st.markdown(f"**Distância espectral vs REF:** `{dist:.4f}`")

                # --- Gráfico REF vs TEST ---
                freqs_ref, spec_ref = dsp.mean_spectrum(ref_m)
                freqs_test, spec_test = dsp.mean_spectrum(test_m)

                fig, ax = plt.subplots(figsize=(7, 4))
                ax.plot(freqs_ref, spec_ref, label="REF", linewidth=2)
                ax.plot(freqs_test, spec_test, label="TEST", linestyle="--")

                ax.set_xscale("log")
                ax.set_xlim(20, 20000)
                ax.set_xlabel("Frequência (Hz)")
                ax.set_ylabel("Magnitude (dB)")
                ax.set_title("Espectro Médio — REF vs TEST")
                ax.grid(True, which="both", ls="--", alpha=0.3)
                ax.legend()

                st.pyplot(fig)
