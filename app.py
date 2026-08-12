import os
import streamlit as st
import pandas as pd
import plotly.express as px

from utils.preprocessing import preprocess_text
from utils.predictor import (
    load_model,
    predict_sentiment,
    predict_multi_aspek,
    ASPEK_LIST,
)

MODEL_DIR = os.environ.get("MODEL_REPO", "hanifzs/indobert-absa-iot-pertanian")
SENT_EMOJI = {"Positif": "🟢", "Netral": "🟡", "Negatif": "🔴"}
SENT_COLOR = {"Positif": "#2ecc71", "Netral": "#f1c40f", "Negatif": "#e74c3c"}

st.set_page_config(page_title="AgriIoT-ABSA", page_icon="🌾", layout="wide")

st.title("🌾 AgriIoT-ABSA")
st.caption(
    "Aspect-Based Sentiment Analysis komentar YouTube tentang IoT di Pertanian "
    "menggunakan IndoBERT — Hanif Zaidan Sinaga"
)


@st.cache_resource(show_spinner="Memuat model IndoBERT...")
def get_model():
    return load_model(MODEL_DIR)


try:
    get_model()
    st.success("✅ Model IndoBERT berhasil dimuat.")
except FileNotFoundError as e:
    st.error(f"❌ {e}")
    st.stop()

tab1, tab2, tab3 = st.tabs(
    ["🔍 Prediksi Komentar", "📊 Multi-Aspek", "📁 Analisis Dataset"]
)

with tab1:
    st.subheader("Prediksi Sentimen per Aspek")
    comment = st.text_area("Komentar YouTube:", height=100)
    aspek = st.selectbox("Pilih Aspek (UTAUT):", ASPEK_LIST)

    if st.button("🔮 Prediksi Sentimen", use_container_width=True):
        if not comment.strip():
            st.warning("Silakan isi komentar terlebih dahulu.")
        else:
            text_clean = preprocess_text(comment)
            st.write(f"**Hasil preprocessing:** `{text_clean}`")
            r = predict_sentiment(text_clean, aspek, MODEL_DIR)

            c1, c2, c3 = st.columns(3)
            c1.metric("Aspek", r["aspek"])
            c2.metric("Sentimen", f"{SENT_EMOJI[r['sentimen']]} {r['sentimen']}")
            c3.metric("Confidence", f"{r['confidence']:.2%}")

            prob_df = pd.DataFrame({
                "Kelas": ["Positif", "Netral", "Negatif"],
                "Probabilitas": [
                    r["prob_positif"], r["prob_netral"], r["prob_negatif"]
                ],
            })
            fig = px.bar(prob_df, x="Kelas", y="Probabilitas", color="Kelas",
                         color_discrete_map=SENT_COLOR)
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Analisis Semua Aspek Sekaligus")
    comment2 = st.text_area("Komentar YouTube:", height=100, key="input_multi")

    if st.button("🔮 Analisis Semua Aspek", use_container_width=True):
        if not comment2.strip():
            st.warning("Silakan isi komentar terlebih dahulu.")
        else:
            text_clean = preprocess_text(comment2)
            st.write(f"**Hasil preprocessing:** `{text_clean}`")
            results = predict_multi_aspek(text_clean, MODEL_DIR)
            df = pd.DataFrame(results)
            df["Sentimen"] = df["sentimen"].map(lambda s: f"{SENT_EMOJI[s]} {s}")
            df["Confidence"] = df["confidence"].map(lambda x: f"{x:.2%}")
            st.dataframe(
                df[["aspek", "Sentimen", "Confidence"]],
                use_container_width=True,
                column_config={"aspek": "Aspek UTAUT"},
            )

with tab3:
    st.subheader("Analisis Massal (Upload CSV/Excel)")
    uploaded = st.file_uploader(
        "Upload file dengan kolom komentar", type=["csv", "xlsx"]
    )

    if uploaded is not None:
        if uploaded.name.endswith(".csv"):
            df_up = pd.read_csv(uploaded)
        else:
            df_up = pd.read_excel(uploaded)

        st.write(f"📋 Jumlah baris: **{len(df_up)}**")
        st.dataframe(df_up.head(10), use_container_width=True)

        text_col = next(
            (c for c in ["text_preprocessing", "comment_text", "komentar", "text"]
             if c in df_up.columns),
            None,
        )

        if text_col is None:
            st.error(
                "Kolom teks tidak ditemukan. Gunakan salah satu nama kolom: "
                "text_preprocessing, comment_text, komentar, atau text."
            )
        else:
            aspek_batch = st.selectbox(
                "Pilih aspek untuk analisis massal:", ASPEK_LIST, key="aspek_batch"
            )

            if st.button("🚀 Analisis Semua Komentar", use_container_width=True):
                bar = st.progress(0)
                rows = []
                for i, row in df_up.iterrows():
                    raw = str(row[text_col])
                    clean = raw if text_col == "text_preprocessing" else preprocess_text(raw)
                    r = predict_sentiment(clean, aspek_batch, MODEL_DIR)
                    r["komentar_asli"] = raw
                    rows.append(r)
                    bar.progress((i + 1) / len(df_up))

                df_out = pd.DataFrame(rows)
                st.dataframe(
                    df_out[["komentar_asli", "aspek", "sentimen", "confidence"]],
                    use_container_width=True,
                )

                counts = df_out["sentimen"].value_counts().reset_index()
                counts.columns = ["sentimen", "jumlah"]
                fig2 = px.pie(
                    counts, names="sentimen", values="jumlah",
                    title="Distribusi Sentimen",
                    color="sentimen", color_discrete_map=SENT_COLOR,
                )
                st.plotly_chart(fig2, use_container_width=True)

                st.download_button(
                    "📥 Download Hasil (CSV)",
                    df_out.to_csv(index=False),
                    file_name="hasil_absa_iot.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
