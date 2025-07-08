import streamlit as st
import joblib
import pandas as pd

# --- Load model dan komponen ---
model = joblib.load('RidgeClassifier - Ukulele by Yousician.pkl')
vectorizer = joblib.load('tfidf_vectorizer_Ukulele by Yousician.pkl')
label_encoder = joblib.load('label_encoder_Ukulele by Yousician.pkl')

# --- Judul App ---
st.title("🎵 Sentiment Analysis - Ukulele by Yousician")

# --- Pilih Mode ---
st.header("Pilih Metode Input")
input_mode = st.radio("Mode Input:", ["📝 Input Manual", "📁 Upload CSV"])

# ========================================
# 📌 MODE 1: INPUT MANUAL
# ========================================
if input_mode == "📝 Input Manual":
    st.subheader("Masukkan 1 Review Pengguna")
    user_review = st.text_area("Tulis review di sini:")

    if st.button("Prediksi Sentimen"):
        if user_review.strip() == "":
            st.warning("🚨 Silakan masukkan teks terlebih dahulu.")
        else:
            vec = vectorizer.transform([user_review])
            pred = model.predict(vec)
            label = label_encoder.inverse_transform(pred)[0]
            st.success(f"🎯 Sentimen Prediksi: **{label}**")

# ========================================
# 📁 MODE 2: UPLOAD CSV
# ========================================
else:
    st.subheader("Upload File CSV Review")
    uploaded_file = st.file_uploader("Pilih file CSV (dengan kolom 'review')", type=['csv'])

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)

            # Validasi kolom
            if 'review' not in df.columns:
                st.error("❌ File harus memiliki kolom 'review'.")
            else:
                # Prediksi
                X_vec = vectorizer.transform(df['review'].fillna(""))
                y_pred = model.predict(X_vec)
                df['predicted_sentiment'] = label_encoder.inverse_transform(y_pred)

                st.success("✅ Prediksi berhasil!")
                st.dataframe(df.head())

                # Download hasil
                csv_result = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Hasil",
                    data=csv_result,
                    file_name="predicted_reviews.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"Terjadi error saat membaca file: {e}")
