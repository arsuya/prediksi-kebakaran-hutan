import streamlit as st
import pandas as pd

def run():
    st.title("🔥 Project Deep Learning: Deteksi Kebakaran Hutan dengan Computer Vision")
    st.markdown("---")

    # Background
    st.markdown("### Latar Belakang")
    st.markdown("""
    Kebakaran hutan merupakan salah satu bencana lingkungan yang dapat menimbulkan kerugian besar terhadap ekosistem, kesehatan manusia, dan ekonomi.
    Salah satu tantangan dalam mitigasi kebakaran hutan adalah keterlambatan dalam mendeteksi awal munculnya api.
    Oleh karena itu, dibutuhkan sistem otomatis yang mampu mendeteksi kebakaran dengan cepat dan akurat menggunakan pendekatan berbasis Computer Vision.
    """)

    # Problem Statement
    st.markdown("### Rumusan Masalah")
    st.markdown("""
    Bagaimana membangun sistem deteksi visual yang mampu membedakan gambar hutan yang terbakar dan tidak terbakar agar deteksi dini bisa dilakukan dan respon lapangan menjadi lebih cepat?
    """)

    # Objective
    st.markdown("### Tujuan")
    st.markdown("""
    Mengembangkan model Deep Learning berbasis Convolutional Neural Network (CNN) untuk mengklasifikasikan citra hutan 
    menjadi dua kelas: **Kebakaran** dan **Tidak Kebakaran**, dengan fokus evaluasi menggunakan metrik **akurasi** 
    """)

    # Model Overview
    st.markdown("### Ringkasan Model")
    st.markdown("""
    Dua arsitektur CNN telah dibangun dan diuji:
    - **Model Awal (Baseline)**: terdiri dari 3 lapisan konvolusi, 1 dropout, dan 2 dense layer.
    - **Model Improvement**: arsitektur yang lebih ringan dan cepat dilatih, namun menghasilkan akurasi lebih tinggi.

    Model terbaik adalah **Model Improvement**, dengan akurasi **91%** dan ukuran model yang lebih kecil serta waktu pelatihan yang lebih singkat dibanding baseline.
    """)

    # Dataset Info
    st.markdown("### Informasi Dataset")
    st.markdown("""
    Dataset terdiri dari dua kategori gambar:
    - **Fire**: Gambar yang memperlihatkan hutan terbakar
    - **No Fire**: Gambar hutan tanpa kebakaran

    Dataset dibagi menjadi dua bagian:
    - **Training & Validation Set**: 928 gambar kelas *fire* dan 904 gambar kelas *no fire*
    - **Testing Set**: Data uji yang digunakan untuk mengevaluasi performa model
    """)

    dataset_info = pd.DataFrame([
        ["Jumlah Gambar - Fire (Train/Val)", "928"],
        ["Jumlah Gambar - No Fire (Train/Val)", "904"],
        ["Jenis Klasifikasi", "Binary (Fire / No Fire)"]
    ], columns=["Informasi", "Detail"])
    st.dataframe(dataset_info, use_container_width=True)

    # Cara Pakai
    st.markdown("### Cara Menggunakan Dashboard")
    st.markdown("""
    1. Buka halaman **EDA** untuk melihat eksplorasi dan visualisasi data.
    2. Gunakan halaman **Prediksi** untuk mengunggah gambar dan melihat hasil klasifikasinya.
    3. Hasil klasifikasi dapat dijadikan referensi awal dalam sistem deteksi kebakaran hutan secara otomatis.
    """)

if __name__ == '__main__':
    run()
