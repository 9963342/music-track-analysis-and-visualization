import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

st.title("Music Track Analysis and Visualization")

uploaded_file = st.file_uploader("Upload audio file", type=["mp3", "wav"])

if uploaded_file is not None:
    with open("temp_audio.mp3", "wb") as f:
        f.write(uploaded_file.read())

    y, sr = librosa.load("temp_audio.mp3")

    st.subheader("Waveform")
    fig, ax = plt.subplots(figsize=(10, 3))
    librosa.display.waveshow(y, sr=sr, ax=ax)
    st.pyplot(fig)

    st.subheader("Spectrogram")
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    fig, ax = plt.subplots(figsize=(10, 3))
    img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', ax=ax)
    plt.colorbar(img, ax=ax)
    st.pyplot(fig)

    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    st.write("Tempo (BPM):", tempo)

    zcr = librosa.feature.zero_crossing_rate(y)
    st.write("Zero Crossing Rate:", np.mean(zcr))

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    st.write("Spectral Centroid:", np.mean(centroid))

    energy = np.sum(y ** 2) / len(y)
    st.write("Energy:", energy)