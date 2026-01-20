import librosa
import numpy as np

TARGET_SR = 44100

def load_audio(path):
    y, sr = librosa.load(path, sr=TARGET_SR, mono=True)
    return y

def rms_normalize(y):
    return y / (np.sqrt(np.mean(y**2)) + 1e-9)

def match_length(y1, y2):
    n = min(len(y1), len(y2))
    return y1[:n], y2[:n]

def crest_factor(y):
    return np.max(np.abs(y)) / (np.sqrt(np.mean(y**2)) + 1e-9)

def spectral_features(y, sr=TARGET_SR):
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    return centroid, rolloff

def spectral_distance(y_ref, y_test):
    D_ref = np.abs(librosa.stft(y_ref, n_fft=4096))
    D_test = np.abs(librosa.stft(y_test, n_fft=4096))
    return np.mean(np.abs(D_ref - D_test))

import numpy as np
import librosa

def mean_spectrum(y, sr=44100, n_fft=4096):
    S = np.abs(librosa.stft(y, n_fft=n_fft))
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    mean_db = np.mean(S_db, axis=1)
    freqs = np.linspace(0, sr / 2, len(mean_db))
    return freqs, mean_db