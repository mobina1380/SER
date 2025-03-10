
import os
import numpy as np
import librosa
import librosa.display

# Define paths and labels
path = 'shemo'  # Path to your audio files
emo_codes = {"A": 0, "W": 1, "H": 2, "S": 3, "N": 4, "F": 5}
emo_labels = ["anger", "surprise", "happiness", "sadness", "neutral", "fear"]

# Target length in seconds
TARGET_LENGTH = 7.52
FRAME_LENGTH = int(0.032 * 16000)  # 32 milliseconds
HOP_LENGTH = FRAME_LENGTH // 2     # 50% overlap
N_FFT = FRAME_LENGTH

def get_emotion_label(file_name):
    emo_code = file_name[3]
    return emo_codes[emo_code]

def preprocess_audio(file_path):
    # Load audio file with 16kHz sample rate
    y, sr = librosa.load(file_path, sr=16000)

    # Adjust length to 7.52 seconds (120320 samples at 16kHz)
    target_samples = int(TARGET_LENGTH * sr)
    if len(y) > target_samples:
        y = y[:target_samples]  # Truncate
    elif len(y) < target_samples:
        y = np.pad(y, (0, target_samples - len(y)), 'constant')  # Pad with zeros

    return y, sr

def extract_features(y, sr):
    # Compute MFCCs (20 coefficients) for the entire signal
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, hop_length=HOP_LENGTH, n_fft=N_FFT)
    
    # Delta and delta-delta of MFCCs across the entire MFCC matrix
    mfcc_delta = librosa.feature.delta(mfccs, width=3)
    mfcc_delta2 = librosa.feature.delta(mfccs, order=2, width=3)
    
    # Mean and standard deviation for MFCCs, delta, and delta-delta
    mfccs_mean = mfccs.mean(axis=1)
    mfcc_delta_mean = mfcc_delta.mean(axis=1)
    mfcc_delta2_mean = mfcc_delta2.mean(axis=1)
    mfccs_std = mfccs.std(axis=1)
    mfcc_delta_std = mfcc_delta.std(axis=1)
    mfcc_delta2_std = mfcc_delta2.std(axis=1)

    # Extract other features
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=HOP_LENGTH, n_fft=N_FFT).mean()
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=HOP_LENGTH, n_fft=N_FFT).mean()
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=HOP_LENGTH, n_fft=N_FFT).mean()
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=HOP_LENGTH, n_fft=N_FFT).mean()
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y, hop_length=HOP_LENGTH).mean()
    
    # Combine all features
    return np.hstack([spectral_centroid, spectral_bandwidth, spectral_contrast, spectral_rolloff,
                      zero_crossing_rate, mfccs_mean, mfcc_delta_mean, mfcc_delta2_mean,
                      mfccs_std, mfcc_delta_std, mfcc_delta2_std])

def opensmile_Functionals():
    features = []
    emotions = []
    for file in os.listdir(path):
        if emo_labels[get_emotion_label(file)] != 'fear':
            file_path = os.path.join(path, file)
            y, sr = preprocess_audio(file_path)
            feature_vector = extract_features(y, sr)
            features.append(feature_vector)
            emotions.append(get_emotion_label(file))
    features = np.array(features)
    emotions = np.array(emotions)
    return features, emotions

if __name__ == "__main__":
    features, emotions = opensmile_Functionals()
    np.save('NewVersion/features2.npy', features)
    np.save('NewVersion/emotions2.npy', emotions)
