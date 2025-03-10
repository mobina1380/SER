import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from PIL import Image

seed_value = 42
np.random.seed(seed_value)

# Define paths and labels
path = 'shemo'  # Path to your audio files
emo_codes = {"A": 0, "W": 1, "H": 2, "S": 3, "N": 4, "F": 5}
emo_labels = ["anger", "surprise", "happiness", "sadness", "neutral", "fear"]

def get_emotion_label(file_name):
    emo_code = file_name[3]
    return emo_codes[emo_code]

def generate_spectrogram(file_path):
    # Load the audio file
    y, sr = librosa.load(file_path, sr=16000)

    # Generate Mel-scaled power (log) spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    log_S = librosa.power_to_db(S, ref=np.max)

    # Convert to image
    plt.figure(figsize=(3, 3))
    librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel')
    plt.axis('off')
    
    # Save spectrogram image to a temporary file
    plt.savefig('images/temp_spectrogram.png', bbox_inches='tight', pad_inches=0)
    plt.close()

    # Open the image and resize to a fixed size (e.g., 128x128)
    image = Image.open('images/temp_spectrogram.png').convert('L')
    image = image.resize((128, 128))  # Resize to a fixed size
    image_array = np.array(image)

    # Flatten the image into a feature vector
    return image_array.flatten()

def process_spectrograms():
    features = []
    emotions = []
    for file in os.listdir(path):
        if emo_labels[get_emotion_label(file)] != 'fear':  # Skip 'fear' class
            file_path = os.path.join(path, file)
            feature_vector = generate_spectrogram(file_path)  # Generate spectrogram
            features.append(feature_vector)
            emotions.append(get_emotion_label(file))
    features = np.array(features)
    emotions = np.array(emotions)
    return features, emotions

if __name__ == "__main__":
    features, emotions = process_spectrograms()
    np.save('images/spectrogram_features.npy', features)  # Save spectrogram features
    np.save('images/spectrogram_emotions.npy', emotions)  # Save corresponding emotion labels
