import os
import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import librosa

processor = AutoProcessor.from_pretrained("steja/whisper-small-persian")
model = AutoModelForSpeechSeq2Seq.from_pretrained("steja/whisper-small-persian")

# Define the path to your SHEMo dataset
dataset_path = "shemo"

# Define the output file for 10-best sentences
output_file = "files/sentences_10_best.txt"

# Ensure output directory exists
os.makedirs(os.path.dirname(output_file), exist_ok=True)

def get_best_10_sentences(audio_file, device):
    # Load the audio file and preprocess it
    audio, sr = librosa.load(audio_file, sr=16000)  # Load audio with sample rate of 16000
    inputs = processor(audio, return_tensors="pt", sampling_rate=16000)

    # Send input to the correct device
    input_features = inputs.input_features.to(device)

    # Ensure the model is on the same device
    model.to(device)

    # Generate the 10-best sentences using beam search
    with torch.no_grad():
        generated_ids = model.generate(
            input_features,
            num_beams=10,
            num_return_sequences=10,
            early_stopping=True
        )

    # Decode the generated sentences
    best_10_sentences = processor.batch_decode(generated_ids, skip_special_tokens=True)

    return best_10_sentences

def process_dataset(dataset_path):
    all_sentences = []

    # Determine the device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Process each audio file in the dataset
    for audio_file in os.listdir(dataset_path):
        if audio_file.endswith(".wav"):  # Assuming your files are in .wav format
            audio_path = os.path.join(dataset_path, audio_file)
            print(f"Processing {audio_path}...")

            # Get the 10-best sentences
            best_10 = get_best_10_sentences(audio_path, device)

            # Add them to the list
            all_sentences.extend(best_10)

    return all_sentences

def save_sentences_to_file(sentences, output_file):
    # Save the sentences to a text file
    with open(output_file, 'w', encoding='utf-8') as f:
        for sentence in sentences:
            f.write(sentence + "\n")

if __name__ == "__main__":
    # Process the SHEMo dataset and get all 10-best sentences
    sentences_10_best = process_dataset(dataset_path)

    # Save the 10-best sentences to the specified output file
    save_sentences_to_file(sentences_10_best, output_file)

    print(f"10-best sentences have been saved to {output_file}.")
