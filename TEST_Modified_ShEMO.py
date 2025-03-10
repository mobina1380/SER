# !curl --remote-name-all https://storage.googleapis.com/danielk-files/farsi-text/merged_files/w2c_merged.txt

# !pip install parsivar
# !pip install num2fawords
# !pip install hazm

import re
import string

from hazm import *
from num2fawords import words

sentence_tokenizer = SentenceTokenizer()
_normalizer = Normalizer()

chars_to_ignore = [
    ",", "?", ".", "!", "-", ";", ":", '""', "%", "'", '"', "�",
    "#", "!", "؟", "?", "«", "»", "،", "(", ")", "؛", "'ٔ", "٬", 'ٔ', ",", "?",
    ".", "!", "-", ";", ":", '"', "“", "%", "‘", "”", "�", "–", "…", "_", "”", '“', '„',
    'ā', 'š',
    # "ء",
]

# In case of farsi
chars_to_ignore = chars_to_ignore + list(string.ascii_lowercase + string.digits)

chars_to_mapping = {
    'ك': 'ک', 'دِ': 'د', 'بِ': 'ب', 'زِ': 'ز', 'ذِ': 'ذ', 'شِ': 'ش', 'سِ': 'س', 'ى': 'ی',
    'ي': 'ی', 'أ': 'ا', 'ؤ': 'و', "ے": "ی", "ۀ": "ه", "ﭘ": "پ", "ﮐ": "ک", "ﯽ": "ی",
    "ﺎ": "ا", "ﺑ": "ب", "ﺘ": "ت", "ﺧ": "خ", "ﺩ": "د", "ﺱ": "س", "ﻀ": "ض", "ﻌ": "ع",
    "ﻟ": "ل", "ﻡ": "م", "ﻢ": "م", "ﻪ": "ه", "ﻮ": "و", 'ﺍ': "ا", 'ة': "ه",
    'ﯾ': "ی", 'ﯿ': "ی", 'ﺒ': "ب", 'ﺖ': "ت", 'ﺪ': "د", 'ﺮ': "ر", 'ﺴ': "س", 'ﺷ': "ش",
    'ﺸ': "ش", 'ﻋ': "ع", 'ﻤ': "م", 'ﻥ': "ن", 'ﻧ': "ن", 'ﻭ': "و", 'ﺭ': "ر", "ﮔ": "گ",

    # "ها": "  ها", "ئ": "ی",

    "a": " ای ", "b": " بی ", "c": " سی ", "d": " دی ", "e": " ایی ", "f": " اف ",
    "g": " جی ", "h": " اچ ", "i": " آی ", "j": " جی ", "k": " کی ", "l": " ال ",
    "m": " ام ", "n": " ان ", "o": " او ", "p": " پی ", "q": " کیو ", "r": " آر ",
    "s": " اس ", "t": " تی ", "u": " یو ", "v": " وی ", "w": " دبلیو ", "x": " اکس ",
    "y": " وای ", "z": " زد ",
    "\u200c": " ", "\u200d": " ", "\u200e": " ", "\u200f": " ", "\ufeff": " ",
}


def multiple_replace(text, chars_to_mapping):
    pattern = "|".join(map(re.escape, chars_to_mapping.keys()))
    return re.sub(pattern, lambda m: chars_to_mapping[m.group()], str(text))


def remove_special_characters(text, chars_to_ignore_regex):
    text = re.sub(chars_to_ignore_regex, '', text).lower() + " "
    return text


def normalizer(text, chars_to_ignore=chars_to_ignore, chars_to_mapping=chars_to_mapping):
    chars_to_ignore_regex = f"""[{"".join(chars_to_ignore)}]"""
    text = text.lower().strip()

    text = _normalizer.normalize(text)
    text = multiple_replace(text, chars_to_mapping)
    text = remove_special_characters(text, chars_to_ignore_regex)
    text = re.sub(" +", " ", text)
    _text = []
    for word in text.split():
        try:
            word = int(word)
            _text.append(words(word))
        except:
            _text.append(str(word))

    text = " ".join(_text) + " "

    text = text.strip() + " "

    return text

def load_sentences():
    sentences = []
    with open('w2c_merged.txt', 'r', encoding='utf-8' ) as f:
      for line in f:
        lines = sentence_tokenizer.tokenize(line)
        for l in lines:
            ll = normalizer(l)
            sentences.append(ll)
    return sentences


sents = load_sentences()
with open('w2c_cleaned.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(sents))


with open("words.arpa", "r") as read_file, open("words_correct.arpa", "w") as write_file:
  has_added_eos = False
  for line in read_file:
    if not has_added_eos and "ngram 1=" in line:
      count=line.strip().split("=")[-1]
      write_file.write(line.replace(f"{count}", f"{int(count)+1}"))
    elif not has_added_eos and "<s>" in line:
      write_file.write(line)
      write_file.write(line.replace("<s>", "</s>"))
      has_added_eos = True
    else:
      write_file.write(line)



#!/usr/bin/env python
# ----------------------------------------------------------------------------
# Copyright 2015-2016 Nervana Systems Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
# Modified to support pytorch Tensors

import Levenshtein as Lev
import torch
from six.moves import xrange


class Decoder(object):
    """
    Basic decoder class from which all other decoders inherit. Implements several
    helper functions. Subclasses should implement the decode() method.
    Arguments:
        labels (list): mapping from integers to characters.
        blank_index (int, optional): index for the blank '_' character. Defaults to 0.
        space_index (int, optional): index for the space ' ' character. Defaults to 28.
    """

    def __init__(self, labels, blank_index=0):
        self.labels = labels
        self.int_to_char = dict([(i, c) for (i, c) in enumerate(labels)])
        self.blank_index = blank_index
        space_index = len(labels)  # To prevent errors in decode, we add an out of bounds index for the space
        if ' ' in labels:
            space_index = labels.index(' ')
        self.space_index = space_index

    def wer(self, s1, s2):
        """
        Computes the Word Error Rate, defined as the edit distance between the
        two provided sentences after tokenizing to words.
        Arguments:
            s1 (string): space-separated sentence
            s2 (string): space-separated sentence
        """

        # build mapping of words to integers
        b = set(s1.split() + s2.split())
        word2char = dict(zip(b, range(len(b))))

        # map the words to a char array (Levenshtein packages only accepts
        # strings)
        w1 = [chr(word2char[w]) for w in s1.split()]
        w2 = [chr(word2char[w]) for w in s2.split()]

        return Lev.distance(''.join(w1), ''.join(w2))

    def cer(self, s1, s2):
        """
        Computes the Character Error Rate, defined as the edit distance.
        Arguments:
            s1 (string): space-separated sentence
            s2 (string): space-separated sentence
        """
        s1, s2, = s1.replace(' ', ''), s2.replace(' ', '')
        return Lev.distance(s1, s2)

    def decode(self, probs, sizes=None):
        """
        Given a matrix of character probabilities, returns the decoder's
        best guess of the transcription
        Arguments:
            probs: Tensor of character probabilities, where probs[c,t]
                            is the probability of character c at time t
            sizes(optional): Size of each sequence in the mini-batch
        Returns:
            string: sequence of the model's best guess for the transcription
        """
        raise NotImplementedError


class BeamCTCDecoder(Decoder):
    def __init__(self, labels, lm_path=None, alpha=0, beta=0, cutoff_top_n=40, cutoff_prob=1.0, beam_width=100,
                 num_processes=4, blank_index=0):
        super(BeamCTCDecoder, self).__init__(labels)
        try:
            from ctcdecode import CTCBeamDecoder
        except ImportError:
            raise ImportError("BeamCTCDecoder requires paddledecoder package.")
        labels = list(labels)  # Ensure labels are a list before passing to decoder
        self._decoder = CTCBeamDecoder(labels, lm_path, alpha, beta, cutoff_top_n, cutoff_prob, beam_width,
                                       num_processes, blank_index, log_probs_input=True)

    def convert_to_strings(self, out, seq_len):
        results = []
        for b, batch in enumerate(out):
            utterances = []
            for p, utt in enumerate(batch):
                size = seq_len[b][p]
                if size > 0:
                    transcript = ''.join(map(lambda x: self.int_to_char[x.item()], utt[0:size]))
                else:
                    transcript = ''
                utterances.append(transcript)
            results.append(utterances)
        return results

    def convert_tensor(self, offsets, sizes):
        results = []
        for b, batch in enumerate(offsets):
            utterances = []
            for p, utt in enumerate(batch):
                size = sizes[b][p]
                if sizes[b][p] > 0:
                    utterances.append(utt[0:size])
                else:
                    utterances.append(torch.tensor([], dtype=torch.int))
            results.append(utterances)
        return results

    def decode(self, probs, sizes=None):
        """
        Decodes probability output using ctcdecode package.
        Arguments:
            probs: Tensor of character probabilities, where probs[c,t]
                            is the probability of character c at time t
            sizes: Size of each sequence in the mini-batch
        Returns:
            string: sequences of the model's best guess for the transcription
        """
        probs = probs.cpu()
        out, scores, offsets, seq_lens = self._decoder.decode(probs, sizes)

        strings = self.convert_to_strings(out, seq_lens)
        offsets = self.convert_tensor(offsets, seq_lens)
        return strings, offsets


class GreedyDecoder(Decoder):
    def __init__(self, labels, blank_index=0):
        super(GreedyDecoder, self).__init__(labels, blank_index)

    def convert_to_strings(self, sequences, sizes=None, remove_repetitions=False, return_offsets=False):
        """Given a list of numeric sequences, returns the corresponding strings"""
        strings = []
        offsets = [] if return_offsets else None
        for x in xrange(len(sequences)):
            seq_len = sizes[x] if sizes is not None else len(sequences[x])
            string, string_offsets = self.process_string(sequences[x], seq_len, remove_repetitions)
            strings.append([string])  # We only return one path
            if return_offsets:
                offsets.append([string_offsets])
        if return_offsets:
            return strings, offsets
        else:
            return strings

    def process_string(self, sequence, size, remove_repetitions=False):
        string = ''
        offsets = []
        for i in range(size):
            char = self.int_to_char[sequence[i].item()]
            if char != self.int_to_char[self.blank_index]:
                # if this char is a repetition and remove_repetitions=true, then skip
                if remove_repetitions and i != 0 and char == self.int_to_char[sequence[i - 1].item()]:
                    pass
                elif char == self.labels[self.space_index]:
                    string += ' '
                    offsets.append(i)
                else:
                    string = string + char
                    offsets.append(i)
        return string, torch.tensor(offsets, dtype=torch.int)

    def decode(self, probs, sizes=None):
        """
        Returns the argmax decoding given the probability matrix. Removes
        repeated elements in the sequence, as well as blanks.
        Arguments:
            probs: Tensor of character probabilities from the network. Expected shape of batch x seq_length x output_dim
            sizes(optional): Size of each sequence in the mini-batch
        Returns:
            strings: sequences of the model's best guess for the transcription on inputs
            offsets: time step per character predicted
        """
        _, max_probs = torch.max(probs, 2)
        strings, offsets = self.convert_to_strings(max_probs.view(max_probs.size(0), max_probs.size(1)), sizes,
                                                   remove_repetitions=True, return_offsets=True)
        return strings, offsets

# !pip install folium==0.2.1
# !pip install https://github.com/kpu/kenlm/archive/master.zip pyctcdecode
# !pip install git+https://github.com/huggingface/transformers.git

from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, AutoProcessor, Wav2Vec2ProcessorWithLM, Wav2Vec2CTCTokenizer

model_name_or_path = "m3hrdadfi/wav2vec2-large-xlsr-persian-v2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(model_name_or_path, device)

processor = AutoProcessor.from_pretrained(model_name_or_path)
model = Wav2Vec2ForCTC.from_pretrained(model_name_or_path).to(device)

vocab_dict = processor.tokenizer.get_vocab()
sort_vocab = sorted((value, key) for (key,value) in vocab_dict.items())

vocab = []
for _, token in sort_vocab:
    vocab.append(token.lower())

vocab[vocab.index(processor.tokenizer.word_delimiter_token)] = ' '

import json

with open('vocab.json', 'w') as vocab_file:
    json.dump(vocab_dict, vocab_file)

tokenizer = Wav2Vec2CTCTokenizer(
    "vocab.json", 
    bos_token="<s>",
    eos_token="</s>",
    unk_token="<unk>",
    pad_token="<pad>",
    word_delimiter_token="|",
    do_lower_case=False
)

# !/content/kenlm/build/bin/build_binary -T -s trie words_correct.arpa lm4.binary
# !rm words_correct.arpa

from pyctcdecode import build_ctcdecoder

decoder = build_ctcdecoder(
    labels=vocab,
    kenlm_model_path="lm4.binary",
)

processor_with_lm = Wav2Vec2ProcessorWithLM(
    feature_extractor=processor.feature_extractor,
    tokenizer=tokenizer,
    decoder=decoder
)


import os

# os.system('git clone --recursive https://github.com/parlance/ctcdecode.git')
# os.system('cd ctcdecode && pip install .')
# !git clone --recursive https://github.com/parlance/ctcdecode.git
# !cd ctcdecode && pip install .

# beam_decoder = BeamCTCDecoder(vocab, lm_path='lm4.binary',
#                                 # alpha=0.6, beta=0.8,
#                                 alpha=1.0, beta=3.5,        # fine-tuned
#                                 cutoff_top_n=80, cutoff_prob=1.0,
#                                 beam_width=400, num_processes=16,
#                                 blank_index=vocab.index(processor.tokenizer.pad_token))

# !wget https://github.com/aliyzd95/ShEMO-Modification/raw/main/shemo.zip
# !unzip shemo.zip
# !rm shemo.zip

# !wget https://github.com/aliyzd95/ShEMO-Modification/raw/main/modified_shemo.json

# !wget https://github.com/pariajm/sharif-emotional-speech-dataset/raw/master/shemo.json
greedy_decoder = GreedyDecoder(labels=vocab)

# Example usage for decoding
def decode(logits):
    # Apply softmax to get probabilities
    probs = torch.nn.functional.softmax(logits, dim=-1)
    # Decode using Greedy Decoder
    decoded_output, _ = greedy_decoder.decode(probs)
    return decoded_output
import json

with open('shemo.json', encoding='utf-8') as os:
    original_shemo = json.loads(os.read())

with open('modified_shemo.json', encoding='utf-8') as ms:
    modified_shemo = json.loads(ms.read())

def select_dataset(MODE):

    shemo = []

    if MODE=='modified':
        for name in modified_shemo:
            sample = {}

            modified = modified_shemo[name]
            sample["sentence"] = normalizer(modified["transcript"])
            sample["path"] = modified["path"]
            
            shemo.append(sample) 

    elif MODE=='original':

        import os
        for file in os.listdir('shemo'):
            sample = {}
            file_name = file[:6]

            try:
                original = original_shemo[file_name]
                sample["sentence"] = normalizer(original["transcript"])
                sample["path"] = f'shemo/{file}'
            except:
                continue
            
            shemo.append(sample) 

    else:
        print('MODE must be one of "modified" or "original"')

    return shemo

MODE = 'modified'   ### 'original' or 'modified' 
shemo = select_dataset(MODE)
print(f'the {MODE}_ShEMO has {len(shemo)} samples.')

# !pip install git+https://github.com/huggingface/datasets.git
# !pip install torchaudio
# !pip install librosa
# !pip install jiwer

import torchaudio
import librosa
import numpy as np
import pandas as pd
from datasets import  Dataset

def speech_file_to_array_fn(batch):
    speech_array, sampling_rate = torchaudio.load(batch["path"])
    speech_array = speech_array.squeeze().numpy()
    speech_array = librosa.resample(speech_array, orig_sr=sampling_rate, target_sr=processor_with_lm.feature_extractor.sampling_rate)

    batch["speech"] = speech_array
    return batch
dataset = Dataset.from_pandas(pd.DataFrame(data=shemo))
dataset = dataset.map(speech_file_to_array_fn)

def predict(batch):

    features = processor_with_lm(
        batch["speech"], 
        sampling_rate=processor_with_lm.feature_extractor.sampling_rate, 
        return_tensors="pt", 
        padding=True
    )

    input_values = features.input_values.to(device)
    attention_mask = features.attention_mask.to(device)

    with torch.no_grad():
        logits = model(input_values, attention_mask=attention_mask).logits
    beam_decoded_output = decode(logits)
    #beam_decoded_output, beam_decoded_offsets = beam_decoder.decode(logits)

    batch["predicted"] = beam_decoded_output[0][0]
    
    return batch

result = dataset.map(predict, batched=False)

# wer = load_metric("wer")
# print(f"WER on {MODE}_ShEMO: {100 * wer.compute(predictions=result['predicted'], references=result['sentence'])}")

# from jiwer import *

# count = 0
# for data in result:

#     target = data["sentence"]
#     predict = data["predicted"]

#     w = wer(target, predict)
#     c = cer(target, predict)

#     if w>0.5 and c>0.5:
#         count += 1
#         print(data["path"])
#         print(f'sentence: {target}, predicted: {predict}')
#         print(f'wer={w}, cer={c}')
#         print('==================================================')

# print(f'There are {count} files out of a total of {len(result)} files in the dataset that has wer>0.5 and cer>0.5')

