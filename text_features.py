seed_value = 42
import os

os.environ['PYTHONHASHSEED'] = str(seed_value)
# os.environ["TF_DETERMINISTIC_OPS"] = str(seed_value)
import random

random.seed(seed_value)
import numpy as np

np.random.seed(seed_value)

import hazm
from keras_preprocessing.sequence import pad_sequences
from gensim.models import fasttext, KeyedVectors


emo_codes = {"A": 0, "W": 1, "H": 2, "S": 3, "N": 4, "F": 5}
emo_labels = ["anger", "surprise", "happiness", "sadness", "neutral", "fear"]
embedding_dim = 100


def get_emotion_label(file_name):
    emo_code = file_name[3]
    return emo_codes[emo_code]


def get_emotion_name(file_name):
    emo_code = file_name[5]
    return emo_labels[emo_codes[emo_code]]


# def embeddings_indexes():
#     embeddings_index = dict()
#     f = open(
#         'files/embeddings/PersianWordak/twitter_wikipedia_hamshahri_irblog/simple/twitt_wiki_ham_blog.fa.text.100.vec',
#         encoding='utf-8')
#     for line in f:
#         values = line.split()
#         word = values[0]
#         coefs = np.asarray(values[1:], dtype='float32')
#         embeddings_index[word] = coefs
#     f.close()
#     return embeddings_index


def get_embedding_matrix():
    # sentences = load_sentences()

    with open('files/w2c_cleanedV2.txt', encoding='utf-8') as f:
        sentences = f.read().splitlines()

    # sentences, y, _ = read_csv_data()

    hazmt = hazm.WordTokenizer(join_verb_parts=False)
    t_sents = hazmt.tokenize_sents(sentences)
    words = []
    total_words = 0
    for sent in t_sents:
        for word in sent:
            total_words += 1
            if word not in words:
                words.append(word)
    vocab_size = len(words) + 1
    word_index = dict(zip(sorted(words), list(range(1, vocab_size))))
    encoded_docs = []
    for sent in t_sents:
        s = []
        for word in sent:
            k = word_index[word]
            s.append(k)
        encoded_docs.append(s)
    max_length = len(max(encoded_docs, key=len))
    padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')

    embedding_model = fasttext.load_facebook_model(
        "files/embeddings/farsi-dedup-skipgram.bin")
    print('---------- model loaded! ----------')

    embedding_model.min_count = 1
    embedding_model.build_vocab(t_sents, update=True)
    print('---------- vocab built! ----------')
    embedding_model.train(t_sents, total_examples=len(t_sents), epochs=20)
    print('---------- model retrained! ----------')

    oov_words = []
    n_words = 0
    for word, i in word_index.items():
        n_words += 1
        if word not in embedding_model.wv.index_to_key:
            oov_words.append(word)
    print('Total number of tokens in dataset:', total_words)
    print('Number of unique words in dataset:', n_words)
    print('Number of OOV words:', len(oov_words))
    print(oov_words)

    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embedding_model.wv.get_vector(word)
        embedding_matrix[i] = embedding_vector

    np.save('files/embeddings/new_embedding_matrix(normalized).npy', embedding_matrix)
    np.save('files/embeddings/new_padded_docs(normalized).npy', padded_docs)


if __name__ == '__main__':
    get_embedding_matrix()
