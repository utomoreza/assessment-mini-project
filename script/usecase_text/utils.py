import os
import re
import pickle
import string

import numpy as np
from matplotlib import pyplot as plt

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

#########################

PAD_TYPE = "post"
TRUNC_TYPE = "post"

SEED = 1
FILE_PATH = "/content/sample_data/imdb"
TEST_SIZE = 0.4
BATCH_SIZE = 32
EPOCHS = 5
NUM_WORDS = 1000
OOV_TOKEN = "<UNK>"

embed_dim = 32  # Embedding size for each token
num_heads = 2  # Number of attention heads
ff_dim = 32  # Hidden layer size in feed forward network inside transformer

def cleanse_text(dataset, stopwords_set):
    text = dataset["text"] # extract review text from "text" row

    text = text.lower() # set to lowercase

    # remove all non-word characters (everything except numbers and letters)
    text = re.sub(r"[^\w\s]", '', text)

    # remove digits
    text = re.sub(r"\d", '', text)

    # remove HTML tags
    text = re.sub(r"<.*?>", '', text)

    # remove URLs
    text = re.sub(r"http\S+", '', text)
    text = re.sub(r"www\S+", '', text)

    # remove punctuations
    text = text.translate(str.maketrans('', '', string.punctuation))

    # remove unnecessary whitespaces
    text = re.sub(f" {2,}", '', text)

    # remove stopwords
    text_split = tuple(text.split())
    list_token = (token for token in text_split if token not in stopwords_set)
    text = " ".join(list_token)

    return {"clean_text": text}


def tokenize(dataset, tokenizer):
    tokens = tokenizer.texts_to_sequences([dataset["clean_text"]])
    return {"tokens": tokens[0]}


def padding(dataset, pad_type, trunc_type, maxlen):
    # Pad the sequences
    padded_tokens = pad_sequences(
        [dataset["tokens"]], padding=pad_type,
        truncating=trunc_type, maxlen=maxlen
    )
    return {"ids": padded_tokens}


def find_optimum_maxlen(list_tokens, percentile=0.75):
    maxlen = np.quantile([len(ele) for ele in list_tokens], percentile)
    return int(maxlen)


def plot_graphs(history, metric):
    plt.plot(history.history[metric])
    plt.plot(history.history['val_'+ metric], '')
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, 'val_'+ metric])


def predict_text(text, model, stopword_set, tokenizer, pad_type, trunc_type, maxlen):
    # preprocess
    dict_sample = {"text": text}
    dict_sample = cleanse_text(dict_sample, stopword_set)
    dict_sample = tokenize(dict_sample, tokenizer)
    dict_sample = padding(dict_sample, pad_type, trunc_type, maxlen)
    ids = dict_sample["ids"]
    ids = tf.convert_to_tensor(ids)

    # predict
    y_pred = 1 if model.predict(ids)[0] > 0.5 else 0

    return y_pred


def LstmModel(vocab_size, lstm_dim=64):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input((None,)),
            tf.keras.layers.Embedding(
                input_dim=vocab_size,
                output_dim=64,
                # Use masking to handle the variable sequence lengths
                mask_zero=True),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_dim)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(1, activation="sigmoid")
        ]
    )
    return model


@tf.keras.saving.register_keras_serializable()
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.ffn = tf.keras.Sequential(
            [tf.keras.layers.Dense(ff_dim, activation="relu"),
             tf.keras.layers.Dense(embed_dim),]
        )
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, inputs):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output)


@tf.keras.saving.register_keras_serializable()
class TokenAndPositionEmbedding(tf.keras.layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super().__init__()
        self.token_emb = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = tf.keras.layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions


def TransformerModel(max_length, vocab_size, embed_dim, num_heads, ff_dim):
    inputs = tf.keras.layers.Input(shape=(max_length,))
    embedding_layer = TokenAndPositionEmbedding(max_length, vocab_size, embed_dim)
    x = embedding_layer(inputs)
    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    x = transformer_block(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(20, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model


def predict(text, model, stopword_set, tokenizer, pad_type, trunc_type, maxlen):
    # preprocess
    dict_sample = {"text": text}
    dict_sample = cleanse_text(dict_sample, stopword_set)
    dict_sample = tokenize(dict_sample, tokenizer)
    dict_sample = padding(dict_sample, pad_type, trunc_type, maxlen)
    ids = dict_sample["ids"]
    ids = tf.convert_to_tensor(ids)

    # predict
    y_pred = 1 if model.predict(ids)[0] > 0.5 else 0

    return y_pred, model.predict(ids)[0]


def save_tokenizer_model(tokenizer, model, max_length, save_path):
    if not os.path.exists(save_path):
        # Create the directory
        os.makedirs(save_path)

    # save tokenizer
    with open(os.path.join(save_path, "tokenizer.pickle"), 'wb') as handle:
        pickle.dump((tokenizer, max_length), handle, protocol=pickle.HIGHEST_PROTOCOL)
    # save entire model
    model.save(os.path.join(save_path, "model.keras"))
    # model.export(os.path.join(save_path, "model"))
    print(f"Tokenizer and the entire model saved successfully in {save_path}")


def load_tokenizer_model(save_path):
    # load trained tokenizer
    with open(os.path.join(save_path, "tokenizer.pickle"), 'rb') as handle:
        trained_tokenizer, max_length = pickle.load(handle)
    # load trained model
    trained_model = tf.keras.models.load_model(os.path.join(save_path, "model.keras"))

    return trained_tokenizer, trained_model, max_length
