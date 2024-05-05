"""
"""
import os
import pickle
import argparse

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer

from datasets import load_dataset

from utils import (
    PAD_TYPE,
    TRUNC_TYPE,
    cleanse_text,
    tokenize,
    padding,
    find_optimum_maxlen,
    LstmModel,
    TransformerModel
)

##########################

# collect args
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument("--model-type", type=str,
                    help="Set a model to train, whether LSTM or Transformer",
                    required=True)
parser.add_argument("--dataset-path", type=str,
                    help="Set path for the dataset used for training. Default: './dataset/usecase-text'.",
                    default='./dataset/usecase-text', required=False)
parser.add_argument("--saved-model-path", type=str,
                    help="If you want to save the tokenizer and entire model, input the path to save them.",
                    default=None, required=False)
args = parser.parse_args()
model_type = args.model_type
dataset_path = args.dataset_path
saved_model_path = args.saved_model_path

##########################

train_data = load_dataset("csv", data_files=os.path.join(dataset_path, "train_set.csv"))
valid_data = load_dataset("csv", data_files=os.path.join(dataset_path, "valid_set.csv"))
test_data = load_dataset("csv", data_files=os.path.join(dataset_path, "train_set.csv"))

stopword_set = set(stopwords.words("english"))

# cleansing text
print("Cleansing texts ...")
train_data = train_data.map(
    cleanse_text, fn_kwargs={"stopwords_set": stopword_set}
)
valid_data = valid_data.map(
    cleanse_text, fn_kwargs={"stopwords_set": stopword_set}
)
test_data = test_data.map(
    cleanse_text, fn_kwargs={"stopwords_set": stopword_set}
)

# Tokenize training data
print("Tokenizing texts ...")
tokenizer = Tokenizer(num_words=NUM_WORDS, oov_token=OOV_TOKEN)
tokenizer.fit_on_texts(train_data["clean_text"])

# Get training data word index
word_index = tokenizer.word_index
vocab_size = len(word_index)

## TOKENIZE DATA ###
train_data = train_data.map(
    tokenize, fn_kwargs={"tokenizer": tokenizer}
)
valid_data = valid_data.map(
    tokenize, fn_kwargs={"tokenizer": tokenizer}
)
test_data = test_data.map(
    tokenize, fn_kwargs={"tokenizer": tokenizer}
)

# find max length from Q3 of the length of each text
max_length = find_optimum_maxlen(train_data["tokens"], percentile=0.75)

## PAD TOKENS ###
train_data = train_data.map(
    padding, fn_kwargs={
        "pad_type": PAD_TYPE,
        "trunc_type": TRUNC_TYPE,
        "maxlen": max_length
    }
)
valid_data = valid_data.map(
    padding, fn_kwargs={
        "pad_type": PAD_TYPE,
        "trunc_type": TRUNC_TYPE,
        "maxlen": max_length
    }
)
test_data = test_data.map(
    padding, fn_kwargs={
        "pad_type": PAD_TYPE,
        "trunc_type": TRUNC_TYPE,
        "maxlen": max_length
    }
)

### CREATE DATALOADER ###
# convert to torch tensor
train_data = train_data.with_format(
    type="tf", columns=["ids", "label"]
)
valid_data = valid_data.with_format(
    type="tf", columns=["ids", "label"]
)
test_data = test_data.with_format(
    type="tf", columns=["ids", "label"]
)

## Squeeze tensor's shape
X_train = tf.squeeze(train_data["ids"], axis=[1])
y_train = train_data["label"]
X_valid = tf.squeeze(valid_data["ids"], axis=[1])
y_valid = valid_data["label"]

## SETUP MODEL ARCHITECTURE
print("Preparing for modelling ...")
if model_type.lower() == "lstm":
    model = LstmModel(vocab_size)
elif model_type.lower() == "transformer":
    model = TransformerModel(max_length, vocab_size, embed_dim, num_heads, ff_dim)

model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(1e-4),
    metrics=['accuracy']
)

# set callback for early stopping
callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=3, restore_best_weights=True
)

# start training
print(f"Training model {model_type} ...")
history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    shuffle=True,
    validation_data=(X_valid, y_valid),
    callbacks=[callback]
)

# evaluate the trained model
test_loss, test_acc = model.evaluate(X_train, y_train)
print('Test Loss:', test_loss)
print('Test Accuracy:', test_acc)

# save the trained model
if saved_model_path:
    print("Saving the trained tokenizer and model ...")
    # save tokenizer
    with open(f"{saved_model_path}/tokenizer.pickle", 'wb') as handle:
        pickle.dump((tokenizer, max_length), handle, protocol=pickle.HIGHEST_PROTOCOL)
    # save entire model
    model.save(f"{saved_model_path}/model.keras")
    print(f"Tokenizer and the entire model saved successfully in {saved_model_path}")
