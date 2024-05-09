"""Script for Inference Pipeline in use case Text"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # disable TF warnings

import sys
import argparse

import numpy as np
import tensorflow as tf

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

from utils import (
    PAD_TYPE,
    TRUNC_TYPE,
    predict,
    load_tokenizer_model
)

# Set the seed value for experiment reproducibility.
seed = 1
tf.random.set_seed(seed)
np.random.seed(seed)

##########################

# collect args
parser = argparse.ArgumentParser(description='Inference from a text model.')
parser.add_argument("--saved-model-path", type=str,
                    help="Set the path where the trained tokenizer and model saved.",
                    required=True)
# parser.add_argument("--interactive", type=bool,
#                     help="Set if you want to use the inference in an interactive way.",
#                     default=None)
args = parser.parse_args()
saved_model_path = args.saved_model_path
interactive = True #args.interactive

##########################

trained_tokenizer, trained_model, max_length = load_tokenizer_model(saved_model_path)

print(trained_model.summary())

user_input = ""
while user_input not in ["y", "yes", "n", "no"]:
    user_input = input("Are you sure to process with the model? Please input Y/yes or N/no.\n").lower()

# stop if user chose y
if user_input == "n" or user_input == "no":
    print("Program stopped.")
    sys.exit()

stopword_set = set(stopwords.words("english")) # load stopwords remover

if interactive:
    user_input = ""
    while user_input.lower() != "end":
        user_input = input("Input your text to be predicted. Or type 'end' to stop.\n")
        if user_input.lower() == "end": break
        pred, proba = predict(
            user_input, trained_model, stopword_set,
            trained_tokenizer, PAD_TYPE, TRUNC_TYPE, max_length
        )
        print(f"Sentiment: {pred}-{'Positive' if pred == 1 else 'Negative'}")
        print("Score:", proba)
