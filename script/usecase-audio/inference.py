"""Script for Inference Pipeline in use case Audio"""

import sys
import argparse

import numpy as np
import tensorflow as tf

from utils import load_model, predict_audio_file

# Set the seed value for experiment reproducibility.
seed = 1
tf.random.set_seed(seed)
np.random.seed(seed)

#######################

# collect args
parser = argparse.ArgumentParser(description='Inference from an audio model.')
parser.add_argument("--saved-model-path", type=str,
                    help="Set the path where the trained model saved.",
                    required=True)
parser.add_argument("--audio-path", type=str,
                    help="Set the full path of audio to be predicted.",
                    required=True)
args = parser.parse_args()
saved_model_path = args.saved_model_path
audio_path = args.audio_path

#######################

trained_model = load_model(saved_model_path)
print(trained_model.summary())

user_input = ""
while user_input not in ["y", "yes", "n", "no"]:
    user_input = input("Are you sure to process with the model? Please input Y/yes or N/no.\n").lower()

# stop if user chose y
if user_input == "n" or user_input == "no":
    print("Program stopped.")
    sys.exit()

y_pred, pred_label, proba = predict_audio_file(trained_model, audio_path)
print(f"The predicted label: {y_pred}-{pred_label}")
print("Probability:", proba)
