"""Script for Training Pipeline in use case Image"""

import argparse

import numpy as np
from sklearn.model_selection import train_test_split

import tensorflow as tf
import keras
from keras.datasets import cifar10

from utils import (
    normalize,
    save_model,
    ResNet34,
    code_to_label,
    TEST_SIZE,
    BATCH_SIZE,
    EPOCHS
)

# Set the seed value for experiment reproducibility.
seed = 1
tf.random.set_seed(seed)
np.random.seed(seed)

#######################

# collect args
parser = argparse.ArgumentParser(description='Training a model for Image Classification.')
parser.add_argument("--saved-model-path", type=str,
                    help="If you want to save the tokenizer and entire model, input the path to save them.",
                    default=None, required=False)
args = parser.parse_args()
saved_model_path = args.saved_model_path

#######################

# Adding TF Cifar10 Data ..
print("Collecting data ...")
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

X_val, X_test, y_val, y_test = train_test_split(
    X_test, y_test, test_size=TEST_SIZE, shuffle=True
)

num_classes = len(code_to_label)

# normalization
X_train = normalize(X_train)
X_val = normalize(X_val)
X_test = normalize(X_test)

# on-hot encoded to 10 classes
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_val = tf.keras.utils.to_categorical(y_val, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

# setup model
print("Preparing for modelling ...")
model = ResNet34()
model.compile(
    optimizer="adam", loss='categorical_crossentropy',
    metrics=["categorical_accuracy",
             tf.keras.metrics.Precision(),
             tf.keras.metrics.Recall(),
             tf.keras.metrics.F1Score()]
)

# set callback
callback = keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=3, restore_best_weights=True
)

# start training
print("Training model...")
history = model.fit(
    X_train, y_train, batch_size=BATCH_SIZE,
    epochs=EPOCHS, shuffle=True,
    validation_data=(X_val, y_val),
    callbacks=[callback]
)

# evaluate the trained model
print("Evaluating model ...")
test_loss, test_acc, test_prec, test_rec, test_f1score = model.evaluate(X_test, y_test)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_acc)
print("Test Precision:", test_prec)
print("Test Recall:", test_rec)
for idx, score in enumerate(test_f1score):
    print(f"Test F1-Score for label {idx}-{code_to_label[idx]}:", score)

# save the trained model
if saved_model_path:
    print("Saving the trained model ...")
    save_model(model, saved_model_path)
