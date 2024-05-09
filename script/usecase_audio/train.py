"""Script for Training Pipeline in use case Audio"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # disable TF warnings

import pathlib
import argparse

import numpy as np
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import models

from utils import (
    EPOCHS,
    DATASET_PATH,
    squeeze,
    make_spec_ds,
    save_model
)

# Set the seed value for experiment reproducibility.
seed = 1
tf.random.set_seed(seed)
np.random.seed(seed)

#######################

# collect args
parser = argparse.ArgumentParser(description='Training a model for Audio Recognition.')
parser.add_argument("--saved-model-path", type=str,
                    help="If you want to save the entire model, input the path to save them.",
                    default=None, required=False)
args = parser.parse_args()
saved_model_path = args.saved_model_path

#######################


# download data if not existing
print("Collecting data if unavailable ...")
data_dir = pathlib.Path(DATASET_PATH)
if not data_dir.exists():
    # Create the directory
    os.makedirs(data_dir)
    tf.keras.utils.get_file(
        'mini_speech_commands.zip',
        origin="http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
        extract=True,
        cache_dir=data_dir, cache_subdir='data'
    )

# load data
print("Loading data ...")
train_ds, val_ds = tf.keras.utils.audio_dataset_from_directory(
    directory=data_dir/"data/mini_speech_commands",
    batch_size=64,
    validation_split=0.2,
    seed=0,
    output_sequence_length=16000,
    subset='both'
)

label_names = np.array(train_ds.class_names)

print("Preprocessing data ...")
# squeeze train & valid sets
train_ds = train_ds.map(squeeze, tf.data.AUTOTUNE)
val_ds = val_ds.map(squeeze, tf.data.AUTOTUNE)

# split validation & test sets
test_ds = val_ds.shard(num_shards=2, index=0)
val_ds = val_ds.shard(num_shards=2, index=1)

# create spectrogram datasets from the audio datasets:
train_spectrogram_ds = make_spec_ds(train_ds)
val_spectrogram_ds = make_spec_ds(val_ds)
test_spectrogram_ds = make_spec_ds(test_ds)

for example_spectrograms, example_spect_labels in train_spectrogram_ds.take(1):
    break

# add Dataset.cache and Dataset.prefetch operations to reduce read latency while training the model:
train_spectrogram_ds = train_spectrogram_ds.cache().shuffle(10000).prefetch(tf.data.AUTOTUNE)
val_spectrogram_ds = val_spectrogram_ds.cache().prefetch(tf.data.AUTOTUNE)
test_spectrogram_ds = test_spectrogram_ds.cache().prefetch(tf.data.AUTOTUNE)

# define model
print("Preparing for modelling ...")
input_shape = example_spectrograms.shape[1:]
num_labels = len(label_names)

# Instantiate the `tf.keras.layers.Normalization` layer.
norm_layer = layers.Normalization()
# Fit the state of the layer to the spectrograms
# with `Normalization.adapt`.
norm_layer.adapt(data=train_spectrogram_ds.map(map_func=lambda spec, label: spec))

model = models.Sequential([
    layers.Input(shape=input_shape),
    # Downsample the input.
    layers.Resizing(32, 32),
    # Normalize.
    norm_layer,
    layers.Conv2D(32, 3, activation='relu'),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.25),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_labels, activation='softmax'),
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
)

# start training
print("Training the model ...")
history = model.fit(
    train_spectrogram_ds,
    validation_data=val_spectrogram_ds,
    epochs=EPOCHS,
    callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=2),
)

print("Evaluating the model ...")
test_loss, test_acc = model.evaluate(test_spectrogram_ds)
print('Test Loss:', test_loss)
print('Test Accuracy:', test_acc)

# save the trained model
if saved_model_path:
    print("Saving the trained model ...")
    save_model(model, saved_model_path)
