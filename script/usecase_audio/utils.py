import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # disable TF warnings

import numpy as np
import tensorflow as tf

# Set the seed value for experiment reproducibility.
seed = 1
tf.random.set_seed(seed)
np.random.seed(seed)

home_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

EPOCHS = 10
LABELS = ['no', 'yes', 'down', 'go', 'left', 'up', 'right', 'stop']
DATASET_PATH = os.path.join(home_dir, 'dataset/usecase-audio/mini_speech_commands')

#######################

def squeeze(audio, labels):
    audio = tf.squeeze(audio, axis=-1)
    return audio, labels


def get_spectrogram(waveform):
  # Convert the waveform to a spectrogram via a STFT.
  spectrogram = tf.signal.stft(
      waveform, frame_length=255, frame_step=128)
  # Obtain the magnitude of the STFT.
  spectrogram = tf.abs(spectrogram)
  # Add a `channels` dimension, so that the spectrogram can be used
  # as image-like input data with convolution layers (which expect
  # shape (`batch_size`, `height`, `width`, `channels`).
  spectrogram = spectrogram[..., tf.newaxis]
  return spectrogram


def make_spec_ds(ds):
    return ds.map(
        map_func=lambda audio,label: (get_spectrogram(audio), label),
        num_parallel_calls=tf.data.AUTOTUNE
    )


def save_model(model, save_path):
    if not os.path.exists(save_path):
        # Create the directory
        os.makedirs(save_path)

    # save entire model
    model.save(os.path.join(save_path, "model.keras"))
    print(f"The entire model saved successfully in {save_path}")


def load_model(save_path):
    # load trained model
    trained_model = tf.keras.models.load_model(os.path.join(save_path, "model.keras"))
    return trained_model


def predict_audio_file(model, audio_path):
    x = tf.io.read_file(str(audio_path))
    x, _ = tf.audio.decode_wav(x, desired_channels=1, desired_samples=16000,)
    x = tf.squeeze(x, axis=-1)
    x = get_spectrogram(x)
    x = x[tf.newaxis,...]

    prediction = model(x)
    pred = tf.argmax(prediction, axis=-1).numpy()[0]
    proba = tf.reduce_max(prediction, axis=-1, keepdims=False).numpy()[0]
    label_pred = LABELS[pred]
    return pred, label_pred, proba
