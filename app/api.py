import os
import io
import sys
import base64
import warnings
warnings.filterwarnings("ignore")

import numpy as np

from PIL import Image

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

from flask import abort, Flask, jsonify, request

from keras.preprocessing import image
import tensorflow as tf

#####################

work_dir = os.path.dirname(os.path.abspath(__file__))
home_dir = os.path.dirname(work_dir)

sys.path.append(home_dir)
import script.usecase_text.utils as utils_text
import script.usecase_image.utils as utils_image
import script.usecase_audio.utils as utils_audio

#####################

SAVED_MODEL_PATH_LSTM = os.path.join(home_dir, "model/usecase_text/lstm")
SAVED_MODEL_PATH_TRANSFORMER = os.path.join(home_dir, "model/usecase_text/transformer")
SAVED_MODEL_PATH_RESNET = os.path.join(home_dir, "model/usecase_image")
SAVED_MODEL_PATH_AUDIO = os.path.join(home_dir, "model/usecase_audio")

#####################

# load pretrained tokenizer and models

# for use-case text
print("Loading tokenizer and models for use case text ...")
tokenizer, model_lstm, max_length = utils_text.load_tokenizer_model(
    SAVED_MODEL_PATH_LSTM
)
_, model_transformer, _ = utils_text.load_tokenizer_model(
    SAVED_MODEL_PATH_TRANSFORMER
)
stopword_set = set(stopwords.words("english")) # load stopwords remover

# for use-case image
print("Loading model for use case image ...")
model_resnet = utils_image.load_model(SAVED_MODEL_PATH_RESNET)

# for use-case audio
print("Loading model for use case audio ...")
model_audio = utils_audio.load_model(SAVED_MODEL_PATH_AUDIO)

#####################

def predict_image_bytes(model, bytes_string):
    # preprocess
    # get target_size from the dimensions of 1st layer
    target_size = model.layers[0].output_shape[1:]
    # read image from bytes
    img_bytes = base64.b64decode(bytes_string)
    # img_bytes = bytes(bytes_string, "utf-8")
    img = Image \
        .open(io.BytesIO(img_bytes)) \
        .convert("RGB") \
        .resize(target_size, Image.NEAREST)

    example = image.img_to_array(img)
    example = np.expand_dims(example, axis=0)
    example = utils_image.normalize(example)

    img.close() # close oponed image

    # predict
    pred = model.predict(example)
    y_pred = np.argmax(pred)
    proba = np.max(pred)
    pred_label = utils_image.code_to_label[y_pred]

    return y_pred, pred_label, proba


def recognize_audio_bytes(model, bytes_string):
    # read image from bytes
    audio_bytes = base64.b64decode(bytes_string)
    x, _ = tf.audio.decode_wav(
        audio_bytes, desired_channels=1,
        desired_samples=16000
    )

    x = tf.squeeze(x, axis=-1)
    x = utils_audio.get_spectrogram(x)
    x = x[tf.newaxis,...]

    # recognize
    prediction = model(x)
    pred = tf.argmax(prediction, axis=-1).numpy()[0]
    proba = tf.reduce_max(prediction, axis=-1, keepdims=False).numpy()[0]
    label_pred = utils_audio.LABELS[pred]

    return pred, label_pred, proba

#####################

app = Flask(__name__)

@app.route("/text", methods=["POST"])
def predict_text():
    if request.method != "POST":
        abort(400)

    if not request.json:
        abort(400)

    req_json = request.json
    model_type = req_json["model_type"].lower()
    user_input = req_json["user_input"]

    # check model_type must be either lstm or transformer
    assert model_type == "lstm" or model_type == "transformer"

    print("Model used:", model_type)
    if model_type == "lstm":
        pred, proba = utils_text.predict(
            user_input, model_lstm, stopword_set,
            tokenizer, utils_text.PAD_TYPE,
            utils_text.TRUNC_TYPE, max_length
        )
    elif model_type == "transformer":
        pred, proba = utils_text.predict(
            user_input, model_transformer, stopword_set,
            tokenizer, utils_text.PAD_TYPE,
            utils_text.TRUNC_TYPE, max_length
        )

    return jsonify(
        {
            "pred": pred,
            "proba": str(proba[0])
        }
    )


@app.route("/image", methods=["POST"])
def predict_image():
    if request.method != "POST":
        abort(400)

    if not request.json:
        abort(400)

    bytes_string = request.json["img_bytes"]
    y_pred, pred_label, proba = predict_image_bytes(
        model_resnet, bytes_string
    )

    return jsonify(
        {
            "pred": y_pred,
            "pred_label": pred_label,
            "proba": str(proba[0])
        }
    )


@app.route("/audio", methods=["POST"])
def recognize_audio():
    if request.method != "POST":
        abort(400)

    if not request.json:
        abort(400)

    bytes_string = request.json["audio_bytes"]
    y_pred, pred_label, proba = recognize_audio_bytes(
        model_audio, bytes_string
    )

    return jsonify(
        {
            "pred": str(y_pred),
            "pred_label": pred_label,
            "proba": str(proba)
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7070, debug=False)
