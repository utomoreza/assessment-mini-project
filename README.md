# Assessment Mini Project

## Use Cases

### Text Classification (Sentiment Analysis)

- Two model architectures, i.e. LSTM and Vanilla Transformer
- Trained on IMDB dataset
- The LSTM architecture is shown [here](./asset/model_lstm.png), whereas that of Vanilla Transformer is [here](./asset/model_transformer.png)

### Image Classification

- Using ResNet-34 model architecture
- Trained on CIFAR-10 dataset
- The ResNet-34 architecture is shown [here](./asset/model_resnet.png)

### Audio Recognition

- Using CNN 2D
- Trained on Speed Commands dataset
- The model architecture is shown [here](./asset/model_audio.png)

## Repository Structure

- `asset/` -> consisting of images of models
- `dataset/` -> consisting of datasets used for training models
- `model/` -> consisting of saved, trained models
- `notebook/` -> consisting of notebooks used to explore the use cases
- `sample/` -> consisting of sample data to test the models
- `script/` -> consisting of scripts to run training and inference pipeline
- `LICENSE`
- `README.md`
- `requirements.txt` -> listing libraries used in this mini project

**Notes:** Not all models for each use case is saved in this repo since some models are too big to be included here.

## Usages

- Clone this repo: `git clone https://github.com/utomoreza/assessment-mini-project.git`
- `cd assessment-mini-project`
- Make sure you already have your python environment, e.g. venv or conda, activated
- Install the dependencies, `pip install -r requirements.txt`

### For notebooks

You can explore the step-by-step process for each use case in the directory [`./assessment-mini-project/notebook`](./assessment-mini-project/notebook).

### For training & inference pipelines

- Go to the directory [`script`](./assessment-mini-project/script) `cd assessment-mini-project/script/<your-desired-usecase>`
- Run the Python script `python train.py --<arg>==<param>` for training pipeline
- Run the Python script `python inference.py --<arg>==<param>` for inference pipeline

The list of args for each script:
- Training pipeline
    - use case text
        - `--model-type=MODEL_TYPE`
        Set a model to train, whether LSTM or Transformer. This arg is required. The valid value is either `lstm` or `transformer`.
        - `--dataset-path=DATASET_PATH`
        Set path for the dataset used for training. Default: `./dataset/usecase-text`. This arg is optional.
        - `--saved-model-path=SAVED_MODEL_PATH`
        If you want to save the tokenizer and entire model, input the path to save them. If you don't set this, the trained model won't be saved. This arg is optional.
    - use case image
        - `--saved-model-path=SAVED_MODEL_PATH`
        If you want to save the tokenizer and entire model, input the path to save them. If you don't set this, the trained model won't be saved. This arg is optional.
    - use case audio
        
- Inference pipeline
    - use case text
        - `--saved-model-path=SAVED_MODEL_PATH`
        Set the path where the trained tokenizer and model saved. This arg is required.
    - use case image
        - `--saved-model-path=SAVED_MODEL_PATH`
        Set the path where the trained model saved. This arg is required.
        - `--image-path=IMAGE_PATH`
        Set the full path of image to be predicted. This arg is required.
    - use case audio
        - `--saved-model-path=SAVED_MODEL_PATH`
        Set the path where the trained model saved. This arg is required.
        - `--audio-path=AUDIO_PATH`
        Set the full path of audio to be predicted. This arg is required.

## References

### For text use case
- https://www.kdnuggets.com/2020/03/tensorflow-keras-tokenization-text-data-prep.html
- https://github.com/kpot/keras-transformer/blob/master/keras_transformer/position.py
- https://www.kaggle.com/code/samuelnordmann/transformer-in-tensorflow-from-scratch
- https://pyimagesearch.com/2022/11/07/a-deep-dive-into-transformers-with-tensorflow-and-keras-part-3/
- https://keras.io/examples/nlp/text_classification_with_transformer/
- https://medium.com/@max_garber/simple-keras-transformer-model-74724a83bb83
- https://keras.io/guides/keras_nlp/transformer_pretraining/
- https://www.tensorflow.org/tutorials/keras/text_classification
- https://keras.io/examples/nlp/text_classification_from_scratch/
- https://www.tensorflow.org/text/tutorials/classify_text_with_bert
- https://colab.research.google.com/github/tensorflow/text/blob/master/docs/tutorials/text_classification_rnn.ipynb#scrollTo=9TnJztDZGw-n

### For image use case
- https://www.kaggle.com/c/cifar-10/data
- https://www.analyticsvidhya.com/blog/2021/08/how-to-code-your-resnet-from-scratch-in-tensorflow/
- https://www.kaggle.com/code/songrise/implementing-resnet-18-using-keras/notebook
- https://github.com/songrise/CNN_Keras/blob/main/src/ResNet-18.py
- https://github.com/alinarw/ResNet/blob/master/ResNet.ipynb
- https://github.com/pythonlessons/Keras-ResNet-tutorial/blob/master/Keras-ResNet-tutorial.ipynb
- https://www.tensorflow.org/datasets/keras_example
- https://www.kaggle.com/code/namansood/resnet50-training-on-mnist-transfer-learning
- https://github.com/shoji9x9/Fashion-MNIST-By-ResNet/blob/master/Fashion-MNIST-by-ResNet-50.ipynb
- https://www.kaggle.com/code/donatastamosauskas/using-resnet-for-mnist
- https://github.com/Natsu6767/ResNet-Tensorflow/blob/master/ResNet%20Train.ipynb

### For audio use case
- https://www.kaggle.com/c/tensorflow-speech-recognition-challenge/data
- https://fortes-arthur.medium.com/hands-on-speech-recognition-engine-with-keras-and-python-c60488ac53cd
- https://github.com/arthurfortes/speech2text_keras/blob/master/Speech2Text%20Approach.ipynb
- https://www.kaggle.com/code/sunyuanxi/speech-recognition-keras
- https://lyronfoster.com/2023/03/24/speech-recognition-with-tensorflow-and-keras-libraries-in-python-yes-like-siri-and-alexa/
- https://github.com/tensorflow/docs/blob/master/site/en/tutorials/audio/simple_audio.ipynb
- https://www.tensorflow.org/datasets/catalog/speech_commands
- https://www.tensorflow.org/tutorials/audio/simple_audio
- https://keras.io/examples/audio/ctc_asr/
