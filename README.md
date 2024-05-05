# Assessment Mini Project

## Use Cases

### Text Classification (Sentiment Analysis)

### Image Classification

### Audio Recognition

## Repository Structure

- ./
-- asset -> consisting of images of models
-- dataset -> consisting of datasets used for training models
-- model -> consisting of saved, trained models
-- notebook -> consisting of notebooks used to explore the use cases
-- sample -> consisting of sample data to test the models
-- script -> consisting of scripts to run training and inference pipeline
-- LICENSE
-- README.md
-- requirements.txt -> listing libraries used in this mini project

**Notes:**
- Not all models for each use case is saved in this repo since some models are too big to be included here


## Usages

- Clone this repo: `git clone https://github.com/utomoreza/assessment-mini-project.git`
- `cd assessment-mini-project`
- Make sure you already have your python environment, e.g. venv or conda, activated
- Install the dependencies, `pip install -r requirements.txt`

### For notebooks

You can explore the step-by-step process for each use case in the directory `./assessment-mini-project/notebook`.

### For training & inference pipelines

- Go to directory script `cd assessment-mini-project/script/<your-desired-usecase>`
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
