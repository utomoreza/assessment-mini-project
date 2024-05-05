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
    list_token = (token for token in text_split if token not in stopword_set)
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

    return y_pred