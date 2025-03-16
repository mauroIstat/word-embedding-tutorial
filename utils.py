# utils.py

import os
from gensim.models import KeyedVectors
import gensim.downloader as api

import numpy as np


def load_or_download_embedding(model_name, save_dir="models"):
    """
    Loads a pre-trained model from local disk if available,
    otherwise downloads it via gensim API and saves it.

    Supports models like:
    - "word2vec-google-news-300"
    - "glove-wiki-gigaword-100"
    - "fasttext-wiki-news-subwords-300"
    """
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, f"{model_name}.model")

    if os.path.exists(model_path):
        print(f"📦 Loading model from {model_path}")
        model = KeyedVectors.load(model_path)
    else:
        print(f"⬇️ Downloading model '{model_name}' from gensim API...")
        model = api.load(model_name)
        print(f"💾 Saving model to {model_path}")
        model.save(model_path)

    return model


def get_embedding(word, model):
    """
    Returns the embedding vector for a given word from the specified model.
    If the word is not in the vocabulary, returns a zero vector of the same size.

    Parameters:
    - word (str): the word to look up
    - model (gensim KeyedVectors): the embedding model

    Returns:
    - numpy.ndarray: embedding vector (or zero vector if word is OOV)
    """
    if word in model.key_to_index:
        return model[word]
    else:
        vector_size = model.vector_size
        return np.zeros(vector_size)
