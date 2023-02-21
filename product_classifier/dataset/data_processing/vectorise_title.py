import re
from typing import List, Dict

import matplotlib.colors as colors
import torch
from torch import Tensor
import nltk
from nltk.corpus import stopwords

from product_classifier.config import ConfigModel


nltk.download('stopwords')


def vectorize_title(title: str, embeddings_dict: Dict[str, Tensor]) -> Tensor:
    tokens = _tokenize(title, excluded=ConfigModel.excluded_tokens)
    tokens = _remove_stop_words(tokens)
    tokens = _remove_bad_words(tokens, ConfigModel.bad_words)
    try:
        title_vector = _average_of_word_vectors(tokens, embeddings_dict)
    except ZeroDivisionError:
        title_vector = torch.zeros(ConfigModel.word_embedding_vector_length)
    return title_vector


def _tokenize(text: str, excluded: str) -> List[str]:
    text = text.lower()
    return re.split("[^" + excluded + r"\w]+", text)


def _remove_stop_words(text_tokens: List[str]) -> List[str]:
    stop_words = stopwords.words() + list(colors.cnames.keys())
    return [word for word in text_tokens if word not in stop_words]


def _remove_bad_words(text_tokens: List[str], bad_words: List[str]) -> List[str]:
    return [word for word in text_tokens if word not in bad_words]


def _average_of_word_vectors(words: List[str], embeddings_dict: Dict[str, Tensor]) -> Tensor:
    word_vectors = []
    for word in words:
        try:
            word_vectors.append(embeddings_dict[word])
        except KeyError:
            continue
    return sum(word_vectors) / len(word_vectors)
