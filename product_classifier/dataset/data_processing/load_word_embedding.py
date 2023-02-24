from typing import Dict

import torch


def load_word_embedding(embedding_file_path: str) -> Dict[str, torch.Tensor]:
    embeddings_dict = {}
    with open(embedding_file_path, 'r', encoding="utf-8") as f:
        for line in f:
            line = line.rstrip('\n')
            word_and_vector = line.split(sep=' ')
            word = word_and_vector[0]
            vector = torch.tensor([float(n) for n in word_and_vector[1:]])

            embeddings_dict[word] = vector
    return embeddings_dict
