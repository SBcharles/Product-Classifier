from typing import Tuple
from copy import copy
import random

import numpy as np

from product_classifier.dataset.dataset import AmazonDataset


def train_val_test_split(dataset: AmazonDataset, train_val_test_proportions: Tuple[float, float, float]) -> Tuple[
        AmazonDataset, AmazonDataset, AmazonDataset]:
    assert np.isclose(sum(train_val_test_proportions), 1)

    train = copy(dataset)
    val = copy(dataset)
    test = copy(dataset)

    num_products = len(dataset)
    train_prop, val_prop, test_prop = train_val_test_proportions
    last_train_idx = int(num_products * train_prop)
    last_val_idx = last_train_idx + int(num_products * val_prop)

    random.shuffle(dataset.products)

    train.products = dataset.products[:last_train_idx]
    val.products = dataset.products[last_train_idx:last_val_idx]
    test.products = dataset.products[last_val_idx:]

    assert len(train) > 0
    assert len(val) > 0
    assert len(test) > 0

    return train, val, test
