import os
from typing import List, Tuple, Set, Dict

from torch import Tensor
from torch.utils.data import Dataset as TorchDataset
from torchvision.io import read_image

from product_classifier.dataset.data_processing.vectorise_title import vectorize_title
from product_classifier.dataset.product import AmazonProduct
from product_classifier.dataset.data_processing.transform_image import transform_image


class AmazonDataset(TorchDataset):
    def __init__(self, dataset_dir: str):
        self.products: List[AmazonProduct] = []
        self.dataset_dir: str = dataset_dir
        self.images_dir: str = os.path.join(dataset_dir, 'images')
        self.category_to_idx = None
        self.embeddings_dict = None

    def __getitem__(self, idx: int) -> Tuple[Tuple[Tensor, Tensor], int]:
        product = self.products[idx]
        image: Tensor = read_image(product.image.file_path)
        image = transform_image(image)

        vectorised_title = vectorize_title(
            title=product.title,
            embeddings_dict=self.embeddings_dict)

        class_idx = self.category_to_idx[product.category]

        return (image, vectorised_title), class_idx

    def __len__(self):
        return len(self.products)

    @property
    def categories(self) -> Set[str]:
        return set([product.category for product in self.products])

    def load(self) -> None:   # todo
        """Loads the JSON file containing the amazon dataset, parses each
        product dictionary into an AmazonProduct object and appends it to
        self.products"""
        pass

    def set_word_embedding(self, embeddings_dict: Dict[str, Tensor]):
        self.embeddings_dict = embeddings_dict

    def set_category_to_idx(self):
        self.category_to_idx = {category: idx for idx, category in enumerate(sorted(self.categories))}
        self.idx_to_category = {idx: category for idx, category in enumerate(sorted(self.categories))}
