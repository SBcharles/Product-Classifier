import os
from typing import List, Tuple, Set

from torch import Tensor
from torch.utils.data import Dataset as TorchDataset
from torchvision.io import read_image

from product_classifier.dataset.product import AmazonProduct
from product_classifier.dataset.data_processing.transform_image import transform_image


class AmazonDataset(TorchDataset):
    def __init__(self, dataset_dir: str):
        self.products: List[AmazonProduct] = []
        self.dataset_dir: str = dataset_dir
        self.images_dir: str = os.path.join(dataset_dir, 'images')
        self.category_to_idx = None

    def __getitem__(self, idx: int) -> Tuple[Tuple[Tensor, Tensor], int]:
        product = self.products[idx]
        image: Tensor = read_image(product.image.file_path)
        image = transform_image(image)

        vectorised_title = self.vectorise_title(product.title)

        class_idx = self.category_to_idx(product.category)

        return (image, vectorised_title), class_idx

    def __len__(self):
        return len(self.products)

    @property
    def categories(self) -> Set[str]:
        return set([product.category for product in self.products])

    def vectorise_title(self, title: str) -> Tensor:   # todo
        pass

    def load(self) -> None:   # todo
        """Loads the JSON file containing the amazon dataset, parses each
        product dictionary into an AmazonProduct object and appends it to
        self.products"""
        pass
