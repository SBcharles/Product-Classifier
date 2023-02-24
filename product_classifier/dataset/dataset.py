import os
from typing import List, Tuple, Set, Dict

from pydantic import ValidationError
from torch import Tensor
from torch.utils.data import Dataset as TorchDataset
from torchvision.io import read_image

from product_classifier.dataset.data_processing.vectorise_title import vectorize_title
from product_classifier.dataset.product import AmazonProduct, Image
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

    def load(self, file_name: str, max_products: int) -> None:
        """Loads the amazon dataset JSON file which contains the amazon dataset, parses each
        product dictionary into an AmazonProduct object and appends it to self.products"""
        dataset_file_path = os.path.join(self.dataset_dir, file_name)

        with open(dataset_file_path) as file:
            print('Loading dataset...')
            incomplete_product_count = 0

            for _ in range(max_products):
                try:
                    product = AmazonProduct.parse_product(next(file))
                    self.products.append(product)
                except ValidationError:
                    incomplete_product_count += 1
                    continue
                except StopIteration:
                    break

            print(f'Number of incomplete products: {incomplete_product_count}',
                  f'({incomplete_product_count/(incomplete_product_count + len(self.products)):.3%})')

    def set_word_embedding(self, embeddings_dict: Dict[str, Tensor]):
        self.embeddings_dict = embeddings_dict

    def set_category_to_idx(self):
        self.category_to_idx = {category: idx for idx, category in enumerate(sorted(self.categories))}
        self.idx_to_category = {idx: category for idx, category in enumerate(sorted(self.categories))}
