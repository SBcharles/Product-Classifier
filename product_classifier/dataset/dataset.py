import os
from typing import List, Tuple, Set, Dict, Optional

import requests
from pydantic import ValidationError
from torch import Tensor
from torch.utils.data import Dataset as TorchDataset
from torchvision.io import read_image
from tqdm import tqdm
from collections import OrderedDict

from product_classifier.dataset.data_processing.vectorise_title import vectorize_title
from product_classifier.dataset.exceptions import UnsupportedImageType
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

    @property
    def class_distribution(self) -> Dict[str, float]:
        """Returns a mapping from class name to number of products in that class, ordered
         by decreasing class proportion"""
        class_counts = {category: 0 for category in self.categories}
        for product in self.products:
            class_counts[product.category] += 1
        class_distribution = {category: (count / len(self.products)) for category, count in class_counts.items()}
        return OrderedDict(sorted(class_distribution.items(), key=lambda item: item[1]))

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

    def download_product_images(self, force_download: Optional[bool] = False):
        """Loops through each product in self.products and downloads image from image Url (if of allowed file type),
         and saves image in a nested "images" folder within the dataset_dir, with file name equal to the product's ID."""
        if not os.path.exists(self.images_dir):
            os.mkdir(self.images_dir)

        products_with_images = []
        print('Downloading product images...')
        for product in tqdm(self.products):
            try:
                self._download_product_image(product, force_download=force_download)
                products_with_images.append(product)
            except (requests.RequestException, UnsupportedImageType):
                continue
        self.products = products_with_images

    def _download_product_image(self, product: AmazonProduct, force_download: bool):
        if product.image_file_extension not in ('.jpg', '.jpeg', '.png'):
            raise UnsupportedImageType(f'Detected file extension: {product.image_file_extension}')

        image_file_path = os.path.join(self.images_dir, f'{product.id}{product.image_file_extension}')
        product.image.file_path = image_file_path
        if not os.path.exists(image_file_path) or force_download:
            response = requests.get(url=product.image.url, allow_redirects=True)
            response.raise_for_status()
            with open(image_file_path, 'wb') as file:
                file.write(response.content)
