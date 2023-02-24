from typing import List

from tqdm import tqdm
from torchvision.io import read_image

from product_classifier.config.training import ConfigTraining
from product_classifier.dataset.product import Product


def filter_out_products_to_balance_dataset(products: List[Product]) -> List[Product]:
    """Returns filtered list of products where products are removed such that the number of products belonging to each
    class is equal."""
    print('Filtering out products to balance the dataset...')

    categories = set(product.category for product in products)
    class_counts = {category: 0 for category in categories}

    for product in products:
        class_counts[product.category] += 1
    num_examples_in_smallest_class = min(class_counts.values())

    filtered_products = []
    for category in categories:
        products_in_this_category = [product for product in products if product.category == category]
        filtered_products.extend(products_in_this_category[:num_examples_in_smallest_class])

    return filtered_products


def filter_out_products_with_invalid_images(products: List[Product]) -> List[Product]:
    """Returns filtered list of products, keeping only those with RGB images of format JPG or PNG."""
    print('Filtering out non RGB, and non jpg/png images...')

    filtered_products = []
    for product in tqdm(products):
        try:
            if _is_product_image_rgb(product):
                filtered_products.append(product)
        except RuntimeError:
            print(f'Cannot load for image for {product.title} ({product.image_url}) \n because file type not jpeg/png.')
            continue
    return filtered_products


def filter_out_products_from_minority_classes(products: List[Product]) -> List[Product]:
    print('Filtering out products belonging to minority classes...')
    return _filter_products_by_class_name(products, ConfigTraining.classes_to_keep)


def _filter_products_by_class_name(products: List[Product], classes_to_keep: List[str]) -> List[Product]:
    return [product for product in tqdm(products) if product.category in classes_to_keep]


def _is_product_image_rgb(product: Product) -> bool:
    return read_image(product.image.file_path).shape[0] == 3
