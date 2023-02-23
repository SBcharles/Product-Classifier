from unittest.mock import Mock
import os
from pathlib import Path

import pytest
from torchvision.io import write_png
import torch
from requests import Response

from product_classifier.config import ConfigModel
from product_classifier.dataset.dataset import AmazonDataset
from product_classifier.dataset.product import AmazonProduct, Image


@pytest.fixture
def generate_amazon_product_strings():
    def _generate_amazon_product_strings(num_products: int):
        product_strings = []
        for idx in range(num_products):
            product_string = f'{{\'asin\': \'asin_{idx}\', \'image\': Image(url=\'imUrl_{idx}.png\', file_path=\'file_path_{idx}\'), \'categories\': [[\'category_{idx}\']], \'title\': \'title_{idx}\'}}'
            product_strings.append(product_string)
        return product_strings
    return _generate_amazon_product_strings


@pytest.fixture
def generate_amazon_dataset(tmp_path, generate_amazon_product_strings):
    def _generate_amazon_dataset(num_products: int):
        products = []
        for product_string in generate_amazon_product_strings(num_products):
            products.append(AmazonProduct(**eval(product_string)))
        dataset = AmazonDataset(str(tmp_path))
        dataset.products = products
        return dataset
    return _generate_amazon_dataset


@pytest.fixture
def generate_amazon_dataset_file(tmp_path, generate_amazon_product_strings):
    def _generate_amazon_dataset_file(number_of_products: int):
        dataset_dir = tmp_path / "amazon_dataset"
        dataset_dir.mkdir()
        file_name = "test_dataset.json"
        file_path = dataset_dir / file_name
        product_strings = generate_amazon_product_strings(number_of_products)
        file_path.write_text('\n'.join(product_strings))

        return dataset_dir, file_name
    return _generate_amazon_dataset_file


@pytest.fixture
def mock_image_response():
    mock_response = Response()
    mock_response.status_code = 200
    mock_response._content = b'a test image encoded as bytes'
    mock_response.headers = {'Content-Type': 'image/png'}
    return mock_response


@pytest.fixture
def mock_requests_get(mock_image_response, monkeypatch):
    mock_requests_get = Mock(return_value=mock_image_response)
    monkeypatch.setattr('natebbproductclassifier.dataset_creation.dataset.requests.get', mock_requests_get)
    return mock_requests_get


@pytest.fixture()
def example_product():
    product_dict = {
        'asin': 'asin_0',
        'title': 'title_0',
        'image': Image(url='', filepath='imUrl_0.png'),
        'categories': [['category_0']]
    }
    product = AmazonProduct(**product_dict)
    return product


@pytest.fixture()
def example_dataset_dir(tmp_path):
    dataset_dir = tmp_path / "amazon_dataset"
    dataset_dir.mkdir()
    return dataset_dir


@pytest.fixture()
def example_product_image_path(example_dataset_dir, example_product) -> Path:
    images_dir = example_dataset_dir / "images"
    images_dir.mkdir()
    file_name = example_product.id + '.png'
    image_path = images_dir / file_name
    return image_path


def test_len_returns_number_of_products():
    num_products = 4
    products = [
        AmazonProduct(
            asin='',
            title='',
            image=Image(url='', filepath=''),
            categories=[['Movies & TV', 'Movies', 'Horror']]
        )
        for i in range(num_products)
    ]

    amazon_dataset = AmazonDataset(dataset_dir='')
    amazon_dataset.products = products

    assert len(amazon_dataset) == num_products


def test_getitem(example_product, example_dataset_dir, example_product_image_path):
    start_image = torch.zeros(3, 400, 600, dtype=torch.uint8)
    write_png(start_image, str(example_product_image_path))
    an_amazon_dataset = AmazonDataset(str(example_dataset_dir))

    random_vector = torch.rand(ConfigModel.word_embedding_vector_length)
    embeddings_dict = {'title_0': random_vector}
    an_amazon_dataset.set_word_embedding(embeddings_dict)

    example_product.image = Image(url='', file_path=str(example_product_image_path))
    an_amazon_dataset.products = [example_product]
    an_amazon_dataset.set_category_to_idx()

    (image, title_vector), category = an_amazon_dataset[0]

    assert isinstance(image, torch.Tensor)
    assert isinstance(title_vector, torch.Tensor)
    assert torch.equal(title_vector, random_vector)
    assert category == an_amazon_dataset.category_to_idx[example_product.category]


def test_load_reads_file_and_can_parse_a_product(generate_amazon_dataset_file):
    number_of_products = 1
    dataset_dir, file_name = generate_amazon_dataset_file(number_of_products)
    amazon_dataset = AmazonDataset(str(dataset_dir))

    amazon_dataset.load(file_name=file_name, max_products=number_of_products)

    assert len(amazon_dataset.products) == number_of_products
    assert amazon_dataset.products[0].dict() == {
        'id': 'asin_0',
        'title': 'title_0',
        'image_url': 'imUrl_0.png',
        'category': 'category_0',
        'image_file': None
    }


def test_load_parses_specified_number_of_products(generate_amazon_dataset_file):
    number_of_products_in_file = 12
    max_number_of_products_to_load = 10
    dataset_dir, file_name = generate_amazon_dataset_file(number_of_products_in_file)
    amazon_dataset = AmazonDataset(str(dataset_dir))

    amazon_dataset.load(file_name=file_name, max_products=max_number_of_products_to_load)

    assert len(amazon_dataset.products) == max_number_of_products_to_load


def test_load_parses_all_products_in_file_if_max_products_greater(generate_amazon_dataset_file):
    """In case where number_of_products_in_file < max_number_of_products, we want to load all
    products in the file."""
    number_of_products_in_file = 6
    max_number_of_products_to_load = 10
    dataset_dir, file_name = generate_amazon_dataset_file(number_of_products_in_file)
    amazon_dataset = AmazonDataset(str(dataset_dir))

    amazon_dataset.load(file_name=file_name, max_products=max_number_of_products_to_load)

    assert len(amazon_dataset.products) == number_of_products_in_file


def test_download_product_images_downloads_image_and_saves(mock_requests_get, example_product, example_dataset_dir,
                                                           example_product_image_path, tmp_path):
    an_amazon_dataset = AmazonDataset(str(example_dataset_dir))
    an_amazon_dataset.products = [example_product]

    an_amazon_dataset.download_product_images()

    assert os.path.exists(example_product_image_path)
    mock_requests_get.assert_called_with(url=example_product.image_url, allow_redirects=True)


def test_download_product_images_skips_download_if_file_exists_already(mock_requests_get, example_product,
                                                                       example_dataset_dir, example_product_image_path,
                                                                       tmp_path):
    expected_file_content = 'a png image to not be overwritten'
    example_product_image_path.write_text(expected_file_content)
    an_amazon_dataset = AmazonDataset(str(example_dataset_dir))
    an_amazon_dataset.products = [example_product]

    an_amazon_dataset.download_product_images()

    assert example_product_image_path.read_text() == expected_file_content


def test_download_product_images_redownloads_image_if_forced(mock_requests_get, mock_image_response, example_product,
                                                             example_dataset_dir, example_product_image_path, tmp_path):
    original_file_content = 'original file content'
    example_product_image_path.write_text(original_file_content)
    an_amazon_dataset = AmazonDataset(str(example_dataset_dir))
    an_amazon_dataset.products = [example_product]

    an_amazon_dataset.download_product_images(force_download=True)

    assert example_product_image_path.read_text() != original_file_content
    assert example_product_image_path.read_text() == mock_image_response.text


def test_set_class_name_to_idx(generate_amazon_dataset):
    num_products = 3
    dataset = generate_amazon_dataset(num_products)

    dataset.set_category_to_idx()

    assert dataset.category_to_idx == {
        'category_0': 0,
        'category_1': 1,
        'category_2': 2
    }
