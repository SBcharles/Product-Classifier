import pytest

from product_classifier.dataset.product import AmazonProduct, Image


def test_category_type_transformed_to_string():
    amazon_product = AmazonProduct(
        asin='',
        title='',
        image=Image(url='', file_path=''),
        categories=[['books', 'fiction', 'sci-fi'], ['sports', 'football', 'gloves']]
    )

    assert amazon_product.category == 'books'


def test_exception_raised_if_category_empty():
    with pytest.raises(ValueError):
        AmazonProduct(
            asin='',
            title='',
            image=Image(url='', file_path=''),
            categories=[[]]
        )


def test_image_file_extension_correct():
    product = AmazonProduct(
        asin='',
        title='',
        image=Image(url='https://test.com/test_image.jpg', file_path=''),
        categories=[['']]
    )

    assert product.image_file_extension == '.jpg'
