from __future__ import annotations
from typing import Optional, List

from pydantic import BaseModel, validator, Field


class Image(BaseModel):
    url: str
    file_path: Optional[str]


class Product(BaseModel):
    id: str
    title: str
    image: Image
    category: str


class AmazonProduct(Product):
    id: str = Field(..., alias='asin')
    title: str
    image: Image
    category: str = Field(..., alias='categories')

    @classmethod
    def parse_product(cls, amazon_product_str: str) -> AmazonProduct:
        amazon_product_dict = eval(amazon_product_str)
        new_product_dict = {k: amazon_product_dict.get(k, None) for k in ('asin', 'categories', 'title')}
        new_product_dict['image'] = Image(url=amazon_product_dict['imUrl'])
        return cls(**new_product_dict)

    @validator('category', pre=True)
    def transform_category_type(cls, categories: List[List[str]]) -> str:
        """Converts nested category type to a string of the root category

        Examples
        --------
        [["books", "fiction", "horror"]]  --> "books"

        Notes
        -----
        Some products within the amazon dataset are tagged with multiple
        categories. We will take the first category for simplicity.
        """
        if (not categories) or (not categories[0]):
            raise ValueError('Field: "categories" is empty')
        return categories[0][0]

    @property
    def image_file_extension(self) -> str:
        return '.' + self.image.url.split('.').pop()
