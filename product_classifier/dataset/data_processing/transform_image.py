import base64
import io

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torchvision import transforms

from product_classifier.config import ConfigModel


def base64_image_to_tensor(image: str) -> Tensor:
    image = base64.b64decode(image)
    image = Image.open(io.BytesIO(image))
    image = torch.tensor(np.array(image))
    return image


def transform_image(image: Tensor) -> Tensor:
    if _are_channels_last(image):
        image = _permute_channels_to_first(image)

    new_size = (ConfigModel.image_width, ConfigModel.image_height)
    image = image.to(torch.uint8)  # transform tensor dtype for Normalize step to work
    normalised_image = image / 255.0

    transform = transforms.Compose([
        transforms.Resize(new_size),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # as defined by torch docs
    ])
    return transform(normalised_image)


def _are_channels_last(image: Tensor) -> bool:
    return image.size()[2] == 3


def _permute_channels_to_first(image: Tensor) -> Tensor:
    return image.permute(2, 0, 1)
