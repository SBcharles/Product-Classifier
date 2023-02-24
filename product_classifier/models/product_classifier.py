import os

import torch
import torch.nn as nn
import torchvision.models as models
from torch import Tensor
from typing import Dict, Optional, Tuple

from product_classifier.config import ConfigModel
from product_classifier.config.training import ConfigTraining
from product_classifier.dataset.data_processing.load_word_embedding import load_word_embedding
from product_classifier.dataset.data_processing.transform_image import transform_image
from product_classifier.dataset.data_processing.vectorise_title import vectorize_title
from product_classifier.utils.file_handling import read_json_file


class ProductClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = ConfigModel()
        self.word_embedding: Optional[Dict[str, torch.Tensor]] = None
        self.idx_to_class_name: Optional[Dict[int, str]] = None

        self.resnet = models.resnet18(pretrained=True)
        for param in self.resnet.parameters():
            param.requires_grad = False  # freeze layers. the new (last) layer will default to requires_grad=True
        num_ftrs = self.resnet.fc.in_features  # number input features from last layer
        self.resnet.fc = nn.Linear(num_ftrs, self.config.resnet_output_units)
        self.fc2 = nn.Linear(self.config.word_embedding_vector_length + self.config.resnet_output_units, len(ConfigTraining.classes_to_keep))

    def forward(self, x):
        image = x[0]
        title = x[1]
        resnet_output = self.resnet(image)
        res_output_and_title_concatentated = torch.cat((resnet_output, title), dim=1).float()
        final_output = self.fc2(res_output_and_title_concatentated)
        return final_output

    def load(self, model_dir) -> None:
        self.config.load(model_dir)
        self.load_state_dict(torch.load(os.path.join(model_dir, 'model_state_dict.pkl')))
        self.idx_to_class_name = {int(k): v for k, v in read_json_file(os.path.join(model_dir, 'idx_to_class_name.json')).items()}
        self.word_embedding = load_word_embedding(self.config.word_embedding_file_path)

    def predict(self, image: Tensor, title: str) -> Tuple[Dict[str, float], str]:
        """ Returns all classes alongside their predicted probabilities, in addition
        to the single class name with the highest probability.

        Notes
        =====
        - Expects the original image. Do Not transform prior. """

        self.eval()
        vectorized_title = vectorize_title(title, self.word_embedding).unsqueeze(dim=0)
        image = transform_image(image).unsqueeze(dim=0)

        pred_probas = self(image, vectorized_title).squeeze().softmax(dim=0)

        predicted_class_index = torch.argmax(pred_probas).item()
        predicted_class_name = self.idx_to_class_name[predicted_class_index]
        class_name_to_probability = {self.idx_to_class_name[idx]: prob.item() for idx, prob in enumerate(pred_probas)}
        return class_name_to_probability, predicted_class_name
