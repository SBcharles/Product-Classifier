import logging
import os
import json
from datetime import datetime
import pytz

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from product_classifier.config import ConfigModel
from product_classifier.config.training import ConfigTraining
from product_classifier.dataset.data_processing.load_word_embedding import load_word_embedding
from product_classifier.dataset.dataset import AmazonDataset
from product_classifier.dataset.dataset_filters import (
    filter_out_products_with_invalid_images,
    filter_out_products_from_minority_classes,
    filter_out_products_to_balance_dataset
)
from product_classifier.dataset.train_val_test_split import train_val_test_split
from product_classifier.models.product_classifier import ProductClassifier
from product_classifier.training.evaluation import display_class_distribution_chart, run_evaluation, \
    display_confusion_matrix
from product_classifier.training.training_loop import run_training_loop


module_logger = logging.getLogger(__name__)


def write_json_file(file_path, data_dict):
    module_logger.debug("Saving data to '{}'".format(file_path))
    with open(file_path, 'w', encoding='utf-8') as json_file:
        json.dump(data_dict, json_file, ensure_ascii=False, sort_keys=True, indent=4)


def create_directory_path(prefix, training_dir_name):

    if not training_dir_name:
        raise Exception('Directory path not defined ! Check training/data directory.')

    sub_dir = os.path.join(prefix, '{}'.format(training_dir_name))

    if os.path.exists(sub_dir):
        files_in_directory = os.listdir(sub_dir)
        # ignore log files and .json files if they already exist
        files_in_directory = [file for file in files_in_directory if not(file.endswith('.log') or file.endswith('.json'))]
        if files_in_directory:
            raise Exception('Directory {} already exists and not empty. Aborting.'.format(sub_dir))
    elif not os.path.exists(sub_dir):
        os.makedirs(sub_dir)

    return sub_dir


def create_directory_path_with_timestamp(destination_dir, dir_prefix=''):

    directory_name = datetime.now(pytz.timezone('Europe/London')).strftime("%Y_%m_%d_T%H_%M_%S")
    if dir_prefix != '':
        directory_name = dir_prefix + directory_name

    sub_dir = create_directory_path(destination_dir, directory_name)

    return sub_dir


if __name__ == "__main__":
    config_training = ConfigTraining()
    config_model = ConfigModel()

    timestamped_model_dir = create_directory_path_with_timestamp(destination_dir=config_training.model_weights_dir)
    config_training.save(timestamped_model_dir)
    config_model.save(timestamped_model_dir)

    # Track model training on ClaerML
    # task = initialise_clearml_experiment(
    #     configurations=[config_training, config_model],
    #     experiment_type='Training',
    #     timestamp=os.path.basename(timestamped_model_dir)
    # )

    amazon_dataset = AmazonDataset(config_training.dataset_dir)

    amazon_dataset.load('metadata.json', max_products=config_training.max_products)
    amazon_dataset.set_word_embedding(embeddings_dict=load_word_embedding(ConfigModel.word_embedding_file_path))
    amazon_dataset.download_product_images(force_download=False)
    amazon_dataset.filter_products(filter_out_products_with_invalid_images)
    amazon_dataset.filter_products(filter_out_products_from_minority_classes)
    amazon_dataset.filter_products(filter_out_products_to_balance_dataset)
    amazon_dataset.set_class_name_to_idx()

    idx_to_class_name_file_path = os.path.join(timestamped_model_dir, "idx_to_class_name.json")
    write_json_file(idx_to_class_name_file_path, amazon_dataset.idx_to_class_name)

    train, val, test = train_val_test_split(amazon_dataset, config_training.train_val_test_proportions)

    train_loader = DataLoader(train, batch_size=config_training.batch_size, shuffle=True)
    val_loader = DataLoader(val, batch_size=config_training.batch_size, shuffle=True)
    test_loader = DataLoader(test, batch_size=config_training.batch_size, shuffle=True)

    net = ProductClassifier()
    optimiser = Adam(net.parameters())

    display_class_distribution_chart(train, 'train set')

    run_training_loop(
        net=net,
        optimiser=optimiser,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=ConfigTraining.epochs,
        model_output_dir=timestamped_model_dir
    )

    print('Loading saved model for evaluation...')
    model_path = os.path.join(timestamped_model_dir, 'model_state_dict.pkl')
    net.load_state_dict(torch.load(model_path))

    test_accuracy, test_loss = run_evaluation(
        net=net,
        dataloader=test_loader
    )

    class_names = [test.idx_to_class_name[idx] for idx in range(len(test.idx_to_class_name))]
    display_confusion_matrix(net, test_loader, class_names)

    print('Testset accuracy: ', test_accuracy)
    # task.close()
