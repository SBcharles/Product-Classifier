import logging
import os
import yaml

module_logger = logging.getLogger(__name__)


class ConfigModel:
    image_width = 256
    image_height = 256

    resnet_output_units = 50

    word_embedding_file_path = ''
    word_embedding_vector_length = 50
    bad_words = []  # additional words to be removed in title embedding process e.g. ['small', 'medium', 'large']
    excluded_tokens = '-/.%'

    def load(self, destination_dir):
        """
        Load config from yaml file within destination_dir
        Parameters
        ----------
        destination_dir : str

        Returns
        -------
        success : bool
        """

        destination_path = os.path.join(destination_dir, f'{self.__class__.__name__}.yaml')
        config_dict = self.read_yaml_file(destination_path)

        # update instance configuration
        for key, value in config_dict.items():
            setattr(self, key, value)

        module_logger.info(f'Loaded {self.__class__.__name__} configuration from directory {destination_dir}.')
        module_logger.info(self.__str__())

        return True

    def read_yaml_file(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as yaml_file:
            data_dict = yaml.load(yaml_file, Loader=yaml.FullLoader)

        return data_dict
