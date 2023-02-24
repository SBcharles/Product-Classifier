import os
import logging

import yaml

module_logger = logging.getLogger(__name__)


class ConfigBase:

    def __init__(self):
        self.config_class_name = self.__class__.__name__

    def to_dict(self):
        """
        Returns config as a dictionary
        Returns
        -------
        config_dict : dict
        """
        config_dict = dict(self.__class__.__dict__)

        # update with keys from the instance of the class
        for key, item in self.__dict__.items():
            config_dict[key] = item

        # remove any '__{}__' keys
        config_dict = {key: value for key, value in config_dict.items() if not key.startswith('__')}

        return config_dict

    def save(self, destination_dir):
        """
        Save config as yaml file
        Parameters
        ----------
        destination_dir : str

        Returns
        -------
        success : bool
        """
        aggregated_instance_config = self.to_dict()
        module_logger.info('Saving {} configuration in directory {}'.format(self.config_class_name, destination_dir))
        self.write_yaml_file(
            file_path=os.path.join(destination_dir, '{}.yaml'.format(self.config_class_name)),
            data_dict=aggregated_instance_config
        )
        return True

    def write_yaml_file(self, file_path, data_dict):
        module_logger.debug("Saving data to '{}'".format(file_path))
        with open(file_path, 'w', encoding='utf-8') as yaml_file:
            yaml.dump(data_dict, yaml_file, encoding='utf-8', sort_keys=True, indent=4)
