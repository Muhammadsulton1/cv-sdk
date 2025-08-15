import os
import yaml

from utils.logger import logger


class YamlReader:
    __instance = None
    __initialized = False

    def __init__(self):
        if not YamlReader.__initialized:
            try:
                with open(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../config/setup.yaml')), 'r') as f:
                    self.config = yaml.safe_load(f)
                    self._validate_channels()
            except FileNotFoundError:
                raise
            except yaml.YAMLError as err:
                raise err
            YamlReader.__initialized = True

    def __new__(cls):
        if not cls.__instance:
            cls.__instance = super().__new__(cls)
        return cls.__instance

    def get(self, key):
        try:
            return self.config.get(key)
        except KeyError:
            logger.error(f'Нету ключа в setup.yaml: {key}')
            raise KeyError(key)

    def _validate_channels(self):
        for name, item in self.config.items():
            if item.get('type') != 'reader':
                if 'in_channel' not in item:
                    error_msg = f"В конфиге setup.yaml '{name}' отсутствует обязательный ключ 'in_channel'"
                    logger.error(error_msg)
                    raise KeyError(error_msg)
            if 'out_channel' not in item:
                error_msg = f"В конфиге setup.yaml '{name}' отсутствует обязательный ключ 'out_channel'"
                logger.error(error_msg)
                raise KeyError(error_msg)
