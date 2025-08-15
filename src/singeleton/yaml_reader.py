import os
import yaml


class YamlReader:
    __instance = None
    __initialized = False

    def __init__(self):
        if not YamlReader.__initialized:
            try:
                with open(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../config/setup.yaml')), 'r') as f:
                    self.config = yaml.safe_load(f)
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
        return self.config.get(key)
    