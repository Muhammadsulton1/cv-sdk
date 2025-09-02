import os
import yaml

from typing import Dict, Any
from utils.logger import logger


class YamlReader:
    __instance = None
    __initialized = False

    def __init__(self):
        if not YamlReader.__initialized:
            try:
                with open(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../config/setup.yaml')),
                          'r') as f:
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

    @property
    def reader_settings(self) -> Dict[Any, Dict[str, Any]]:
        """Возвращает размеры батчей для всех reader'ов в конфигурации."""
        result = {}
        readers_found = False

        for name, item in self.config.items():
            if item.get('type') == 'reader':
                readers_found = True
                settings = item.get('settings')
                if settings is None:
                    raise ValueError(f"Reader '{name}' не имеет настроек (settings)")

                batch_size = settings.get('batch_size')
                skip_frames = settings.get('skip_frames')
                video_source = settings.get('source')
                if batch_size is None:
                    raise ValueError(f"Reader '{name}' не имеет batch_size в настройках")

                result[name] = {'batch_size': batch_size, 'skip_frames': skip_frames, 'source': video_source}

        if not readers_found:
            raise RuntimeError("В конфигурации не найден ни один reader")

        return result

    def _validate_channels(self):
        """Проверяет корректность конфигурации каналов."""
        for name, item in self.config.items():
            if 'out_channel' not in item:
                error_msg = f"В конфиге setup.yaml '{name}' отсутствует обязательный ключ 'out_channel'"
                logger.error(error_msg)
                raise KeyError(error_msg)

            if item.get('type') != 'reader':
                if 'in_channel' not in item:
                    error_msg = f"В конфиге setup.yaml '{name}' отсутствует обязательный ключ 'in_channel'"
                    logger.error(error_msg)
                    raise KeyError(error_msg)

            if item.get('type') == 'reader':
                if 'settings' not in item:
                    error_msg = f"Reader '{name}' не имеет раздела 'settings'"
                    logger.error(error_msg)
                    raise KeyError(error_msg)

                settings = item['settings']
                if 'batch_size' not in settings:
                    error_msg = f"Reader '{name}' не имеет 'batch_size' в настройках"
                    logger.error(error_msg)
                    raise KeyError(error_msg)


if __name__ == '__main__':
    config = YamlReader()
    print(config.batch_reader_size)