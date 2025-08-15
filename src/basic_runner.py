from abc import ABC, abstractmethod
from typing import List

from utils.logger import logger


class AbstractRunner(ABC):
    @abstractmethod
    def run_process(self, processors: List) -> None:
        pass


class BasicRunner(AbstractRunner):
    def run_process(self, processors: List) -> None:
        if not isinstance(processors, list):
            raise ValueError('Параметр processors должен быть списком')

        for module in processors:
            try:
                module.process()
            except Exception as err:
                logger.error(f"Ошибка в процессе {module}: {err}")
                if hasattr(module, 'close') and callable(module.close):
                    try:
                        module.close()
                    except Exception as close_err:
                        logger.error(f"Ошибка при закрытии {module}: {close_err}")
