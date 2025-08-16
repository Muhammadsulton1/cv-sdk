from dataclasses import dataclass

import numpy as np


@dataclass
class FrameData:
    """Структура данных кадра"""
    image: np.ndarray
    number: int
    timestamp: float


@dataclass
class S3Data:
    """Результат загрузки"""
    file_url: str
    file_id: str
    size: int