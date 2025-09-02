import uuid
from abc import ABC, abstractmethod
import os
from typing import List

import numpy as np

from src.singeleton.yaml_reader import YamlReader


class AbstractReader(ABC):
    def __init__(self):
        self.setup_settings = YamlReader().reader_settings
        self.reader_service = os.getenv('SERVICE_NAME')
        self.skip_frames = int(self.setup_settings.get(self.reader_service).get('skip_frames'))
        self.source = self.setup_settings.get(self.reader_service).get('source')
        self.batch_size = int(self.setup_settings.get(self.reader_service).get('batch_size'))
        if not self.source:
            raise ValueError("Источник видео не установлен")
        self.stream_index = int(os.getenv('video_index', 0))
        self.is_open = False

    @abstractmethod
    def open(self) -> None:
        """Открыть источник"""
        pass

    @abstractmethod
    def get_frame(self) -> List[np.ndarray]:
        """Получить следующий кадр"""
        pass

    @abstractmethod
    def close(self) -> None:
        """Закрыть ресурсы"""
        pass

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


class AbstractFrameReader(ABC):
    def __init__(self):
        self.frame_height = None
        self.frame_width = None
        self.framerate = None
        self.frames_processed = 0
        self.stream = None
        self.container = None

        self.setup_settings = YamlReader().reader_settings
        self.reader_service = os.getenv('SERVICE_NAME')
        service_config = self.setup_settings.get(self.reader_service, {})

        self.skip_frames = int(service_config.get('skip_frames', 0))
        self.frames_skip_batch = int(service_config.get('frames_skip_batch', 0))
        self.need_skip_batch = self.frames_skip_batch > 0

        self.source = service_config.get('source')
        self.batch_size = int(service_config.get('batch_size', 1))
        if not self.source:
            raise ValueError("Источник видео не установлен")

        self.stream_name = f"cam_{str(uuid.uuid4())[:3]}"

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @abstractmethod
    def open(self) -> None:
        pass

    @abstractmethod
    def close(self) -> None:
        pass

    @abstractmethod
    def get_frame(self):
        pass

    def __iter__(self):
        return self

    def __next__(self):
        frames = self.get_frame()
        if frames is None:
            raise StopIteration
        return frames

    @property
    def info(self):
        return {
            "source": self.source,
            "source_name": self.stream_name,
            "width": self.frame_width,
            "height": self.frame_height,
            "framerate": self.framerate,
            "frames_skipped": self.skip_frames,
            "frames_processed": self.frames_processed,
            "batch_size": self.batch_size
        }
