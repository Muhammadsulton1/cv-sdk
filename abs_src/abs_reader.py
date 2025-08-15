from abc import ABC, abstractmethod
import os


class AbstractReader(ABC):
    def __init__(self):
        self.source = os.getenv('video_source')  # будет взято из конфига
        if not self.source:
            raise ValueError("Источник видео не установлен")
        self.skip_frames = int(os.getenv('video_skip_frames', 0))  # будет взято из конфига
        self.stream_index = int(os.getenv('video_index', 0))  # будет взято из конфига
        self.is_open = False

    @abstractmethod
    def open(self):
        """Открыть источник"""
        pass

    @abstractmethod
    def get_frame(self):
        """Получить следующий кадр"""
        pass

    @abstractmethod
    def close(self):
        """Закрыть ресурсы"""
        pass

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
