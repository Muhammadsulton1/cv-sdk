from abc import ABC, abstractmethod
import os


class AbstractReader(ABC):
    def __init__(self):
        self.source = os.getenv('video_source')  # будет взято из конфига
        self.skip_frames = int(os.getenv('video_skip_frames'))  # будет взято из конфига
        self.stream_index = int(os.getenv('video_index'))  # будет взято из конфига
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


class FileUploader(ABC):
    @abstractmethod
    async def upload_file(self, file_data: bytes, **kwargs) -> str:
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        pass
