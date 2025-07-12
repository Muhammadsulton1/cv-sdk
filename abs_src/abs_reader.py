from abc import ABC, abstractmethod


class AbstractReader(ABC):
    def __init__(self, source, skip_frames=0, stream_index=0):
        self.source = source
        self.skip_frames = skip_frames
        self.stream_index = stream_index
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


class AbstractBucketManager(ABC):
    @abstractmethod
    async def upload_image(self):
        raise NotImplementedError

    @abstractmethod
    async def get_image_url(self):
        raise NotImplementedError
