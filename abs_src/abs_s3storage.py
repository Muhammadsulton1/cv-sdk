from abc import ABC, abstractmethod


class AbstractS3Storage(ABC):
    @abstractmethod
    async def upload_object(self, file_data: bytes) -> str:
        pass

    @abstractmethod
    async def download_object(self, object_url: str):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        pass
