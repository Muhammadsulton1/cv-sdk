import uuid
from abc import ABC, abstractmethod

from botocore.exceptions import ClientError

from common.utils.data_types import FrameData, S3DownloadData, S3UploadData
from common.utils.err import S3DownloadError, S3UploadError
from common.utils.logger import logger
from abc import ABC, abstractmethod

from botocore.exceptions import ClientError

from common.utils.err import S3CreateBucketError
from common.utils.logger import logger


class AbsS3FrameStorage(ABC):

    @abstractmethod
    async def upload_frame(self, data: FrameData) -> S3UploadData:
        ...

    @abstractmethod
    async def download_frame(self, data: S3UploadData) -> S3DownloadData:
        ...


class AbsS3BucketManager(ABC):

    @abstractmethod
    async def create_bucket(self, name: str) -> bool:
        ...

    @abstractmethod
    async def list_buckets(self) -> list[str]:
        ...

    @abstractmethod
    async def ensure_bucket(self, name: str) -> None:
        ...
