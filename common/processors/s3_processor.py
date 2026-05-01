import asyncio
import uuid

from botocore.exceptions import ClientError

from common.abstractions.abs_s3manager import AbsS3BucketManager, AbsS3FrameStorage
from common.utils.data_types import FrameData, S3DownloadData, S3UploadData
from common.utils.err import S3DownloadError, S3UploadError, S3CreateBucketError
from common.utils.logger import logger


class S3FrameStorage(AbsS3FrameStorage):

    def __init__(self, client) -> None:
        self._client = client

    async def upload_frame(self, data: FrameData) -> S3UploadData:
        full_key = f"cam_{data.cam_id}/{data.key}"
        try:
            await self._client.put_object(
                Bucket=data.bucket,
                Key=full_key,
                Body=data.frame,
                ContentType="image/jpeg",
            )
            logger.debug(f"Кадр загружен: бакет={data.bucket!r}, ключ={full_key!r}")
            return S3UploadData(bucket=data.bucket, cam_id=data.cam_id, key=full_key)
        except ClientError as err:
            code = err.response.get("Error", {}).get("Code", "")
            logger.error(f"S3 upload error: бакет={data.bucket!r}, ключ={full_key!r}, код={code!r}")
            raise S3UploadError(f"Не удалось загрузить кадр: {code}") from err

    async def download_frame(self, data: S3UploadData) -> S3DownloadData:
        try:
            resp = await self._client.get_object(Bucket=data.bucket, Key=data.key)
            frame = await resp["Body"].read()
            return S3DownloadData(
                bucket=data.bucket,
                cam_id=data.cam_id,
                key=data.key,
                frame=frame,
            )
        except ClientError as err:
            code = err.response.get("Error", {}).get("Code", "")
            logger.error(f"S3 download error: бакет={data.bucket!r}, ключ={data.key!r}, код={code!r}")
            raise S3DownloadError(f"Не удалось скачать кадр: {code}") from err


class S3BucketManager(AbsS3BucketManager):

    def __init__(self, client) -> None:
        self._client = client
        self._cache: dict[str, bool] = {}
        self._ensure_lock = asyncio.Lock()

    async def list_buckets(self) -> list[str]:
        response = await self._client.list_buckets()
        return [b["Name"] for b in response.get("Buckets", [])]

    async def create_bucket(self, name: str) -> bool:
        try:
            await self._client.create_bucket(Bucket=name)
            self._cache[name] = True
            logger.info(f"Бакет {name!r} создан")
            return True
        except ClientError as err:
            code = err.response.get("Error", {}).get("Code", "")
            if code in ("BucketAlreadyExists", "BucketAlreadyOwnedByYou"):
                self._cache[name] = True
                logger.info(f"Бакет {name!r} уже существует ({code})")
                return True
            logger.error(f"Не удалось создать бакет {name!r}: {err!r}")
            raise S3CreateBucketError(f"Ошибка создания бакета {name!r}") from err

    async def ensure_bucket(self, name: str) -> None:
        if self._cache.get(name):
            return
        async with self._ensure_lock:
            if self._cache.get(name):
                return
            try:
                await self._client.head_bucket(Bucket=name)
                self._cache[name] = True
                logger.debug(f"Бакет {name!r} найден")
            except ClientError as err:
                meta = err.response.get("ResponseMetadata", {})
                http_code = meta.get("HTTPStatusCode")
                err_code = err.response.get("Error", {}).get("Code", "")
                if err_code in ("404", "NoSuchBucket", "NotFound") or http_code in (400, 403, 404):
                    logger.info(f"Бакет {name!r} не найден — создаю")
                    await self.create_bucket(name)
                else:
                    raise


class S3Pipeline:
    """
    Оркестратор: собирает FrameData из сырых данных ридера
    и делегирует I/O в storage.
    """
    def __init__(self, storage: AbsS3FrameStorage, bucket_manager: AbsS3BucketManager,
                 bucket: str = "reader-frames") -> None:

        self._storage = storage
        self._bucket_manager = bucket_manager
        self._bucket = bucket

    async def initialize(self) -> None:
        """Один раз при старте — гарантирует наличие бакета."""
        await self._bucket_manager.ensure_bucket(self._bucket)

    async def save_frame(self, cam_id: str, frame_bytes: bytes) -> S3UploadData:
        """
        Принимает сырые данные от ридера,
        собирает FrameData и передаёт в storage.
        """
        data = FrameData(
            bucket=self._bucket,
            cam_id=cam_id,
            key=f"{uuid.uuid4().hex}.jpg",
            frame=frame_bytes,
        )
        return await self._storage.upload_frame(data)

    async def load_frame(self, cam_id: str, key: str) -> S3DownloadData:
        """
        Принимает cam_id + ключ (из S3UploadData.key),
        возвращает S3DownloadData с байтами кадра.
        """
        data = S3UploadData(
            bucket=self._bucket,
            cam_id=cam_id,
            key=key,
        )
        return await self._storage.download_frame(data)
