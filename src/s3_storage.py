import io
import os
import uuid
import asyncio
import aiohttp
import cv2
import numpy as np

from PIL import Image
from utils.decorators import measure_latency_async, retry
from utils.err import S3UploadError, S3DownloadError
from utils.logger import logger


class SeaweedFSManager:
    def __init__(self):
        self.master_url = os.getenv("master_url")
        self.bucket_name = os.getenv("bucket_name")
        self.volume_url = os.getenv("volume_url")
        self.ttl = os.getenv("ttl_bucket", "5m")
        self._session = None

    async def __aenter__(self):
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=10),
            connector=aiohttp.TCPConnector(limit_per_host=100)
        )
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self._session:
            await self._session.close()

    @retry(retries=3)
    async def upload_object(self, frame: np.ndarray) -> str:
        try:
            return await self._upload(frame)
        except (aiohttp.ClientError, asyncio.TimeoutError) as err:
            logger.error(f"Ошибка загрузки данных в S3")
            raise S3UploadError from err

    async def _upload(self, frame: np.ndarray) -> str:
        try:
            assign_url = f"http://{self.master_url}:9333/dir/assign?ttl={self.ttl}&collection={self.bucket_name}"
            async with self._session.get(assign_url, raise_for_status=True) as resp:
                assign_data = await resp.json()
                fid = assign_data["fid"]

            upload_url = f"http://{self.volume_url}:8888/{self.bucket_name}/{fid}?ttl={self.ttl}"

            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])

            form_data = aiohttp.FormData()
            form_data.add_field(
                'file',
                buf.tobytes(),
                filename=f"{uuid.uuid4()}.jpg",
                content_type='image/jpeg'
            )

            async with self._session.post(
                    upload_url,
                    data=form_data,
                    raise_for_status=True
            ) as resp:
                pass

            return f"{self.volume_url}/{self.bucket_name}/{fid}"

        except (aiohttp.ClientError, asyncio.TimeoutError, KeyError) as err:
            logger.error(f"Сетевая ошибка при загрузки данных в S3: {type(err).__name__} - {str(err)}")
            raise err
        except S3UploadError as err:
            logger.error(f"Ошибка загрузки данных в S3: {type(err).__name__} - {str(err)}")
            raise err

    @retry(retries=3)
    async def download_object(self, object_url: str):
        """Оптимизированное скачивание файла с кешированием соединения"""
        try:
            async with self._session.get(object_url, raise_for_status=True) as response:
                image_data = await response.read()
                return Image.open(io.BytesIO(image_data))

        except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as err:
            logger.error(f"Сетевая ошибка при скачивании данных из S3: {type(err).__name__} - {str(err)}")
            raise err
        except S3DownloadError as err:
            logger.error(f"Ошибка скачивания данных из S3: {type(err).__name__} - {str(err)}")
            raise err
