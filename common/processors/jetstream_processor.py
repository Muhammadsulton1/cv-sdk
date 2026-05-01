from nats.js.api import ConsumerConfig, DeliverPolicy, AckPolicy

from common.abstractions.abs_jetstream import AbstractStreamManager
from common.processors.s3_processor import S3BucketManager, S3FrameStorage
from common.utils.data_types import FrameData, S3UploadData, S3DownloadData
from common.utils.decorators import async_timing
from common.utils.logger import logger


class StreamPublisher(AbstractStreamManager):
    def __init__(self) -> None:
        super().__init__()
        self._bucket_mgr: S3BucketManager | None = None
        self._storage: S3FrameStorage | None = None

    async def __aenter__(self):
        await super().__aenter__()
        if not self.topic_out_channel:
            raise ValueError(f"out_channel не задан для сервиса '{self._service_name}'")
        self._bucket_mgr = S3BucketManager(self._s3_raw_client)
        self._storage = S3FrameStorage(self._s3_raw_client)
        await self._bucket_mgr.ensure_bucket(self.topic_out_channel)
        logger.info("FramePublisher: S3-бакет %r готов к публикации кадров", self.topic_out_channel)
        return self

    def _get_storage(self) -> tuple[S3BucketManager, S3FrameStorage]:
        if self._bucket_mgr is None:
            self._bucket_mgr = S3BucketManager(self._s3_raw_client)
        if self._storage is None:
            self._storage = S3FrameStorage(self._s3_raw_client)
        return self._bucket_mgr, self._storage

    @async_timing("FramePublisher.s3_upload_frame")
    async def _upload_frame(self, storage: S3FrameStorage, frame_data: FrameData) -> S3UploadData:
        return await storage.upload_frame(frame_data)

    @async_timing("FramePublisher.jetstream_publish")
    async def _publish_key(self, subject: str, key_bytes: bytes) -> None:
        await self.js.publish(subject, key_bytes)

    async def process(self, frame_data: FrameData) -> S3UploadData:
        """
        Бакет гарантируется один раз в __aenter__.
        Загружает кадр в S3, публикует S3-ключ в subject {out_channel}.{cam_id}.
        """
        _, storage = self._get_storage()

        upload_result = await self._upload_frame(storage, frame_data)

        subject = self._publish_subject(upload_result.cam_id)
        await self._publish_key(subject, upload_result.key.encode())

        logger.debug("Published: subject=%s key=%s", subject, upload_result.key)
        return upload_result


class StreamConsumer(AbstractStreamManager):
    """
    Инференс-сторона: читает ключи из JetStream, скачивает кадры из S3.
    in_channel обязателен в конфиге.
    Переопределите _handle() для своей бизнес-логики.
    """

    async def process(self) -> None:
        """Бесконечно читает сообщения из in_channel и вызывает _handle()."""
        durable = self.__class__.__name__
        subject_filter = self._subscribe_subject()

        sub = await self.js.subscribe(
            subject_filter,
            durable=durable,
            config=ConsumerConfig(
                durable_name=durable,
                deliver_policy=DeliverPolicy.ALL,
                ack_policy=AckPolicy.EXPLICIT,
                filter_subject=subject_filter,
            ),
            manual_ack=True,
        )

        logger.info("StreamConsumer подписан: subject=%s durable=%s", subject_filter, durable)

        async for msg in sub.messages:
            key = msg.data.decode()
            # cam_id извлекаем из subject: {channel}.{cam_id}
            cam_id = msg.subject.split(".")[-1]

            try:
                await self._handle(cam_id=cam_id, key=key)
                await msg.ack()
            except Exception as e:
                logger.error(
                    "Ошибка обработки key=%s cam_id=%s: %s",
                    key, cam_id, e, exc_info=True,
                )
                await msg.nak()

    async def _download_frame(self, cam_id: str, key: str) -> S3DownloadData:
        """Скачивает кадр из S3 по ключу из JetStream-сообщения."""
        storage = S3FrameStorage(self._s3_raw_client)
        download_data = S3UploadData(
            bucket=self.topic_in_channel,  # bucket == канал ридера
            cam_id=cam_id,
            key=key,
        )
        return await storage.download_frame(download_data)

    async def _handle(self, cam_id: str, key: str) -> None:
        """
        Переопределите в подклассе.
        download_data = await self._download_frame(cam_id, key)
        frame_ndarray = cv2.imdecode(np.frombuffer(download_data.frame, np.uint8), cv2.IMREAD_COLOR)
        """
        raise NotImplementedError
