import os
import time
import uuid

import asyncio
import contextlib
import cv2
import numpy as np

from pathlib import Path
from typing import Any
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

from common.abstractions.abs_jetstream import AbstractProcessor
from common.processors.jetstream_processor import StreamPublisher
from common.utils.data_types import FrameData
from common.utils.logger import logger


class ReaderProcessor(AbstractProcessor):
    """
    Оркестратор ридера:
    _producer → читает кадры → Queue[FrameData]
    _consumer → берёт из очереди → FramePublisher (S3 + JetStream)
    """
    STREAM_SCHEMES = {"rtsp", "rtsps", "rtmp", "rtmps", "http", "https", "udp", "tcp"}
    FILE_EXTENSIONS = {
        ".avi", ".m4v", ".mjpeg", ".mjpg", ".mkv", ".mov",
        ".mp4", ".mpeg", ".mpg", ".ts", ".webm", ".wmv",
    }

    def __init__(self) -> None:
        super().__init__()

        self.reader = None  # создаём после валидации конфига
        settings = self._cfg.get("settings", {})

        self.bucket: str = self._cfg.get("out_channel", "conveyor-frames")
        self.source: str = settings.get("source", "")
        self.source_type: str = settings.get("source_type", "auto").lower()
        self.reconnect_streams: bool = self._as_bool(settings.get("reconnect_streams", True))
        try:
            self.reconnect_attempts = max(
                0, int(settings.get("stream_reconnect_attempts", 0))
            )
        except (TypeError, ValueError):
            self.reconnect_attempts = 0
        try:
            self.reconnect_delay = max(0.1, float(settings.get("stream_reconnect_delay", 1.0)))
        except (TypeError, ValueError):
            self.reconnect_delay = 1.0
        try:
            self.reconnect_max_delay = max(self.reconnect_delay,
                                           float(settings.get("stream_reconnect_max_delay", 30.0)))
        except (TypeError, ValueError):
            self.reconnect_max_delay = 30.0
        self._is_stream_source = self._detect_stream_source(
            self.source, self.source_type
        )
        self.buffer_size: int = int(settings.get("reader_buffer", 100))
        self.batch_size: int = int(settings.get("batch_size", 1))
        self.publisher_workers: int = max(1, int(settings.get("publisher_workers", 4)))
        self.target_fps: float = max(0.0, float(settings.get("target_fps", 0.0)))
        self.queue_policy: str = settings.get("queue_policy", "block").lower()
        self.drop_log_interval: float = float(settings.get("drop_log_interval", 5.0))
        self._last_drop_log: float = 0.0

        raw_threads = settings.get("jpeg_encode_threads", 0)
        try:
            self._jpeg_encode_threads = max(0, int(raw_threads))
        except (TypeError, ValueError):
            self._jpeg_encode_threads = 0
        cpus = os.cpu_count() or 4
        self._jpeg_encode_worker_count = 0
        if self._jpeg_encode_threads > 0:
            self._jpeg_encode_worker_count = max(
                1, min(self._jpeg_encode_threads, cpus),
            )
            self._jpeg_executor = ThreadPoolExecutor(
                max_workers=self._jpeg_encode_worker_count,
                thread_name_prefix="jpeg_enc")
        else:
            self._jpeg_executor = None

        raw_quality = settings.get("jpeg_quality")
        if raw_quality is None:
            self._jpeg_encode_params = tuple()
            self._jpeg_quality_log = "default"
        else:
            try:
                img_qual = max(1, min(100, int(raw_quality)))
            except (TypeError, ValueError):
                img_qual = 85
            self._jpeg_encode_params = (int(cv2.IMWRITE_JPEG_QUALITY), img_qual)
            self._jpeg_quality_log = str(img_qual)

        self._queue: asyncio.Queue[FrameData] = asyncio.Queue(maxsize=self.buffer_size)
        self._stop = asyncio.Event()

    @staticmethod
    def _as_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() not in {"0", "false", "no", "off"}
        return bool(value)

    @classmethod
    def _detect_stream_source(cls, source: str, source_type: str) -> bool:
        if source_type in {"stream", "rtsp", "camera", "network"}:
            return True
        if source_type in {"file", "video"}:
            return False

        if len(source) >= 3 and source[1:3] in {":\\", ":/"}:
            return False

        parsed = urlparse(source)
        if parsed.scheme.lower() in cls.STREAM_SCHEMES:
            return True

        suffix = Path(source).suffix.lower()
        if suffix in cls.FILE_EXTENSIONS:
            return False

        return False

    @staticmethod
    def _jpeg_encode_frame(frame_bgr: np.ndarray, encode_params: tuple[int, Any]) -> tuple[bool, np.ndarray]:
        if encode_params:
            return cv2.imencode(".jpg", frame_bgr, list(encode_params))
        return cv2.imencode(".jpg", frame_bgr)

    @staticmethod
    def _build_reader():
        from common.processors.reader_processor import FrameReaderFabric
        return FrameReaderFabric().get_reader()

    async def _open_reader(self):
        failed_attempts = 0
        while not self._stop.is_set():
            try:
                return self._build_reader()
            except Exception as exc:
                if not self.reconnect_streams or not self._is_stream_source:
                    raise

                if 0 < self.reconnect_attempts <= failed_attempts:
                    raise

                failed_attempts += 1
                delay = min(self.reconnect_delay * (2 ** (failed_attempts - 1)), self.reconnect_max_delay)
                logger.error("Stream open failed, retrying in %.1f sec: source=%s attempt=%d error=%s",
                             delay, self.source, failed_attempts, exc, exc_info=True)
                await asyncio.sleep(delay)

        return None

    async def _reconnect_reader(self, failed_attempts: int) -> tuple[bool, int]:
        if not self.reconnect_streams or not self._is_stream_source:
            return False, failed_attempts

        if 0 < self.reconnect_attempts <= failed_attempts:
            logger.error("Stream reconnect attempts exhausted: source=%s attempts=%d", self.source, failed_attempts)
            return False, failed_attempts

        failed_attempts += 1
        delay = min(self.reconnect_delay * (2 ** (failed_attempts - 1)), self.reconnect_max_delay)
        logger.warning("Stream interrupted, reconnecting in %.1f sec: source=%s attempt=%d",
                       delay, self.source, failed_attempts)
        await asyncio.sleep(delay)

        if self.reader:
            with contextlib.suppress(Exception):
                self.reader.release()

        try:
            self.reader = self._build_reader()
        except Exception as exc:
            logger.error(
                "Stream reconnect failed: source=%s attempt=%d error=%s",
                self.source, failed_attempts, exc, exc_info=True,
            )
            return True, failed_attempts

        logger.info("Stream reconnected: source=%s cam_id=%s", self.source, self.reader.source_id)
        return True, 0

    async def _producer(self) -> None:
        """Читает кадры из cv-ридера, кодирует в JPEG, кладёт в очередь."""
        self.reader = await self._open_reader()
        if self.reader is None:
            return

        frame_interval = 1.0 / self.target_fps if self.target_fps > 0 else 0.0
        next_frame_at = time.monotonic()
        reconnect_failures = 0

        logger.info("Reader started: cam_id=%s buffer=%d workers=%d queue_policy=%s "
                    "target_fps=%s jpeg_threads=%s jpeg_quality=%s source_type=%s",
                    self.reader.source_id,
                    self.buffer_size,
                    self.publisher_workers,
                    self.queue_policy,
                    self.target_fps or "unlimited",
                    self._jpeg_encode_worker_count,
                    self._jpeg_quality_log,
                    "stream" if self._is_stream_source else "file")

        while not self._stop.is_set():
            try:
                frame_ndarray = self.reader.get_frame()
            except Exception as exc:
                if not self._is_stream_source:
                    logger.error(
                        "Ошибка чтения файла source=%s: %s",
                        self.source, exc, exc_info=True,
                    )
                    raise

                logger.warning(
                    "Ошибка чтения stream source=%s: %s",
                    self.source, exc, exc_info=True,
                )
                frame_ndarray = None

            if frame_ndarray is None:
                reconnected, reconnect_failures = await self._reconnect_reader(reconnect_failures)
                if reconnected:
                    next_frame_at = time.monotonic()
                    continue

                logger.info("Источник завершился, останавливаем producer")
                self._stop.set()
                break

            reconnect_failures = 0
            raw = self.reader.to_raw_frame(frame_ndarray)

            encode_params = self._jpeg_encode_params
            loop = asyncio.get_running_loop()
            if self._jpeg_executor is not None:
                ok, buf = await loop.run_in_executor(
                    self._jpeg_executor,
                    ReaderProcessor._jpeg_encode_frame,
                    raw.frame,
                    encode_params,
                )
            else:
                ok, buf = ReaderProcessor._jpeg_encode_frame(
                    raw.frame, encode_params)

            if not ok:
                logger.warning("Failed to encode frame cam_id=%s", raw.cam_id)
                await self._throttle(frame_interval, next_frame_at)
                next_frame_at = time.monotonic()
                continue

            data = FrameData(
                bucket=self.bucket,
                cam_id=raw.cam_id,
                key=f"{uuid.uuid4().hex}.jpg",
                frame=buf.tobytes())

            await self._enqueue(data)
            next_frame_at = await self._throttle(frame_interval, next_frame_at)

    async def _enqueue(self, data: FrameData) -> None:
        if self.queue_policy == "block":
            await self._queue.put(data)
            return

        if self.queue_policy == "drop_oldest" and self._queue.full():
            try:
                self._queue.get_nowait()
                self._queue.task_done()
                self._log_drop(data.cam_id, "oldest")
            except asyncio.QueueEmpty:
                pass

        try:
            self._queue.put_nowait(data)
        except asyncio.QueueFull:
            self._log_drop(data.cam_id, "newest")

    @staticmethod
    async def _throttle(frame_interval: float, next_frame_at: float) -> float:
        if frame_interval <= 0:
            await asyncio.sleep(0)
            return next_frame_at

        next_frame_at += frame_interval
        delay = next_frame_at - time.monotonic()
        if delay > 0:
            await asyncio.sleep(delay)
            return next_frame_at

        await asyncio.sleep(0)
        return time.monotonic()

    def _log_drop(self, cam_id: str, dropped: str) -> None:
        now = time.monotonic()
        if now - self._last_drop_log < self.drop_log_interval:
            return
        self._last_drop_log = now
        logger.warning("Queue full (%d), dropping %s frame cam_id=%s qsize=%d",
                       self.buffer_size, dropped, cam_id, self._queue.qsize())

    async def _consumer(self, publisher: StreamPublisher, worker_id: int) -> None:
        while not self._stop.is_set() or not self._queue.empty():
            try:
                data: FrameData = await asyncio.wait_for(
                    self._queue.get(), timeout=0.5
                )
            except asyncio.TimeoutError:
                continue

            try:
                result = await publisher.process(data)
                logger.debug("[FRAME READER CONSUMER]: Кадр опубликован в stream: worker=%d cam_id=%s key=%s",
                             worker_id, result.cam_id, result.key)
            except Exception as e:
                logger.error("[ERROR][FRAME READER CONSUMER] ошибка публикации в stream - worker=%d cam_id=%s: %s",
                             worker_id, data.cam_id, e, exc_info=True)
            finally:
                self._queue.task_done()

    async def process(self, *args, **kwargs) -> None:
        producer_task = None
        consumer_tasks: list[asyncio.Task] = []

        try:
            async with StreamPublisher() as publisher:
                producer_task = asyncio.create_task(self._producer(), name="reader-producer")
                consumer_tasks = [
                    asyncio.create_task(
                        self._consumer(publisher, worker_id),
                        name=f"reader-consumer-{worker_id}",
                    )
                    for worker_id in range(self.publisher_workers)
                ]

                await producer_task
                await self._queue.join()
        except asyncio.CancelledError:
            logger.info("ReaderProcessor: tasks cancelled")
        except Exception as e:
            logger.error("ReaderProcessor error: %s", e, exc_info=True)
            raise
        finally:
            self._stop.set()
            if producer_task is not None:
                producer_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await producer_task
            for task in consumer_tasks:
                task.cancel()
            if consumer_tasks:
                await asyncio.gather(*consumer_tasks, return_exceptions=True)
            if self.reader:
                self.reader.release()
            if self._jpeg_executor is not None:
                self._jpeg_executor.shutdown(wait=True, cancel_futures=False)
            logger.info("ReaderProcessor stopped")
