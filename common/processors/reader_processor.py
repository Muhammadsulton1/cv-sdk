import os

import cv2
from common.abstractions.abs_reader import AbstractFrameReader, AbstractFrameReaderFabric
import time
import av
import numpy as np

from av.error import FFmpegError
from common.utils.err import StreamError
from common.singeleton.config_manager import ConfigManager


class PyAVReader(AbstractFrameReader):
    def __init__(self) -> None:
        super().__init__()

        try:
            options = {"rtsp_flags": "prefer_tcp"} if self.source.startswith("rtsp://") else {}
            self.container = av.open(self.source, options=options)
        except FFmpegError as e:
            raise StreamError(f"Не удалось открыть контейнер: {e}") from e

        try:
            self.stream = self.container.streams.video[0]
        except IndexError:
            self.container.close()
            raise StreamError("Не удалось найти видео поток")

        self._frame_iter = self.container.decode(video=0)

        rate = self.stream.average_rate
        self._declared_fps = float(rate) if rate and float(rate) > 0.0 else 0.0

        self.width = self.stream.width or None
        self.height = self.stream.height or None
        self._released = False
        self.start_time = time.time()

    def release(self) -> None:
        if self._released:
            return
        self.container.close()
        self._released = True

    def get_fps(self) -> float:
        self.fps = self._declared_fps if self._declared_fps > 0.0 else self._measured_fps()
        return self.fps

    def _next_decoded(self):
        try:
            return next(self._frame_iter)
        except StopIteration:
            return None

    def get_frame(self) -> np.ndarray | None:
        if self._released:
            return None

        av_frame = None
        for _ in range(self.skip_frames):
            av_frame = self._next_decoded()
            if av_frame is None:
                self.release()
                return None

        frame = av_frame.to_ndarray(format="bgr24")
        self.height, self.width = frame.shape[:2]
        self.frames_yielded += 1
        self.stream_frames_consumed += self.skip_frames
        return frame

    def __iter__(self):
        return self

    def __next__(self) -> np.ndarray:
        frame = self.get_frame()
        if frame is None:
            raise StopIteration
        return frame


class OpenCVReader(AbstractFrameReader):
    def __init__(self) -> None:
        super().__init__()

        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            self.cap.release()
            raise StreamError("Не удалось открыть источник видео")

        raw_fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        self._declared_fps = raw_fps if raw_fps > 0.0 else 0.0
        self._released = False
        self.start_time = time.time()

    def release(self) -> None:
        if self._released:
            return
        self.cap.release()
        self._released = True

    def get_fps(self) -> float:
        self.fps = self._declared_fps if self._declared_fps > 0.0 else self._measured_fps()
        return self.fps

    def _compute_frame_time(self, reported_fps: float) -> float:
        if reported_fps > 0.0:
            return self.stream_frames_consumed / reported_fps
        if not self._released:
            msec = self.cap.get(cv2.CAP_PROP_POS_MSEC)
            if msec and msec > 0:
                return float(msec) / 1000.0
        return self._fallback_frame_time()

    def get_frame(self) -> np.ndarray | None:
        if self._released:
            return None

        if self.skip_frames <= 1:
            ret, frame = self.cap.read()
        else:
            for _ in range(self.skip_frames - 1):
                if not self.cap.grab():
                    self.release()
                    return None
            ret, frame = self.cap.retrieve()
            if not ret:
                ret, frame = self.cap.read()

        if not ret or frame is None or not isinstance(frame, np.ndarray):
            self.release()
            return None

        self.height, self.width = frame.shape[:2]
        self.frames_yielded += 1
        self.stream_frames_consumed += self.skip_frames
        return frame

    def __iter__(self):
        return self

    def __next__(self) -> np.ndarray:
        frame = self.get_frame()
        if frame is None:
            raise StopIteration
        return frame


class FrameReaderFabric(AbstractFrameReaderFabric):
    _BACKENDS: dict[str, type[AbstractFrameReader]] = {
        "opencv": OpenCVReader,
        "pyav": PyAVReader,
    }

    def __init__(self) -> None:
        self._backends = dict(self._BACKENDS)

    def register_backend(self, name: str, backend: type[AbstractFrameReader]) -> None:
        if name in self._backends:
            raise ValueError(
                f"Бэкенд {name!r} уже зарегистрирован. "
                f"Доступные: {list(self._backends)}"
            )
        self._backends[name] = backend

    def get_reader(self) -> AbstractFrameReader:
        """
        Бэкенд читается из settings.reader_backend через экземпляр ридера.
        Сначала создаём временный объект нужного класса — он сам знает свой бэкенд.

        Но правильнее: читаем бэкенд ДО создания ридера через ConfigManager напрямую.
        """
        service_name = os.getenv("SERVICE_NAME", "")
        if not service_name:
            raise EnvironmentError("SERVICE_NAME environment variable not set")

        cfg = ConfigManager().config_service_name(service_name)
        if not cfg:
            raise ValueError(f"No config for service '{service_name}'")

        settings = cfg.get("settings", {})
        backend_name = (
            settings.get("reader_backend")
            or os.getenv("CV_BACKEND", "opencv")).lower()

        if backend_name not in self._backends:
            raise ValueError(
                f"Бэкенд {backend_name!r} не найден. "
                f"Доступные: {list(self._backends)}"
            )

        return self._backends[backend_name]()
