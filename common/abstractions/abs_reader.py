import os
import time
import hashlib
from abc import ABC, abstractmethod

import numpy as np

from common.utils.data_types import FrameReaderInfo, RawFrame
from common.singeleton.config_manager import ConfigManager


class AbstractFrameReader(ABC):
    DEFAULT_FPS = 20.0

    def __init__(self) -> None:
        self.fps: float = 0.0
        self.frames_yielded: int = 0
        self.stream_frames_consumed: int = 0
        self._declared_fps: float = 0.0
        self.width: int | None = None
        self.height: int | None = None
        self.start_time: float = 0.0

        self._service_name = os.getenv("SERVICE_NAME")  # только один раз
        self._validate_env()

        self._cfg = ConfigManager().config_service_name(self._service_name)
        self._validate_cfg()

        self.reader_settings = self._cfg.get("settings")
        self._validate_settings()

        self.source: str = self.reader_settings.get("source")
        self.cam_id: str = hashlib.md5(self.source.encode("utf-8")).hexdigest()[:6]

        raw_skip = self.reader_settings.get("skip_frames", 1)
        try:
            self.skip_frames = max(1, int(raw_skip))
        except (TypeError, ValueError):
            self.skip_frames = 1

    def _validate_env(self) -> None:
        if not self._service_name:
            raise EnvironmentError("SERVICE_NAME environment variable not set")

    def _validate_cfg(self) -> None:
        if not self._cfg:
            raise ValueError(f"No config found for service '{self._service_name}'")

    def _validate_settings(self) -> None:
        if not self.reader_settings:
            raise ValueError(f"No 'settings' in config for service '{self._service_name}'")
        if not self.reader_settings.get("source"):
            raise ValueError(f"'source' not set in settings for service '{self._service_name}'")

    @property
    def backend_name(self) -> str:
        """Бэкенд из настроек конкретного сервиса."""
        return self.reader_settings.get("reader_backend", "opencv").lower()

    def _measured_fps(self) -> float:
        elapsed = time.time() - self.start_time
        if elapsed > 1.0 and self.frames_yielded > 0:
            return self.frames_yielded / elapsed
        return self.DEFAULT_FPS

    def _compute_frame_time(self, reported_fps: float) -> float:
        if reported_fps > 0.0:
            return self.stream_frames_consumed / reported_fps
        return self._fallback_frame_time()

    def _fallback_frame_time(self) -> float:
        mf = self._measured_fps()
        return self.stream_frames_consumed / mf if mf > 0 else 0.0

    @property
    def source_id(self) -> str:
        """cam_id как property (без get_ префикса — PEP 8)."""
        return self.cam_id

    def collect_info(self) -> FrameReaderInfo:
        fps = self.get_fps()
        return FrameReaderInfo(
            cam_id=self.cam_id,           # было self.set_source_id — баг
            source_fps=fps,
            frames_yielded=self.frames_yielded,
            stream_frames_consumed=self.stream_frames_consumed,
            frame_width=self.width,
            frame_height=self.height,
            frame_time=self._compute_frame_time(fps),
        )

    def to_raw_frame(self, frame: np.ndarray) -> RawFrame:
        """Оборачивает кадр в датакласс для передачи в pipeline."""
        fps = self.get_fps()
        return RawFrame(
            cam_id=self.cam_id,
            frame=frame,
            frame_time=self._compute_frame_time(fps),
            source_fps=fps,
        )

    # ── Abstract ──────────────────────────────────────────────────────────────

    @abstractmethod
    def get_fps(self) -> float:
        pass

    @abstractmethod
    def release(self) -> None:
        pass

    @abstractmethod
    def get_frame(self) -> np.ndarray | None:
        pass


class AbstractFrameReaderFabric(ABC):
    @abstractmethod
    def get_reader(self) -> AbstractFrameReader:
        pass

    @abstractmethod
    def register_backend(self, name: str, backend: type[AbstractFrameReader]) -> None:
        pass
