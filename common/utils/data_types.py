# common/utils/data_types.py
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(slots=True)
class FrameReaderInfo:
    """Метаданные потока — результат collect_info()."""
    cam_id: str
    source_fps: float
    frames_yielded: int
    stream_frames_consumed: int
    frame_width: Optional[int]
    frame_height: Optional[int]
    frame_time: float


@dataclass(slots=True)
class RawFrame:
    """Сырой кадр от ридера — входные данные для pipeline."""
    cam_id: str
    frame: np.ndarray
    frame_time: float
    source_fps: float


@dataclass(slots=True)
class FrameData:
    """Входные данные от ридера — всё что нужно для загрузки."""
    bucket: str
    cam_id: str
    key: str
    frame: bytes


@dataclass(slots=True)
class S3UploadData:
    """Результат после успешного upload — без bytes (уже в S3)."""
    bucket: str
    cam_id: str
    key: str  # full_key: cam_{cam_id}/{uuid}.jpg


@dataclass(slots=True)
class S3DownloadData:
    """Результат после успешного download — с байтами кадра."""
    bucket: str
    cam_id: str
    key: str
    frame: bytes
