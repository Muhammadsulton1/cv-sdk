from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

import re
from dataclasses import dataclass
from datetime import datetime, timezone

Number = Union[int, float]
MaskValue = Union[int, float, bool]


class FrameEventPayload(BaseModel):
    """
    Событие кадра: ссылка на объект в S3 + метаданные.
    Сериализация в NATS — JSON (bytes UTF-8).
    """

    cam_id: str = Field(..., description="Идентификатор камеры/источника")
    device_id: str = Field(..., description="Уникальное имя камеры/источника который вы задаете через .env")
    frame_id: str = Field(..., description="Уникальный id кадра в рамках пайплайна")
    timestamp: float = Field(..., description="Unix timestamp (сек), обычно UTC")
    s3_bucket: str = Field(..., description="Название бакета куда кладется кадр")

    s3_key: str = Field(..., description="Ключ объекта в bucket")
    s3_url: str = Field(..., description="URI вида s3://bucket/key или http(s) для совместимых endpoint")

    frame_width: int = Field(..., ge=1)
    frame_height: int = Field(..., ge=1)

    source_fps: Optional[float] = Field(default=None, description="Заявленный/измеренный FPS источника")
    frames_yielded: Optional[int] = Field(default=None)
    stream_frames_consumed: Optional[int] = Field(default=None)
    frame_time_sec: Optional[float] = Field(default=None, description="Время относительно начала потока, сек")

    extra: Dict[str, Any] = Field(default_factory=dict, description="Доп. поля без ломки контракта")

    def to_json_bytes(self) -> bytes:
        return self.model_dump_json().encode("utf-8")

    @classmethod
    def from_json_bytes(cls, data: bytes) -> "FrameEventPayload":
        return cls.model_validate_json(data.decode("utf-8"))


class InferenceDataSchema(BaseModel):
    boxes: List[List[Number]] = Field(
        ...,
        description="Список ббоксов объектов",
    )
    score: List[float] = Field(
        ...,
        description="Список уверенностей для каждого объекта",
    )
    mask: Optional[List[List[MaskValue]]] = Field(
        None,
        description="Список масок объектов",
    )

    @model_validator(mode='after')
    def validate_list_lengths(self) -> 'InferenceDataSchema':
        if len(self.boxes) != len(self.score):
            raise ValueError("Длина списка boxes должна быть равна длине списка score")

        if self.mask is not None and len(self.mask) != len(self.score):
            raise ValueError("Длина списка mask должна быть равна длине списка score")

        return self

    @field_validator('boxes')
    @classmethod
    def validate_bbox_structure(cls, bbox_list: List[List[Number]]) -> List[List[Number]]:
        for bbox in bbox_list:
            if len(bbox) != 4:
                raise ValueError("Каждый ббокс должен содержать ровно 4 координаты")
            if not all(isinstance(coord, (int, float)) for coord in bbox):
                raise ValueError("Координаты ббокса должны быть числами (int или float)")
        return bbox_list

    @field_validator('score')
    @classmethod
    def validate_scores(cls, score_list: List[float]) -> List[float]:
        for score in score_list:
            if not isinstance(score, (int, float)):
                raise ValueError("Все значения уверенности должны быть числами")
        return [float(score) for score in score_list]

    @field_validator('mask')
    @classmethod
    def validate_masks(
            cls,
            mask_list: Optional[List[List[MaskValue]]]
    ) -> Optional[List[List[MaskValue]]]:
        if mask_list is None:
            return None

        for mask in mask_list:
            if not isinstance(mask, list):
                raise ValueError("Каждая маска должна быть списком")
            if len(mask) < 5:
                raise ValueError("Каждая маска должна содержать минимум 5 элементов")

        return mask_list


class InferenceOutputSchema(BaseModel):
    predictions: Dict[Union[str, int], InferenceDataSchema] = Field(
        ...,
        description="Словарь предсказаний по классам",
    )

    @field_validator('predictions', mode='before')
    @classmethod
    def validate_class_structure(cls, pred_dict: Any):
        if not isinstance(pred_dict, dict):
            raise TypeError("predictions должен быть словарем")

        for class_name in pred_dict.keys():
            if isinstance(class_name, bool):
                raise TypeError("Ключ класса должен быть str или int, но не bool")
            if not isinstance(class_name, (str, int)):
                raise TypeError(
                    f"Ключ класса должен быть str или int, получен {type(class_name).__name__}"
                )

        return pred_dict


class InferenceResultPayload(BaseModel):
    """Результат инференса для публикации downstream (Registrator и т.д.)."""

    cam_id: str
    device_id: str
    frame_id: str
    timestamp: float
    s3_key: str
    s3_bucket: str
    inference: InferenceOutputSchema
    extra: Dict[str, Any] = Field(default_factory=dict)

    def to_json_bytes(self) -> bytes:
        return self.model_dump_json().encode("utf-8")

    @classmethod
    def from_json_bytes(cls, data: bytes) -> "InferenceResultPayload":
        return cls.model_validate_json(data.decode("utf-8"))


def _safe_segment(s: str) -> str:
    t = str(s).strip().replace("/", "_").replace("\\", "_")
    if not t:
        return "default"
    return t


@dataclass(frozen=True)
class S3ObjectKeyBuilder:
    """
    Схема: ``{cam_id}/{YYYY-MM-DD}/{frame_id}{extension}``
    (cam_id без префикса cam_ — при необходимости добавьте в конфиге).
    """

    use_utc: bool = True

    def build(
            self,
            cam_id: str,
            frame_id: str,
            timestamp: float,
            extension: str = ".jpg",
    ) -> str:
        cam = _safe_segment(cam_id)
        fid = _safe_segment(frame_id)
        if not re.match(r"^[a-zA-Z0-9._-]+$", fid.replace(".", "_")):
            fid = re.sub(r"[^a-zA-Z0-9._-]+", "_", fid)
        ext = extension if extension.startswith(".") else f".{extension}"
        tz = timezone.utc if self.use_utc else datetime.now().astimezone().tzinfo
        day = datetime.fromtimestamp(timestamp, tz=tz).strftime("%Y-%m-%d")
        return f"{cam}/{day}/{fid}{ext}"


if __name__ == '__main__':
    valid_data = {
        "predictions": {
            0: {
                "boxes": [[10, 20, 30, 40], [50, 60, 70, 80]],
                "score": [0.9, 0.8],
                "mask": [
                    [1, 2, 3, 4, 5],
                    [6, 7, 8, 9, 10]
                ]
            },
            "car": {
                "boxes": [[15, 25, 35, 45]],
                "score": [0.95]
            }
        }
    }

    invalid_data = {
        "predictions": {
            'car': {
                "boxes": [
                    [1, 2, 3, 4],
                    [1, 2, 3, 4],
                    [1, 1, 1, 1],
                    [1, 1, 1, 1],
                    [2, 2, 2, 2]
                ],
                "score": [0.1, 0.1, 0.1, 0.1, 0.1],
                "mask": [
                    [1],
                    [1, 2],
                    [13, 1, 1, 1],
                    [12],
                    [13],
                    [15]
                ]
            }
        }
    }

    print("=== VALID DATA ===")
    try:
        parsed_valid = InferenceOutputSchema.model_validate(valid_data)
        print("OK")
        print(parsed_valid.model_dump())
    except ValidationError as e:
        print("ValidationError:")
        print(e)

    print("\n=== INVALID DATA ===")
    try:
        parsed_invalid = InferenceOutputSchema.model_validate(invalid_data)
        print("OK")
        print(parsed_invalid.model_dump())
    except ValidationError as e:
        print("ValidationError:")
        print(e)
