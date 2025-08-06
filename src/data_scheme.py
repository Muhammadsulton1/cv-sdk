from typing import List, Dict, Union, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


class InferenceDataSchema(BaseModel):
    boxes: List[List[Union[float, int]]] = Field(..., description="Список ббоксов объектов")
    score: List[float] = Field(..., description="Список уверенностей для каждого объекта")
    mask: Optional[List[List[Union[float, bool]]]] = Field(None, description="Список масок объектов")

    @model_validator(mode='after')
    def validate_list_lengths(self) -> 'InferenceDataSchema':
        if len(self.boxes) != len(self.score):
            raise ValueError("Длина списка boxes должна быть равна длине списка score")

        if self.mask is not None and len(self.mask) != len(self.score):
            raise ValueError("Длина списка mask должна быть равна длине списка score")
        return self

    @field_validator('boxes')
    def validate_bbox_structure(cls, bbox_list):
        for bbox in bbox_list:
            if len(bbox) != 4:
                raise ValueError("Каждый ббокс должен содержать ровно 4 координаты")
            if not all(isinstance(coord, (int, float)) for coord in bbox):
                raise ValueError("Координаты ббокса должны быть числами (int или float)")
        return bbox_list

    @field_validator('score')
    def validate_scores(cls, score_list):
        if not all(isinstance(score, float) for score in score_list):
            raise ValueError("Все значения уверенности должны быть float")
        return score_list

    @field_validator('mask')
    def validate_masks(cls, mask_list):
        if mask_list is None:
            return None

        if len(mask_list) < 4:
            raise ValueError("маска должна содержать минимум 5 элементов")

        # for mask in mask_list:
        #     if len(mask) < 2:
        #         raise ValueError("Каждая маска должна содержать минимум 5 элементов")
        return mask_list


class InferenceOutputSchema(BaseModel):
    predictions: Dict[
        Union[str, int],
        InferenceDataSchema
    ] = Field(..., description="Словарь предсказаний по классам")

    @field_validator('predictions')
    def validate_class_structure(cls, pred_dict):
        for class_name in pred_dict.keys():
            if not isinstance(class_name, (str, int)) or isinstance(class_name, bool):
                allowed_types = "str или int (но не bool)"
                raise TypeError(f"Ключ класса должен быть {allowed_types}, получен {type(class_name)}")
        return pred_dict


if __name__ == '__main__':

    valid_data = {
        "predictions": {
            0: {
                "boxes": [[10, 20, 30, 40], [50, 60, 70, 80]],
                "score": [0.9, 0.8],
                "mask": [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]
            },
            "car": {
                "boxes": [[15, 25, 35, 45]],
                "score": [0.95]
            }
        }
    }

    invalid_data = {
        "predictions": {
            True: {
                "boxes": [[1, 2, 3, 4], [1, 2, 3, 4], [1, 1, 1, 1], [1, 1, 1, 1], [2, 2, 2, 2]],
                "score": [0.1, 0.1, 0.1, 0.1, 0.1],
                "mask": [[1], [1, 2], [13, 1, 1, 1], [12], [13]]
            }
        }
    }

    parsed = InferenceOutputSchema.model_validate(invalid_data)

    if parsed:
        print(parsed.predictions.keys())
    else:
        print('not')
