from typing import Dict, Any

import numpy as np
import asyncio
from ultralytics import YOLO

from abs_src.abs_inference import BaseInferenceModel


class YOLOModelNew(BaseInferenceModel):
    def __init__(self, model_path: str):
        super().__init__()
        self.model = YOLO(model_path)

    def preprocess(self, image, **kwargs) -> np.ndarray:
        return image

    def inference(self, image: np.ndarray) -> Any:
        return self.model(image)

    def postprocess(self, output: Any, **kwargs) -> Dict[str, Any]:
        result = output[0]

        classes = result.boxes.cls.cpu().numpy().astype(int).tolist()
        boxes = result.boxes.xyxy.cpu().numpy().tolist()
        scores = result.boxes.conf.cpu().numpy().tolist()

        predictions = {}
        for idx in range(len(classes)):
            cls_id = classes[idx]
            box = boxes[idx]
            score = scores[idx]

            if cls_id not in predictions:
                predictions[cls_id] = {
                    "boxes": [],
                    "score": [],
                }

            predictions[cls_id]["boxes"].append(box)
            predictions[cls_id]["score"].append(score)

        return {"predictions": predictions}


async def main():
    model = YOLOModelNew(model_path='../weights/yolo11n.pt')
    await model.connect_nats()


if __name__ == "__main__":
    asyncio.run(main())
