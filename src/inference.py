import asyncio
import numpy as np

from typing import Any, Dict
from abs_src.abs_inference import ModelInference


class DetectionInference(ModelInference):
    def __init__(self):
        super().__init__()

    async def preprocess(self, image: np.ndarray) -> Any:
        """Пример предобработки для детекции лиц"""
        return image.resize((640, 640))

    async def inference(self, preprocessed_data: Any) -> Any:
        """Пример выполнения инференса"""
        return {"boxes": [[10, 20, 100, 100]]}

    async def postprocess(self, inference_output: Any) -> Dict[str, Any]:
        """Постобработка результатов"""

        return {
            "detections": [
                {
                    "box": box,
                    "confidence": 0.95
                }
                for box in inference_output["boxes"]
            ]
        }


class ClassificationInference(ModelInference):
    def __init__(self):
        super().__init__()

    async def preprocess(self, image: np.ndarray) -> Any:
        return image.resize((224, 224))

    async def inference(self, preprocessed_data: Any) -> Any:
        return {"class": 42, "confidence": 0.87}

    async def postprocess(self, inference_output: Any) -> Dict[str, Any]:
        class_names = {42: "cat"}
        return {
            "class_id": inference_output["class"],
            "class_name": class_names.get(inference_output["class"], "unknown"),
            "confidence": inference_output["confidence"]
        }


async def main():
    face_detector = DetectionInference()
    classifier = ClassificationInference()

    await asyncio.gather(
        face_detector.connect_nats(),
        classifier.connect_nats()
    )
    while True:
        await asyncio.sleep(1)


if __name__ == "__main__":
    asyncio.run(main())
