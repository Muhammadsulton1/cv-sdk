import asyncio
import numpy as np

from typing import Any, Dict
from abs_src.modules_interface import ModelInference


class DetectionInference(ModelInference):
    def __init__(self):
        super().__init__("detection_inference")
        #self.model = model('best.pt')

    async def preprocess(self, image: np.ndarray) -> Any:
        """Пример предобработки для детекции лиц"""
        return image.resize((640, 640))

    async def inference(self, preprocessed_data: Any) -> Any:
        """Пример выполнения инференса"""
        return {"boxes": [[10, 20, 100, 100]]}  # Пример результата

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
        super().__init__("classification")

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

    # Подключаемся к NATS
    nats_servers = "nats://localhost:4222"
    await face_detector.connect_nats(nats_servers, "inference_frames")
    await classifier.connect_nats(nats_servers, "inference_frames")

    while True:
        await asyncio.sleep(1)


if __name__ == "__main__":
    asyncio.run(main())