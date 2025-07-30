from typing import Dict, Any

import cv2
import numpy as np
import torch
import asyncio
from PIL import Image
import torchvision.transforms as transforms
import torchvision.ops as ops
from ultralytics import YOLO

from abs_src.abs_inference import BaseInferenceModel, TensorRTConverter
from polygraphy.backend.trt import TrtRunner


class YOLOModelTRT(TensorRTConverter, BaseInferenceModel):
    def __init__(
            self,
            model_path: str,
            converted_path: str,
            input_name: str,
            output_name: str,
    ) -> None:
        super().__init__(model_path, converted_path, input_name, output_name)

    def preprocess(self, image: Image.Image, image_size=640) -> np.ndarray:
        img = image.convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
        return transform(img).unsqueeze(0).numpy().astype(np.float32)

    def inference(self, image: np.ndarray) -> np.ndarray:
        with TrtRunner(self.engine) as runner:  # Используем готовый движок
            outputs = runner.infer(
                feed_dict={self.input_name: image},
                copy_outputs_to_host=True
            )
        return outputs[self.output_name]

    def postprocess(self, output, conf_threshold=0.25, iou_threshold=0.45, image_size=640):
        output = output.transpose(0, 2, 1)
        predictions = output[0]

        boxes = predictions[:, :4]
        scores = np.max(predictions[:, 4:], axis=1)
        class_ids = np.argmax(predictions[:, 4:], axis=1)

        valid = scores > conf_threshold
        boxes = boxes[valid]
        scores = scores[valid]
        class_ids = class_ids[valid]

        boxes_xyxy = np.zeros_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2

        boxes_xyxy[:, [0, 2]] = np.clip(boxes_xyxy[:, [0, 2]], 0, image_size)
        boxes_xyxy[:, [1, 3]] = np.clip(boxes_xyxy[:, [1, 3]], 0, image_size)

        if len(boxes_xyxy) == 0:
            return {"boxes": [], "scores": [], "class_ids": []}

        boxes_tensor = torch.tensor(boxes_xyxy, dtype=torch.float32)
        scores_tensor = torch.tensor(scores, dtype=torch.float32)
        keep_indices = ops.nms(boxes_tensor, scores_tensor, iou_threshold)

        return {
            "boxes": boxes_xyxy[keep_indices].tolist(),
            "scores": scores[keep_indices].tolist(),
            "class_ids": class_ids[keep_indices].tolist()
        }


class YOLOModel(BaseInferenceModel):
    def __init__(self, model_path: str):
        super().__init__()
        self.model = YOLO(model_path)

    def preprocess(self, image, **kwargs) -> np.ndarray:
        return image

    def inference(self, image: np.ndarray) -> Any:
        return self.model(image)

    def postprocess(self, output: Any, **kwargs) -> Dict[str, Any]:
        result = output[0]
        return {
            "boxes": result.boxes.xyxy.cpu().numpy().tolist(),
            "scores": result.boxes.conf.cpu().numpy().tolist(),
            "class_ids": result.boxes.cls.cpu().numpy().astype(int).tolist()
        }


# async def main():
#     trt_model = YOLOModelTRT(
#         model_path="yolov8n.onnx",
#         converted_path="yolov8n.engine",
#         input_name='images',
#         output_name='output0'
#     )
#
#     image = Image.open("test1.jpg")
#     img = cv2.imread('test1.jpg')
#     img = cv2.resize(img, (640, 640))
#     result = await trt_model.run_inference(image)
#     boxes = result["boxes"]
#     for box in boxes:
#         x, y, w, h = map(int, box)
#         cv2.rectangle(img, (x, y), (w, h), (0, 255, 0))
#
#     cv2.imwrite('aa.jpg', img)
#     print(result)


async def main():
    model = YOLOModel(model_path='yolo11n.pt')
    img = cv2.imread('../test1.jpg')
    result = model.run_inference(img)
    boxes = result["boxes"]
    for box in boxes:
        x, y, w, h = map(int, box)
        cv2.rectangle(img, (x, y), (w, h), (0, 255, 0))

    cv2.imwrite('../aa.jpg', img)
    print(result)


if __name__ == "__main__":
    asyncio.run(main())