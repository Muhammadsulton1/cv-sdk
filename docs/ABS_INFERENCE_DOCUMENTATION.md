# Документация по использованию abs_inference

## Обзор

Модуль `abs_inference` предоставляет абстрактную архитектуру для создания моделей машинного обучения с поддержкой:
- Конвертации моделей в оптимизированные форматы (TensorRT)
- Асинхронной обработки через NATS
- Регистрации сервисов в Redis
- Загрузки изображений по URL
- Стандартизированного пайплайна инференса

## Архитектура

### Основные классы

1. **AbstractConverter** - базовый класс для конвертации моделей
2. **TensorRTConverter** - конвертер ONNX → TensorRT
3. **BaseInferenceModel** - базовый класс для моделей инференса

## AbstractConverter

Абстрактный базовый класс для конвертации моделей в оптимизированные форматы.

### Конструктор

```python
def __init__(self, model_path: str, converted_path: str, input_name: str, output_name: str)
```

**Параметры:**
- `model_path` - путь к исходной модели
- `converted_path` - путь для сохранения конвертированной модели
- `input_name` - имя входного тензора модели
- `output_name` - имя выходного тензора модели

### Методы

- `load_or_convert()` - проверяет наличие конвертированной модели и загружает/конвертирует её
- `convert_model()` - абстрактный метод конвертации (реализуется в наследниках)
- `load_engine()` - абстрактный метод загрузки модели (реализуется в наследниках)

## TensorRTConverter

Реализация конвертера моделей в формат TensorRT.

### Конструктор

```python
def __init__(self, model_path: str, converted_path: str, input_name: str, output_name: str)
```

### Особенности

- Автоматически конвертирует ONNX модели в TensorRT engine
- Поддерживает статические и динамические размеры входов
- Использует переменную окружения `ONNX_DYNAMIC_AXIS` для выбора режима

### Динамические размеры

При установке `ONNX_DYNAMIC_AXIS=true` создается профиль с параметрами:
- **min**: (1, 3, 224, 224)
- **max**: (32, 3, 640, 640)  
- **opt**: (1, 3, 640, 640)

## BaseInferenceModel

Базовый класс для моделей инференса с поддержкой NATS и Redis.

### Переменные окружения

```bash
nats_host=nats://localhost:4222  # Адреса NATS-серверов (через запятую)
redis_host=localhost             # Хост Redis
redis_port=6379                  # Порт Redis
routing_ttl=10                   # TTL регистрации в Redis (сек)
```

### Абстрактные методы

Необходимо реализовать в наследниках:

```python
def preprocess(self, image: Any, *args, **kwargs) -> Any:
    """Предобработка входных данных"""
    pass

def postprocess(self, inference_output: Any, *args, **kwargs) -> Dict[str, Any]:
    """Постобработка результатов"""
    pass

def inference(self, input_data: np.ndarray) -> Any:
    """Выполнение инференса"""
    pass
```

### Основные методы

- `run_inference(image)` - полный пайплайн обработки
- `connect_nats()` - подключение к NATS и подписка на топик
- `register_service()` - регистрация в Redis для маршрутизации
- `download_image(url)` - загрузка изображения по URL

## Примеры использования

### 1. Модель YOLO с TensorRT

```python
from abs_src.abs_inference import BaseInferenceModel, TensorRTConverter
from polygraphy.backend.trt import TrtRunner
import numpy as np
from PIL import Image

class YOLOModelTRT(TensorRTConverter, BaseInferenceModel):
    def __init__(self, model_path: str, converted_path: str, input_name: str, output_name: str):
        super().__init__(model_path, converted_path, input_name, output_name)

    def preprocess(self, image: Image.Image, image_size=640) -> np.ndarray:
        """Предобработка изображения для YOLO"""
        img = image.convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
        return transform(img).unsqueeze(0).numpy().astype(np.float32)

    def inference(self, image: np.ndarray) -> np.ndarray:
        """Выполнение инференса через TensorRT"""
        with TrtRunner(self.engine) as runner:
            outputs = runner.infer(
                feed_dict={self.input_name: image},
                copy_outputs_to_host=True
            )
        return outputs[self.output_name]

    def postprocess(self, output, conf_threshold=0.25, iou_threshold=0.45, image_size=640):
        """Постобработка результатов YOLO с NMS"""
        output = output.transpose(0, 2, 1)
        predictions = output[0]

        # Извлечение боксов, скоров и классов
        boxes = predictions[:, :4]
        scores = np.max(predictions[:, 4:], axis=1)
        class_ids = np.argmax(predictions[:, 4:], axis=1)

        # Фильтрация по confidence
        valid = scores > conf_threshold
        boxes = boxes[valid]
        scores = scores[valid]
        class_ids = class_ids[valid]

        # Конвертация в формат xyxy
        boxes_xyxy = np.zeros_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2

        # Обрезка по границам изображения
        boxes_xyxy[:, [0, 2]] = np.clip(boxes_xyxy[:, [0, 2]], 0, image_size)
        boxes_xyxy[:, [1, 3]] = np.clip(boxes_xyxy[:, [1, 3]], 0, image_size)

        if len(boxes_xyxy) == 0:
            return {"boxes": [], "scores": [], "class_ids": []}

        # Non-Maximum Suppression
        boxes_tensor = torch.tensor(boxes_xyxy, dtype=torch.float32)
        scores_tensor = torch.tensor(scores, dtype=torch.float32)
        keep_indices = ops.nms(boxes_tensor, scores_tensor, iou_threshold)

        return {
            "boxes": boxes_xyxy[keep_indices].tolist(),
            "scores": scores[keep_indices].tolist(),
            "class_ids": class_ids[keep_indices].tolist()
        }

# Использование
trt_model = YOLOModelTRT(
    model_path="yolov8n.onnx",
    converted_path="yolov8n.engine",
    input_name='images',
    output_name='output0'
)

image = Image.open("test1.jpg")
result = trt_model.run_inference(image)
print(result)
```

### 2. Модель YOLO с PyTorch

```python
from ultralytics import YOLO

class YOLOModel(BaseInferenceModel):
    def __init__(self, model_path: str):
        super().__init__()
        self.model = YOLO(model_path)

    def preprocess(self, image, **kwargs) -> np.ndarray:
        """Для Ultralytics YOLO предобработка не требуется"""
        return image

    def inference(self, image: np.ndarray) -> Any:
        """Выполнение инференса через Ultralytics"""
        return self.model(image)

    def postprocess(self, output: Any, **kwargs) -> Dict[str, Any]:
        """Извлечение результатов из Ultralytics output"""
        result = output[0]
        return {
            "boxes": result.boxes.xyxy.cpu().numpy().tolist(),
            "scores": result.boxes.conf.cpu().numpy().tolist(),
            "class_ids": result.boxes.cls.cpu().numpy().astype(int).tolist()
        }

# Использование
model = YOLOModel(model_path='yolo11n.pt')
img = cv2.imread('test1.jpg')
result = model.run_inference(img)
```

### 3. Асинхронная обработка через NATS

```python
async def main():
    # Создание модели
    model = YOLOModel(model_path='yolo11n.pt')
    
    # Подключение к NATS и запуск обработки
    await model.connect_nats()

if __name__ == "__main__":
    asyncio.run(main())
```

## Протокол NATS сообщений

### Входящие сообщения

Модель подписывается на топик с именем класса и ожидает JSON сообщения:

```json
{
    "seaweed_url": "http://example.com/image.jpg",
    "frame_id": "unique_frame_identifier"
}
```

### Исходящие сообщения

Результаты отправляются в топик `inference.results`:

```json
{
    "model": "YOLOModel",
    "result": {
        "boxes": [[x1, y1, x2, y2], ...],
        "scores": [0.95, 0.87, ...],
        "class_ids": [0, 1, ...]
    },
    "frame_id": "unique_frame_identifier"
}
```

## Регистрация в Redis

Модели автоматически регистрируются в Redis для маршрутизации:
- **Ключ**: `routing_to_models`
- **Тип**: SET
- **Значение**: имя класса модели
- **TTL**: значение из `routing_ttl` (по умолчанию 10 сек)

## Обработка ошибок

Система включает обработку следующих ошибок:
- Ошибки загрузки изображений по URL
- JSON decode ошибки в NATS сообщениях
- Ошибки конвертации моделей
- Ошибки подключения к Redis/NATS

Все ошибки логируются через модуль `utils.logger`.

## Рекомендации по использованию

1. **Переменные окружения**: Настройте подключения к NATS и Redis через переменные окружения
2. **Имена моделей**: Используйте осмысленные имена классов - они становятся именами топиков NATS
3. **Обработка ошибок**: Реализуйте дополнительную обработку ошибок в методах preprocess/postprocess
4. **Производительность**: Для высокой производительности используйте TensorRT конвертер
5. **Мониторинг**: Следите за логами для отслеживания работы системы

## Зависимости

Основные зависимости модуля:
- `numpy` - работа с массивами
- `PIL` - обработка изображений  
- `aiohttp` - асинхронные HTTP запросы
- `nats-py` - NATS клиент
- `redis` - Redis клиент
- `polygraphy` - TensorRT интеграция
- `torch` - PyTorch (для NMS)
- `ultralytics` - YOLO модели (опционально)