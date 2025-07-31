# Документация по использованию abs_registrator

## Обзор

Модуль `abs_registrator` предоставляет абстрактную архитектуру для регистрации и обработки событий с поддержкой:
- Управления жизненным циклом задач
- Записи видео из последовательности кадров
- Сохранения ключевых кадров как изображений
- Интеграции с NATS для получения результатов инференса
- Сохранения событий в Redis с временными метками

## Архитектура

### Основные классы

1. **AbstractEventRegistrator** - абстрактный базовый класс для регистрации событий
2. **VideoWriter** - класс для записи видео и сохранения кадров
3. **BasicEventRegistrator** - базовая реализация регистратора событий

## AbstractEventRegistrator

Абстрактный базовый класс для регистрации и управления событиями.

### Конструктор

```python
def __init__(self):
    self.redis = Redis(
        host=os.getenv('REDIS_HOST', 'localhost'),
        port=int(os.getenv('REDIS_PORT', 6379)),
        decode_responses=False
    )
    self.nats_conn = None
    self.task_catalog = defaultdict(lambda: {
        'task': '',
        'status': [],
    })
    self._nats_connected = False
```

### Переменные окружения

```bash
REDIS_HOST=localhost    # Хост Redis
REDIS_PORT=6379        # Порт Redis
NATS_HOST=nats://localhost:4222  # Адреса NATS-серверов (через запятую)
TMP_PATH=/tmp          # Путь для временных файлов
```

### Структура task_catalog

Каждая задача в каталоге имеет структуру:
```python
{
    'task': 'CREATED|PENDING|DONE|FINISHED|ERROR',
    'status': [True, False, True, ...]  # Список статусов
}
```

### Абстрактные методы

Необходимо реализовать в наследниках:

```python
def create_task(self, key_name: str) -> None:
    """Создание новой задачи"""
    pass

def stop_task(self, key_name: str) -> None:
    """Остановка задачи"""
    pass

def delete_task(self, key_name: str) -> None:
    """Удаление задачи"""
    pass

def add_task(self, key_name: str, flag: bool) -> None:
    """Добавление статуса к задаче"""
    pass

def clear_key_task(self, key_name: str) -> None:
    """Очистка статусов задачи"""
    pass

def add_video_frame(self, key_name: str, frame: np.ndarray) -> None:
    """Добавление кадра к задаче"""
    pass

async def register_task(self, key_name: str, message: dict) -> None:
    """Регистрация завершенной задачи"""
    pass
```

### Основные методы

#### most_common(key_name: str) -> bool | str
Возвращает наиболее частый статус для задачи.

```python
# Пример использования
registrator = BasicEventRegistrator()
registrator.add_task('task1', True)
registrator.add_task('task1', False)
registrator.add_task('task1', True)
most_frequent = registrator.most_common('task1')  # Вернет True
```

#### connect_nats() -> None
Устанавливает соединение с NATS и подписывается на топик `inference.results`.

```python
await registrator.connect_nats()
```

#### close() -> None
Корректно закрывает соединения с NATS и Redis.

```python
await registrator.close()
```

#### message_handler(msg) -> dict
Статический метод для обработки входящих NATS сообщений.

## VideoWriter

Класс для управления записью видео и сохранением кадров.

### Конструктор

```python
def __init__(self):
    self.tmp_path = os.getenv("TMP_PATH")
    os.makedirs(self.tmp_path, exist_ok=True)
    self.video_frame_store = defaultdict(self._create_video_entry)
```

### Структура video_frame_store

Каждая запись имеет структуру:
```python
{
    'frames': [frame1, frame2, ...],           # Список кадров
    'event_proof_name': 'uuid',                # Уникальный идентификатор
    'frame_proof_path': '/path/to/frame.jpg',  # Путь к ключевому кадру
    'video_proof_path': '/path/to/video.mp4'   # Путь к видеофайлу
}
```

### Методы

#### add_frame(key: str, frame: np.ndarray) -> None
Добавляет кадр в буфер для указанного ключа.

```python
video_writer = VideoWriter()
frame = np.zeros((480, 640, 3), dtype=np.uint8)
video_writer.add_frame('camera1', frame)
```

#### save_keyframe(key: str, frame: np.ndarray) -> None
Сохраняет ключевой кадр как JPEG изображение.

```python
video_writer.save_keyframe('camera1', frame)
```

#### save_video(key: str) -> None
Сохраняет видео из всех собранных кадров в формате MP4 с кодеком H.264.

**Параметры видео:**
- **Кодек**: H.264
- **Частота кадров**: 10 FPS
- **Формат пикселей**: YUV420P

```python
video_writer.save_video('camera1')
```

## BasicEventRegistrator

Базовая реализация регистратора событий.

### Конструктор

```python
def __init__(self):
    super().__init__()
    self.video_writer = VideoWriter()
```

### Реализация абстрактных методов

#### create_task(key_name: str) -> None
Создает новую задачу со статусом 'CREATED'.

```python
registrator.create_task('detection_task_1')
```

#### stop_task(key_name: str) -> None
Устанавливает статус задачи в 'DONE'.

```python
registrator.stop_task('detection_task_1')
```

#### delete_task(key_name: str) -> None
Полностью удаляет задачу из каталога.

```python
registrator.delete_task('detection_task_1')
```

#### add_task(key_name: str, flag: bool) -> None
Добавляет статус к задаче и устанавливает состояние 'PENDING'.

```python
registrator.add_task('detection_task_1', True)   # Обнаружение
registrator.add_task('detection_task_1', False)  # Нет обнаружения
```

#### clear_key_task(key_name: str) -> None
Очищает все статусы задачи.

```python
registrator.clear_key_task('detection_task_1')
```

#### add_video_frame(key_name: str, frame: np.ndarray) -> None
Добавляет кадр к задаче через VideoWriter.

```python
frame = cv2.imread('frame.jpg')
registrator.add_video_frame('detection_task_1', frame)
```

#### register_task(key_name: str, message_event: dict) -> None
Завершает задачу, сохраняет видео и регистрирует событие в Redis.

**Процесс регистрации:**
1. Останавливает задачу (статус 'DONE')
2. Сохраняет видео и ключевой кадр
3. Устанавливает статус 'FINISHED' или 'ERROR'
4. Сохраняет данные в Redis Sorted Set 'events'

```python
message = {
    'model': 'YOLOModel',
    'result': {'boxes': [[100, 100, 200, 200]], 'scores': [0.95]},
    'frame_id': 'frame_001'
}
await registrator.register_task('detection_task_1', message)
```

## Примеры использования

### 1. Базовое использование

```python
import asyncio
import cv2
import numpy as np
from abs_src.abs_registrator import BasicEventRegistrator

async def main():
    # Создание регистратора
    registrator = BasicEventRegistrator()
    
    # Подключение к NATS
    await registrator.connect_nats()
    
    try:
        # Создание задачи
        task_name = 'motion_detection_1'
        registrator.create_task(task_name)
        
        # Добавление кадров
        for i in range(30):  # 30 кадров для 3-секундного видео
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            registrator.add_video_frame(task_name, frame)
            
            # Добавление статуса обнаружения
            detection_result = i % 5 == 0  # Обнаружение каждый 5-й кадр
            registrator.add_task(task_name, detection_result)
        
        # Получение наиболее частого статуса
        most_common_status = registrator.most_common(task_name)
        print(f"Наиболее частый статус: {most_common_status}")
        
        # Регистрация события
        event_message = {
            'event_type': 'motion_detected',
            'confidence': 0.85,
            'timestamp': time.time()
        }
        await registrator.register_task(task_name, event_message)
        
    finally:
        await registrator.close()

if __name__ == "__main__":
    asyncio.run(main())
```

### 2. Интеграция с системой видеонаблюдения

```python
import cv2
import asyncio
from abs_src.abs_registrator import BasicEventRegistrator

class SecuritySystem:
    def __init__(self):
        self.registrator = BasicEventRegistrator()
        self.active_tasks = {}
    
    async def start(self):
        await self.registrator.connect_nats()
    
    async def process_camera_feed(self, camera_id: str, video_path: str):
        """Обработка видеопотока с камеры"""
        cap = cv2.VideoCapture(video_path)
        task_name = f"camera_{camera_id}_event"
        
        # Создание задачи
        self.registrator.create_task(task_name)
        
        frame_count = 0
        detection_buffer = []
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Добавление кадра
                self.registrator.add_video_frame(task_name, frame)
                
                # Симуляция детекции (в реальности здесь был бы ML алгоритм)
                has_motion = self.detect_motion(frame)
                self.registrator.add_task(task_name, has_motion)
                detection_buffer.append(has_motion)
                
                frame_count += 1
                
                # Проверка на событие каждые 30 кадров
                if frame_count % 30 == 0:
                    if self.should_trigger_event(detection_buffer):
                        await self.trigger_security_event(task_name, camera_id)
                        detection_buffer.clear()
                        
                        # Создание новой задачи для следующего сегмента
                        task_name = f"camera_{camera_id}_event_{frame_count}"
                        self.registrator.create_task(task_name)
        
        finally:
            cap.release()
    
    def detect_motion(self, frame):
        """Простая детекция движения (заглушка)"""
        # В реальности здесь был бы алгоритм детекции
        return np.random.random() > 0.7
    
    def should_trigger_event(self, detections):
        """Определяет, нужно ли создавать событие"""
        return sum(detections) > len(detections) * 0.3
    
    async def trigger_security_event(self, task_name: str, camera_id: str):
        """Создание события безопасности"""
        event_data = {
            'event_type': 'security_breach',
            'camera_id': camera_id,
            'severity': 'high',
            'timestamp': time.time(),
            'description': 'Motion detected in restricted area'
        }
        
        await self.registrator.register_task(task_name, event_data)
        print(f"Security event triggered for camera {camera_id}")

# Использование
async def main():
    security = SecuritySystem()
    await security.start()
    
    # Обработка нескольких камер параллельно
    tasks = [
        security.process_camera_feed("001", "camera1.mp4"),
        security.process_camera_feed("002", "camera2.mp4"),
    ]
    
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())
```

### 3. Кастомный регистратор событий

```python
from abs_src.abs_registrator import AbstractEventRegistrator
import json
import time

class CustomEventRegistrator(AbstractEventRegistrator):
    def __init__(self):
        super().__init__()
        self.custom_storage = {}
    
    def create_task(self, key_name: str) -> None:
        self.task_catalog[key_name]['task'] = 'INITIALIZED'
        self.custom_storage[key_name] = {
            'created_at': time.time(),
            'frames_count': 0
        }
    
    def stop_task(self, key_name: str) -> None:
        self.task_catalog[key_name]['task'] = 'COMPLETED'
        if key_name in self.custom_storage:
            self.custom_storage[key_name]['completed_at'] = time.time()
    
    def delete_task(self, key_name: str) -> None:
        if key_name in self.task_catalog:
            del self.task_catalog[key_name]
        if key_name in self.custom_storage:
            del self.custom_storage[key_name]
    
    def add_task(self, key_name: str, flag: bool) -> None:
        self.task_catalog[key_name]['status'].append(flag)
        self.task_catalog[key_name]['task'] = 'PROCESSING'
    
    def clear_key_task(self, key_name: str) -> None:
        self.task_catalog[key_name]['status'].clear()
    
    def add_video_frame(self, key_name: str, frame: np.ndarray) -> None:
        # Кастомная логика сохранения кадров
        if key_name in self.custom_storage:
            self.custom_storage[key_name]['frames_count'] += 1
    
    async def register_task(self, key_name: str, message: dict) -> None:
        self.stop_task(key_name)
        
        # Кастомная логика регистрации
        event_data = {
            'task_name': key_name,
            'message': message,
            'statistics': self.custom_storage.get(key_name, {}),
            'most_common_status': self.most_common(key_name)
        }
        
        # Сохранение в Redis с кастомным ключом
        await self.redis.lpush(
            f'custom_events:{key_name}', 
            json.dumps(event_data)
        )
        
        logger.info(f"Custom event registered: {key_name}")
```

## Структура данных в Redis

### Ключ: 'events' (Sorted Set)

Данные сохраняются как JSON с временной меткой в качестве score:

```json
{
    "task_name": {
        "message": {
            "model": "YOLOModel",
            "result": {"boxes": [], "scores": [], "class_ids": []},
            "frame_id": "frame_001"
        },
        "video_proofs": "/tmp/uuid/uuid.mp4",
        "image_proof_path": "/tmp/uuid/uuid.jpg"
    }
}
```

## Жизненный цикл задачи

1. **CREATED** - задача создана
2. **PENDING** - добавлены статусы
3. **DONE** - задача остановлена
4. **FINISHED** - видео сохранено, событие зарегистрировано
5. **ERROR** - произошла ошибка при сохранении

## Обработка ошибок

Система включает обработку следующих ошибок:
- Ошибки подключения к NATS/Redis
- Ошибки записи видео
- Ошибки сохранения в Redis
- Ошибки декодирования JSON сообщений

Все ошибки логируются через модуль `utils.logger`.

## Рекомендации по использованию

1. **Управление памятью**: Регулярно очищайте завершенные задачи для экономии памяти
2. **Размер буфера**: Ограничивайте количество кадров в буфере для предотвращения переполнения
3. **Обработка ошибок**: Реализуйте retry логику для критических операций
4. **Мониторинг**: Следите за размером Redis и дисковым пространством
5. **Производительность**: Используйте асинхронные операции для I/O операций

## Зависимости

Основные зависимости модуля:
- `numpy` - работа с массивами
- `cv2` - обработка изображений
- `av` - запись видео
- `nats-py` - NATS клиент  
- `redis` - Redis клиент
- `uuid` - генерация уникальных идентификаторов