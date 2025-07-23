# Архитектура системы обработки видео

```mermaid
graph TB
    %% Входные данные
    VIDEO[📹 Видео источник<br/>RTSP/файл] 
    
    %% Основные компоненты
    READER[🎬 VideoReader<br/>AVReader/OpencvVideoReader]
    SEAWEED[🗄️ SeaweedFS<br/>Распределенное хранилище]
    NATS[📡 NATS<br/>Message Broker]
    REDIS[🔴 Redis<br/>Service Discovery]
    ROUTER[🔀 RouterManager<br/>Маршрутизация сообщений]
    
    %% Модели инференса
    DETECTION[🎯 DetectionInference<br/>Детекция объектов]
    CLASSIFICATION[🏷️ ClassificationInference<br/>Классификация]
    
    %% Результаты
    RESULTS[📊 Результаты<br/>inference.results]
    LOGS[📝 Логи<br/>system.log]
    
    %% Поток данных
    VIDEO --> READER
    READER -->|Кадры| SEAWEED
    READER -->|Метаданные кадра| NATS
    
    %% Регистрация сервисов
    DETECTION -.->|Регистрация| REDIS
    CLASSIFICATION -.->|Регистрация| REDIS
    
    %% Маршрутизация
    NATS -->|topic_stream| ROUTER
    ROUTER -->|Получает список моделей| REDIS
    ROUTER -->|Отправляет задачи| DETECTION
    ROUTER -->|Отправляет задачи| CLASSIFICATION
    
    %% Обработка
    DETECTION -->|Скачивает изображение| SEAWEED
    CLASSIFICATION -->|Скачивает изображение| SEAWEED
    
    %% Результаты
    DETECTION --> RESULTS
    CLASSIFICATION --> RESULTS
    
    %% Логирование
    READER -.-> LOGS
    ROUTER -.-> LOGS
    DETECTION -.-> LOGS
    CLASSIFICATION -.-> LOGS
    
    %% Стили
    classDef storage fill:#e1f5fe
    classDef processing fill:#f3e5f5
    classDef messaging fill:#e8f5e8
    classDef models fill:#fff3e0
    
    class SEAWEED,REDIS storage
    class READER,ROUTER processing
    class NATS,RESULTS messaging
    class DETECTION,CLASSIFICATION models
```

## Описание компонентов

### 1. **VideoReader** (Чтение видео)
- **AVReader**: Использует библиотеку `av` для чтения RTSP/видеофайлов
- **OpencvVideoReader**: Использует OpenCV для чтения видео
- Извлекает кадры с настраиваемым пропуском
- Кодирует кадры в JPEG и загружает в SeaweedFS
- Отправляет метаданные кадров в NATS

### 2. **SeaweedFS** (Хранилище)
- Распределенная файловая система
- Хранит кадры видео с TTL
- Компоненты: Master, Volume, Filer
- Предоставляет URL для доступа к кадрам

### 3. **NATS** (Message Broker)
- Асинхронная передача сообщений
- Топики:
  - `topic_stream`: Метаданные новых кадров
  - `{model_name}`: Задачи для конкретных моделей
  - `inference.results`: Результаты обработки

### 4. **Redis** (Service Discovery)
- Хранит список активных моделей в ключе `routing_to_models`
- TTL для автоматического удаления неактивных сервисов
- Используется RouterManager для маршрутизации

### 5. **RouterManager** (Маршрутизация)
- Получает метаданные кадров из NATS
- Динамически отслеживает доступные модели через Redis
- Распределяет задачи между активными моделями
- Реализует паттерн Publisher-Subscriber

### 6. **Модели инференса**
- **DetectionInference**: Детекция объектов (YOLO-подобная)
- **ClassificationInference**: Классификация изображений
- Автоматическая регистрация в Redis
- Скачивание изображений из SeaweedFS
- Пайплайн: preprocess → inference → postprocess

## Поток данных

1. **Захват кадров**: VideoReader читает видео и извлекает кадры
2. **Сохранение**: Кадры сохраняются в SeaweedFS с уникальными ID
3. **Уведомление**: Метаданные кадра отправляются в NATS топик
4. **Маршрутизация**: RouterManager получает сообщение и распределяет между моделями
5. **Обработка**: Модели скачивают изображения и выполняют инференс
6. **Результаты**: Результаты публикуются в топик `inference.results`

## Масштабируемость

- **Горизонтальное масштабирование**: Можно запускать несколько экземпляров каждой модели
- **Автоматическое обнаружение**: Новые модели автоматически регистрируются
- **Отказоустойчивость**: TTL в Redis обеспечивает удаление мертвых сервисов
- **Асинхронность**: Все компоненты работают асинхронно

## Конфигурация

Система настраивается через переменные окружения:
- `video_source`: Источник видео
- `nats_host`: NATS сервер
- `redis_host/redis_port`: Redis подключение
- `master_url/volume_url`: SeaweedFS endpoints
- `topic_stream`: Топик для кадров