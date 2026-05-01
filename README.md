# 🎥 Система обработки видео в реальном времени

Распределенная система для обработки видеопотоков с использованием машинного обучения. Система состоит из микросервисов для чтения видео, хранения кадров, маршрутизации задач и выполнения инференса моделей ИИ.

Документация по сервису чтения кадров, очереди, `target_fps`, `queue_policy` и масштабированию: [docs/READER_PIPELINE.md](docs/READER_PIPELINE.md).

## 🏗️ Архитектура

Система построена на микросервисной архитектуре с использованием:
- **SeaweedFS** - распределенное хранилище для кадров
- **NATS** - брокер сообщений для координации
- **Redis** - service discovery и кэширование
- **Docker** - контейнеризация сервисов

## 📋 Требования

- Python 3.8+
- Docker и Docker Compose
- 4GB+ RAM
- Видеофайл или RTSP поток

## 🚀 Быстрый запуск

### 1. Клонирование и подготовка

```bash
git clone <repository-url>
cd video-processing-system
```

### 2. Установка зависимостей

```bash
pip install -r requirements.txt
```

### 3. Запуск инфраструктуры

```bash
# Запуск SeaweedFS, NATS и Redis
docker-compose up -d

# Проверка статуса сервисов
docker-compose ps
```

### 4. Настройка переменных окружения

Отредактируйте файл `.env`:

```bash
# Источник видео (файл или RTSP поток)
video_source=./video/test_maneken.mp4
video_skip_frames=0
video_index=0

# Тип ридера (av или opencv)
reader_type=av

# SeaweedFS настройки
ttl_bucket=5m
bucket_name=video-stream
master_url=http://localhost:9333
volume_url=http://localhost:8888

# NATS настройки
nats_host=nats://localhost:4222
topic_stream=frames-stream

# Redis настройки
routing_ttl=10
```

### 5. Запуск компонентов системы

Откройте 3 терминала и запустите каждый компонент:

**Терминал 1 - Чтение видео:**
```bash
python src/data_reader.py
```

**Терминал 2 - Маршрутизатор:**
```bash
python src/router.py
```

**Терминал 3 - Модели инференса:**
```bash
python src/inference.py
```

## 📊 Мониторинг

### Web интерфейсы

- **SeaweedFS Master**: http://localhost:9333
- **SeaweedFS Filer**: http://localhost:8888
- **SeaweedFS Volume**: http://localhost:8080

### Проверка работы

```bash
# Проверка NATS подключения
docker logs <container_name>_nats_1

# Проверка Redis
redis-cli ping

# Просмотр логов обработки
tail -f logs/processor.log
```

## 🔧 Конфигурация

### Настройка источника видео

**Для видеофайла:**
```bash
video_source=./video/your_video.mp4
reader_type=opencv
```

**Для RTSP потока:**
```bash
video_source=rtsp://camera_ip:554/stream
reader_type=av
```

### Настройка моделей

Система поддерживает два типа моделей:
- `DetectionInference` - детекция объектов
- `ClassificationInference` - классификация изображений

Для добавления новой модели:

1. Наследуйтесь от `ModelInference`
2. Реализуйте методы `preprocess`, `inference`, `postprocess`
3. Добавьте в `src/inference.py`

### Настройка TTL

```bash
# Время жизни кадров в SeaweedFS
ttl_bucket=5m

# Время жизни сервисов в Redis
routing_ttl=10
```

## 🐛 Устранение неполадок

### Проблемы с подключением

```bash
# Проверка портов
netstat -tulpn | grep -E "(4222|6379|9333|8888)"

# Перезапуск сервисов
docker-compose restart
```

### Проблемы с видео

```bash
# Проверка кодеков (для av reader)
ffmpeg -i your_video.mp4

# Использование opencv reader
reader_type=opencv
```

### Проблемы с памятью

```bash
# Увеличение skip_frames для снижения нагрузки
video_skip_frames=5

# Уменьшение TTL
ttl_bucket=1m
```

## 📈 Масштабирование

### Горизонтальное масштабирование

```bash
# Запуск дополнительных экземпляров моделей
python src/inference.py &
python src/inference.py &
```

### Добавление Volume серверов

```yaml
# В docker-compose.yaml
volume2:
  image: chrislusf/seaweedfs:3.93
  command: "volume -mserver=master:9333 -port=8081 -dir=/data2"
  volumes:
    - ./volume2_data:/data2
  ports:
    - "8081:8081"
```

## 🔒 Безопасность

### Рекомендации

- Используйте переменные окружения для чувствительных данных
- Настройте файрвол для ограничения доступа к портам
- Регулярно обновляйте Docker образы
- Мониторьте использование ресурсов

## 📝 Логирование

Логи сохраняются в:
- `logs/processor.log` - основные логи системы
- Docker логи: `docker-compose logs -f`

Уровни логирования настраиваются в `utils/logger.py`

## 🤝 Разработка

### Структура проекта

```
├── src/                    # Основные компоненты
│   ├── video_reader.py     # Чтение видео
│   ├── router.py          # Маршрутизация
│   └── inference.py       # Модели ИИ
├── abs_src/               # Абстрактные классы
├── utils/                 # Утилиты
├── logs/                  # Логи
├── video/                 # Тестовые видео
└── docker-compose.yaml    # Инфраструктура
```

### Добавление новых компонентов

1. Создайте класс, наследующийся от соответствующего абстрактного класса
2. Реализуйте необходимые методы
3. Добавьте конфигурацию в `.env`
4. Обновите документацию

## 📞 Поддержка

При возникновении проблем:

1. Проверьте логи: `tail -f logs/processor.log`
2. Убедитесь, что все сервисы запущены: `docker-compose ps`
3. Проверьте переменные окружения в `.env`
4. Создайте issue с описанием проблемы и логами

## 📄 Лицензия

MIT License - см. файл LICENSE для деталей.