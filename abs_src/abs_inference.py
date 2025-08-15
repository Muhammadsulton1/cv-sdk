import os
import json
import asyncio
import numpy as np
import nats

from abc import ABC, abstractmethod
from typing import Any, Dict
from polygraphy.backend.trt import CreateConfig, engine_from_network, NetworkFromOnnxPath, save_engine, \
    EngineFromBytes, Profile
from polygraphy.backend.common import BytesFromPath
from pydantic import ValidationError
from redis.asyncio import Redis

from src.data_scheme import InferenceOutputSchema
from src.s3_storage import SeaweedFSManager
from src.singeleton.yaml_reader import YamlReader
from utils.decorators import measure_latency_async, measure_latency_sync
from utils.logger import logger


class AbstractConverter(ABC):
    """
        Абстрактный базовый класс для конвертации моделей в оптимизированные форматы.

        Атрибуты:
            model_path (str): Путь к исходной модели
            converted_path (str): Путь для сохранения конвертированной модели
            input_name (str): Имя входного тензора модели
            output_name (str): Имя выходного тензора модели
            engine: Загруженный движок модели
            loaded_engine (bool): Флаг загрузки движка
    """

    def __init__(self, model_path: str, converted_path: str, input_name: str, output_name: str) -> None:
        """
            Инициализация конвертера.

            Args:
                model_path: Путь к исходной модели
                converted_path: Путь для сохранения конвертированной модели
                input_name: Имя входного тензора
                output_name: Имя выходного тензора
        """
        self.model_path = model_path
        self.converted_path = converted_path
        self.input_name = input_name
        self.output_name = output_name
        self.engine = None
        self.loaded_engine = False

    def load_or_convert(self) -> None:
        """
        Проверяет наличие конвертированной модели либо загружает её,
        иди выполняет конвертацию.
        """
        if os.path.exists(self.converted_path):
            self.load_engine()
        else:
            self.convert_model()

    @abstractmethod
    def convert_model(self) -> None:
        """
        Абстрактный метод конвертации модели (реализуется в наследниках).
        """
        pass

    @abstractmethod
    def load_engine(self) -> None:
        """
        Абстрактный метод загрузки модели (реализуется в наследниках).
        """
        pass


class TensorRTConverter(AbstractConverter):
    """
    Реализация конвертера моделей в формат TensorRT.

    Наследует:
        AbstractConverter
    """

    def __init__(self, model_path: str, converted_path: str, input_name: str, output_name: str) -> None:
        """
           Инициализация конвертера TensorRT.

           Args:
               model_path: Путь к ONNX модели
               converted_path: Путь для сохранения .engine файла
               input_name: Имя входного тензора
               output_name: Имя выходного тензора
       """
        super().__init__(model_path, converted_path, input_name, output_name)
        self.load_or_convert()

    def convert_model(self) -> None:
        """
            Конвертирует ONNX модель в TensorRT-движок.

            Raises:
                Exception: При ошибках конвертации
        """
        try:
            model = NetworkFromOnnxPath(self.model_path)

            if os.getenv("ONNX_DYNAMIC_AXIS"):
                self._compile_dynamic_engine(model)
            else:
                self._compile_static_engine(model)

            logger.info(f"TensorRT engine compiled: {self.converted_path}")
        except Exception as e:
            logger.error(f"Model conversion failed: {e}")
            raise

    def load_engine(self) -> None:
        """
        Загружает предварительно конвертированный TensorRT-движок.
        """
        self.engine = EngineFromBytes(BytesFromPath(self.converted_path))
        self.loaded_engine = True
        logger.info(f"Engine loaded: {self.converted_path}")

    def _compile_static_engine(self, model: Any) -> None:
        """
            Компилирует модель со статическими размерами входов.

            Args:
                model: Объект модели Polygraphy
        """
        config = CreateConfig()
        self.engine = engine_from_network(model, config=config)
        save_engine(self.engine, path=self.converted_path)

    def _compile_dynamic_engine(self, model: Any) -> None:
        """
        Компилирует модель с динамическими размерами входов.

        Args:
            model: Объект модели Polygraphy
        """
        profile = Profile()
        profile.add(
            name=self.input_name,
            min=(1, 3, 224, 224),
            max=(32, 3, 640, 640),
            opt=(1, 3, 640, 640)
        )
        config = CreateConfig(profiles=[profile], fp16=True)
        self.engine = engine_from_network(model, config=config)
        save_engine(self.engine, path=self.converted_path)


class BaseInferenceModel(ABC):
    """
        Базовый класс для моделей инференса с поддержкой NATS и Redis.

        Атрибуты:
            NATS_HOST: Адреса NATS-серверов
            REDIS_HOST: Хост Redis
            REDIS_PORT: Порт Redis
            ROUTING_TTL: TTL регистрации в Redis (сек)
            model_name: Имя модели (автоматически из класса)
    """
    NATS_HOST = os.getenv("nats_host", "nats://localhost:4222").split(",")
    REDIS_HOST = os.getenv('redis_host', 'localhost')
    REDIS_PORT = int(os.getenv('redis_port', 6379))
    REGISTER_TTL = int(os.getenv('routing_ttl', 10))

    def __init__(self) -> None:
        """
        Инициализация базовой модели.
        """
        self.model_name = self.get_name()
        self.nats_conn = None
        self.redis = Redis(host=self.REDIS_HOST, port=self.REDIS_PORT)
        self.s3_client = SeaweedFSManager()

        self.setup_config = YamlReader()
        self.in_channel = self.setup_config.get('InferenceModel')['in_channel']
        self.out_channel = self.setup_config.get('InferenceModel')['out_channel']

        self._service_task = None

    async def close(self) -> None:
        logger.info(f'Инициализирована закрытие всех соединений для {self.model_name}')
        if self.nats_conn:
            await self.nats_conn.drain()
            await self.nats_conn.close()
            self.nats_conn = None

        if self.redis:
            await self.redis.aclose()
            logger.info("Redis соединение закрыто")
            self.redis = None

        if self.s3_client:
            await self.s3_client.__aexit__(None, None, None)
            logger.info("S3 клиент закрыт")
            self.s3_client = None

        logger.info(f'Освобождены все сетевые ресурсы для класса {self.model_name}')

    @classmethod
    def get_name(cls) -> str:
        """
            Возвращает имя модели (по имени класса).

            Returns:
                Имя модели
        """
        return cls.__name__

    @measure_latency_async()
    async def register_service(self) -> None:
        """
        Регистрирует модель в Redis для маршрутизации.
        Обновляет TTL записи каждые ROUTING_TTL секунд.
        """
        while True:
            try:
                await self.redis.sadd("routing_to_models", self.model_name)
                await self.redis.expire("routing_to_models", self.REGISTER_TTL)
                await asyncio.sleep(self.REGISTER_TTL)
            except Exception as e:
                logger.info(f'Ошибка регистрации сервиса: {e}', )
                await asyncio.sleep(1)

    @abstractmethod
    def preprocess(self, image: Any, *args, **kwargs) -> Any:
        """
            Абстрактный метод пред обработки входных данных.

            Args:
                image: Входное изображение
                *args: Дополнительные параметры
                **kwargs: Дополнительные параметры

            Returns:
                Подготовленные данные для инференса
        """
        pass

    @abstractmethod
    def postprocess(self, inference_output: Any, *args, **kwargs) -> Dict[str, Any]:
        """
            Абстрактный метод постобработки результатов.

            Args:
                inference_output: Результат работы модели
                *args: Дополнительные параметры
                **kwargs: Дополнительные параметры

            Returns:
                Словарь с обработанными результатами
            """
        pass

    @abstractmethod
    def inference(self, input_data: np.ndarray) -> Any:
        """
            Абстрактный метод выполнения инференса.

            Args:
                input_data: Подготовленные входные данные

            Returns:
                Результат работы модели
        """
        pass

    @measure_latency_async()
    async def message_handler(self, msg) -> None:
        """
            Обработчик входящих сообщений из NATS.

            Поток обработки:
            1. Декодирование JSON сообщения
            2. Проверка наличия seaweed_url
            3. Загрузка изображения
            4. Запуск пайплайна инференса
            5. Отправка результатов в NATS

            Args:
                msg: Входящее сообщение NATS
        """
        try:
            data = json.loads(msg.data.decode())

            if not isinstance(data, dict):
                raise ValueError(f"Invalid message format: expected JSON object, got {type(data).__name__}")

            if not data.get('seaweed_url'):
                logger.error(f"Missing 'seaweed_url' in message")
                raise ValueError("Missing 'seaweed_url' in message")

            image = await self.s3_client.download_object(data['seaweed_url'])
            result = self.run_inference(image)

            parsed_result = InferenceOutputSchema(**result)

            if parsed_result:
                await self.nats_conn.publish(
                    self.out_channel,
                    json.dumps({
                        "model": self.model_name,
                        "result": result,
                        "frame_id": data.get('frame_id', 'unknown'),
                        "frame_url": data.get('seaweed_url')
                    }).encode()
                )

        except json.JSONDecodeError:
            logger.error(f"Ошибка декодирования JSON от ответа ROUTER: {msg.data.decode()}")
        except ValidationError as e:
            logger.error(
                f"Ошибка валидации данных, конечный результат не соответствует InferenceDataSchema, Ошибка: {e}")
        except Exception as e:
            logger.error(f"Processing error [{self.model_name}]: {str(e)}")

    async def connect_nats(self):
        """
        Подключается к NATS и подписывается на топик с именем модели.
        Запускает фоновую задачу регистрации в Redis.
        """
        self.nats_conn = await nats.connect(self.NATS_HOST)
        await self.s3_client.__aenter__()
        await self.nats_conn.subscribe(f'{self.model_name}_{self.in_channel}', cb=self.message_handler)
        logger.info(f"Подписка класса [{self.model_name}_{self.in_channel}] на топик NATS: {self.model_name} успешно")
        await asyncio.create_task(self.register_service())

    @measure_latency_sync()
    def run_inference(self, image: Any) -> Dict[str, Any]:
        """
            Полный пайплайн обработки изображения:
            1. Пред обработка
            2. Инференс
            3. Постобработка

            Args:
                image: Входное изображение

            Returns:
                Результаты обработки
        """
        preprocessed = self.preprocess(image)
        inference_result = self.inference(preprocessed)
        return self.postprocess(inference_result)
