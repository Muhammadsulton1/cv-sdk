import os
import json
import asyncio
import aiohttp
import io
from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np
import nats
from polygraphy.backend.trt import CreateConfig, engine_from_network, NetworkFromOnnxPath, save_engine, \
    EngineFromBytes, Profile
from polygraphy.backend.common import BytesFromPath
from PIL import Image
from pydantic import ValidationError
from redis.asyncio import Redis

from src.data_scheme import InferenceOutputSchema
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
        config = CreateConfig(profiles=[profile])
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
    ROUTING_TTL = int(os.getenv('routing_ttl', 10))

    def __init__(self) -> None:
        """
        Инициализация базовой модели.
        """
        self.model_name = self.get_name()
        self.nats_conn = None
        self.redis = Redis(host=self.REDIS_HOST, port=self.REDIS_PORT)

    @classmethod
    def get_name(cls) -> str:
        """
            Возвращает имя модели (по имени класса).

            Returns:
                Имя модели
        """
        return cls.__name__

    async def register_service(self) -> None:
        """
        Регистрирует модель в Redis для маршрутизации.
        Обновляет TTL записи каждые ROUTING_TTL секунд.
        """
        while True:
            try:
                await self.redis.sadd("routing_to_models", self.model_name)
                await self.redis.expire("routing_to_models", self.ROUTING_TTL)
                await asyncio.sleep(self.ROUTING_TTL)
            except Exception as e:
                logger.info(f'Ошибка регистрации сервиса: {e}',)
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

    @staticmethod
    async def download_image(url: str) -> Image.Image:
        """
            Скачивает изображение по URL.

            Args:
                url: URL изображения

            Returns:
                Объект PIL.Image

            Raises:
                aiohttp.ClientError: Ошибка сетевого запроса
                OSError: Ошибка обработки изображения
            """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    response.raise_for_status()
                    image_data = await response.read()
                    return Image.open(io.BytesIO(image_data))
        except (aiohttp.ClientError, OSError) as err:
            logger.error(f"Image download failed: {err}")
            raise

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

            if not data.get('seaweed_url'):
                logger.error(f"Missing 'seaweed_url' in message")
                raise ValueError("Missing 'seaweed_url' in message")

            image = await self.download_image(data['seaweed_url'])
            result = self.run_inference(image)

            parsed_result = InferenceOutputSchema(**result)

            if parsed_result:
                response_topic = "inference.results"
                await self.nats_conn.publish(
                    response_topic,
                    json.dumps({
                        "model": self.model_name,
                        "result": result,
                        "frame_id": data.get('frame_id', 'unknown')
                    }).encode()
                )

        except json.JSONDecodeError:
            logger.error(f"JSON decode error: {msg.data.decode()}")
        except ValidationError as e:
            logger.error(f"Ошибка валидации данных, конечный результат не соответствует InferenceDataSchema, Ошибка: {e}")
        except Exception as e:
            logger.error(f"Processing error [{self.model_name}]: {str(e)}")

    async def connect_nats(self):
        """
        Подключается к NATS и подписывается на топик с именем модели.
        Запускает фоновую задачу регистрации в Redis.
        """
        self.nats_conn = await nats.connect(self.NATS_HOST)
        await self.nats_conn.subscribe(self.model_name, cb=self.message_handler)
        logger.info(f"Подписка класса [{self.model_name}] на топик NATS: {self.model_name} успешно")
        await asyncio.create_task(self.register_service())

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
