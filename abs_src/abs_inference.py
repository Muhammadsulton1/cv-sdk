import os
import json
import asyncio
import threading

import numpy as np

from abc import ABC, abstractmethod
from typing import Any, Dict
from polygraphy.backend.trt import CreateConfig, engine_from_network, NetworkFromOnnxPath, save_engine, \
    EngineFromBytes, Profile
from polygraphy.backend.common import BytesFromPath
from pydantic import ValidationError

from src.data_scheme import InferenceOutputSchema
from src.s3_storage import SeaweedFSManager
from src.singeleton.connection_singeleton import RedisClient, NatsClient, S3Client
from src.singeleton.yaml_reader import YamlReader
from utils.decorators import measure_latency_async, measure_latency_sync
from utils.logger import logger


# TODO: Поправить получения кадра от инференса и его правильное скачивание, а также добавить соединенение с NATS И РЕДИС ЧЕРЕЗ Сингелетон

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
    """
    REGISTER_TTL = int(os.getenv('routing_ttl', 10))

    def __init__(self) -> None:
        """
        Инициализация базовой модели.
        """
        self.model_name = self.get_name()

        self.s3_cli = None
        self.nats_cli = None
        self.redis_cli = None

        self.setup_config = YamlReader()
        self.in_channel = self.setup_config.get(os.getenv('SERVICE_NAME')).get('in_channel')
        self.out_channel = self.setup_config.get(os.getenv('SERVICE_NAME')).get('out_channel')

        self._service_task = None

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
                await self.redis_cli.sadd("models_routing", self.model_name)
                await self.redis_cli.expire("models_routing", self.REGISTER_TTL)
                await asyncio.sleep(self.REGISTER_TTL)
            except Exception as e:
                logger.info(f'Ошибка регистрации сервиса: {e}', )
                await asyncio.sleep(1)

    @abstractmethod
    async def preprocess(self, image: Any, *args, **kwargs) -> Any:
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
                raise ValueError(
                    f"Не валидный формат сообщений: ожидался JSON объект, был передан {type(data).__name__}")

            if not data.get('file_url'):
                logger.error(f"Отсутствует ссылка на кадр в file_url")
                raise ValueError("Отсутствует ссылка на кадр в file_url")

            image = await self.s3_cli.download_object(data)
            result = await self.get_inference_results(image)

            parsed_result = InferenceOutputSchema(**result)

            if parsed_result:
                await self.nats_cli.publish(
                    self.out_channel,
                    json.dumps({
                        "model": self.model_name,
                        "result": result,
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

    async def process(self) -> None:
        self.s3_cli = await SeaweedFSManager().__aenter__()
        self.nats_cli = await NatsClient().connect()
        self.redis_cli = await RedisClient().connect()

        await self.nats_cli.subscribe(f'{self.model_name}_{self.in_channel}', cb=self.message_handler)
        logger.info(f"Подписка класса [{self.model_name}_{self.in_channel}] на топик NATS: {self.model_name} успешно")
        await asyncio.create_task(self.register_service())

    @measure_latency_async()
    async def get_inference_results(self, image: np.ndarray) -> Dict[str, Any]:
        """
            Полный пайплайн обработки изображения:
            1. Пред обработка (в отдельном потоке)
            2. Инференс
            3. Постобработка

            Args:
                image: Входное изображение

            Returns:
                Результаты обработки
        """
        preprocessed_data = await asyncio.to_thread(self.preprocess, image)

        image_processed = await preprocessed_data
        inference_result = self.inference(image_processed)
        return self.postprocess(inference_result)
