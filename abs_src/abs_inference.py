import asyncio
import os

import aiohttp
import io
import json
import nats

from abc import ABC, abstractmethod
from typing import Any, Dict
from PIL import Image
from redis.asyncio import Redis
from utils.logger import logger


class ModelInference(ABC):
    def __init__(self):
        self.model_name = self.get_name()
        self.nats_conn = None
        self.nats_host = os.getenv("nats_host", "nats://localhost:4222").split(",")
        self.redis = Redis(host=os.getenv('redis_host', 'localhost'), port=int(os.getenv('redis_port', 6379)))

    @classmethod
    def get_name(cls):
        model_name = cls.__name__
        return model_name

    async def register_service(self):
        while True:
            try:
                await self.redis.sadd("routing_to_models", self.model_name)
                await self.redis.expire("routing_to_models", int(os.getenv('routing_ttl', 10)))
                await asyncio.sleep(int(os.getenv('routing_ttl', 10)))
            except Exception as e:
                logger.error(f"Ошибка регистрации подписчика {self.model_name} на публикации чтения кадров: {e}")
                await asyncio.sleep(1)

    @abstractmethod
    async def preprocess(self, image: Image.Image) -> Any:
        """Предобработка изображения"""
        pass

    @abstractmethod
    async def inference(self, preprocessed_data: Any) -> Any:
        """Выполнение инференса модели"""
        pass

    @abstractmethod
    async def postprocess(self, inference_output: Any) -> Dict[str, Any]:
        """Постобработка результатов"""
        pass

    async def run_pipeline(self, image: Image.Image) -> Dict[str, Any]:
        """Запуск полного пайплайна обработки"""
        preprocessed = await self.preprocess(image)
        inference_result = await self.inference(preprocessed)
        return await self.postprocess(inference_result)

    @staticmethod
    async def download_image(data: dict) -> Image.Image:
        """Скачивание изображения по URL из словаря данных"""
        try:
            if not data.get('seaweed_url'):
                raise ValueError("Нету seaweed_url в теле сообщений для скачивания изображения по ссылке")

            async with aiohttp.ClientSession() as session:
                async with session.get(data['seaweed_url']) as response:
                    response.raise_for_status()
                    image_data = await response.read()
                    return Image.open(io.BytesIO(image_data))

        except (aiohttp.ClientError, OSError, Image.DecompressionBombError) as e:
            logger.error(f"Ошибка скачивания изображения: {str(e)}")
            raise

    async def message_handler(self, msg):
        """Обработчик сообщений из NATS"""
        try:
            raw_data = msg.data.decode()
            data = json.loads(raw_data)

            if not data.get('seaweed_url'):
                raise ValueError("Нету seaweed_url в теле сообщений для скачивания изображения по ссылке")

            image = await self.download_image(data)
            result = await self.run_pipeline(image)
            logger.info(f"[{self.model_name}] {result}")

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
            logger.error(f"Ошибка декодирования сообщения в JSON: {msg.data.decode()}")
        except Exception as e:
            logger.error(f"Ошибка обработки в классе [{self.model_name}]: {str(e)}")

    async def connect_nats(self):
        """Подключение к NATS и подписка на тему"""
        self.nats_conn = await nats.connect(self.nats_host)
        await self.nats_conn.subscribe(self.model_name, cb=self.message_handler)
        logger.info(f"Подписка класса [{self.model_name}] на топик NATS: {self.model_name} успешно")

        await asyncio.create_task(self.register_service())
