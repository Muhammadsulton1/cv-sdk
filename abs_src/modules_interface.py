import aiohttp
import io
import json
import nats

from abc import ABC, abstractmethod
from typing import Any, Dict
from PIL import Image


class ModelInference(ABC):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.nats_conn = None

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
    async def download_image(url: str) -> Image.Image:
        """Скачивание изображения по URL"""
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                response.raise_for_status()
                image_data = await response.read()
        return Image.open(io.BytesIO(image_data))

    async def message_handler(self, msg):
        """Обработчик сообщений из NATS"""
        try:
            data = msg.data.decode()
            if not data:
                raise ValueError("Missing image_url in message")

            image = await self.download_image(data)

            result = await self.run_pipeline(image)

            print(self.model_name, result)

            response_topic = "inference.results"
            await self.nats_conn.publish(
                response_topic,
                json.dumps({
                    "model": self.model_name,
                    "result": result,
                }).encode()
            )

        except Exception as e:
            print(f"[{self.model_name}] Error processing message: {str(e)}")

    async def connect_nats(self, servers: str, topic: str):
        """Подключение к NATS и подписка на тему"""
        self.nats_conn = await nats.connect(servers)
        await self.nats_conn.subscribe(topic, cb=self.message_handler)
        print(f"[{self.model_name}] Subscribed to NATS topic: {topic}")


class FileRouter(ABC):
    @abstractmethod
    async def upload_file(self, file_data: bytes, **kwargs) -> str:
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        pass