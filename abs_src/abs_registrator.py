import os
import time
import json
import av
import asyncio
import cv2
import nats
import numpy as np

from abc import ABC, abstractmethod
from typing import Dict
from uuid import uuid4
from collections import defaultdict, Counter
from nats.aio.errors import ErrTimeout, ErrConnectionClosed, ErrNoServers, ErrBadSubscription
from redis.asyncio import Redis
from utils.logger import logger


class AbstractEventRegistrator(ABC):
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
        self._last_inference_data = None

    @abstractmethod
    def create_task(self, key_name: str) -> None:
        pass

    @abstractmethod
    def stop_task(self, key_name: str) -> None:
        pass

    @abstractmethod
    def delete_task(self, key_name: str) -> None:
        pass

    @abstractmethod
    def add_task(self, key_name: str, flag: bool) -> None:
        pass

    @abstractmethod
    def clear_key_task(self, key_name: str) -> None:
        pass

    @abstractmethod
    def add_video_frame(self, key_name: str, frame: np.ndarray) -> None:
        pass

    @abstractmethod
    async def register_task(self, key_name: str, message: dict) -> None:
        pass

    def most_common(self, key_name: str) -> bool | str:
        task_list = self.task_catalog[key_name]['status']
        if not task_list:
            return False
        return Counter(task_list).most_common(1)[0][0]

    async def message_handler(self, msg):
        """Асинхронный обработчик сообщений NATS."""
        data = json.loads(msg.data.decode())
        logger.debug(f'получено сообщение из топика inference.results {data}')
        self._last_inference_data = data
        return data

    async def connect_nats(self) -> None:
        """Устанавливает соединение с NATS и подписывается."""
        if self._nats_connected:
            return

        try:
            self.nats_conn = await nats.connect(
                servers=os.getenv("NATS_HOST", "nats://localhost:4222").split(","),
                max_reconnect_attempts=5
            )
            await self.nats_conn.subscribe('inference.results', cb=self.message_handler)
            self._nats_connected = True
            logger.info("Подписка на топик NATS 'inference.results' успешно установлена")
        except (ErrTimeout, ErrConnectionClosed, ErrNoServers, ErrBadSubscription) as err:
            logger.error(f"Ошибка подключения и подписка на inference.results: {err}")
            await self.close()

    @property
    def inference_data(self) -> Dict:
        """Возвращает результат из последнего полученного сообщения."""
        if self._last_inference_data is None:
            return {}
        return self._last_inference_data.get('result')

    async def close(self) -> None:
        """Корректно закрывает соединения."""
        try:
            if self.nats_conn and not self.nats_conn.is_closed:
                await self.nats_conn.drain()
                self.nats_conn = None
        except Exception as e:
            logger.error(f"Error draining NATS connection: {e}")
        finally:
            if self.redis:
                await self.redis.aclose()
                self.redis = None


class VideoWriter:
    def __init__(self):
        self.tmp_path = os.getenv("TMP_PATH")
        os.makedirs(self.tmp_path, exist_ok=True)
        self.video_frame_store = defaultdict(self._create_video_entry)

    def _create_video_entry(self):
        """Создает новую запись для хранения видео-данных"""
        event_uuid = str(uuid4())
        event_dir = f'{self.tmp_path}/{str(event_uuid)}'
        os.makedirs(event_dir, exist_ok=True)

        return {
            'frames': [],
            'event_proof_name': event_uuid,
            'frame_proof_path': f'{event_dir}/{event_uuid}.jpg',
            'video_proof_path': f'{event_dir}/{event_uuid}.mp4'
        }

    def add_frame(self, key: str, frame: np.ndarray) -> None:
        """Добавляет кадр в буфер для указанного ключа."""
        self.video_frame_store[key]['frames'].append(frame.copy())

    def save_keyframe(self, key: str, frame: np.ndarray) -> None:
        """Сохраняет ключевой кадр как изображение."""
        filename = self.video_frame_store[key]['frame_proof_path']
        cv2.imwrite(filename, frame)
        logger.debug(f'Saved keyframe to {filename}')

    def save_video(self, key: str) -> None:
        """Сохраняет видео по всем собранным кадрам."""
        frames = self.video_frame_store[key]['frames']
        if not frames:
            logger.warning(f"No frames to save for {key}")
            return

        self.save_keyframe(key, frames[0])

        height, width = frames[0].shape[:2]
        video_filename = self.video_frame_store[key]['video_proof_path']

        logger.debug(f'Начало записи видео для {key}')

        try:
            with av.open(video_filename, "w") as container:
                stream = container.add_stream('h264', rate=10)
                stream.width = width
                stream.height = height
                stream.pix_fmt = 'yuv420p'

                for frame in frames:
                    av_frame = av.VideoFrame.from_ndarray(frame, format='bgr24')
                    for packet in stream.encode(av_frame):
                        container.mux(packet)

                for packet in stream.encode():
                    container.mux(packet)

            logger.info(f"Видео сохранено: {video_filename}")
        except Exception as e:
            logger.error(f"Ошибка записи видео: {e}")
            raise


class BasicEventRegistrator(AbstractEventRegistrator):
    def __init__(self):
        super().__init__()
        self.video_writer = VideoWriter()

    def create_task(self, key_name: str) -> None:
        self.task_catalog[key_name]['task'] = 'CREATED'

    def stop_task(self, key_name: str) -> None:
        self.task_catalog[key_name]['task'] = 'DONE'

    def delete_task(self, key_name: str) -> None:
        if key_name in self.task_catalog:
            del self.task_catalog[key_name]

    def add_task(self, key_name: str, flag: bool) -> None:
        self.task_catalog[key_name]['status'].append(flag)
        self.task_catalog[key_name]['task'] = 'PENDING'

    def clear_key_task(self, key_name: str) -> None:
        self.task_catalog[key_name]['status'].clear()

    def add_video_frame(self, key_name: str, frame: np.ndarray) -> None:
        self.video_writer.add_frame(key_name, frame)

    # TODO СДЕЛАТЬ СКАЧИВАНИЕ КАДРА ПО ССЫЛКЕ И УСТАНОВИТЬ ОПЦИЮ НАДО ЛИ СОХРАНЯТЬ ВИДЕО ИЛИ НЕТ
    # TODO ТАКЖЕ СДЕЛАТЬ СИНГЕЛЕТОН ДЛЯ ЕДИННОЙ ИНИЦИАЛИЗАЦИИ РЕДИСА. НАДО ПОДУМАТЬ НЕОБХОДИМО ЛИ ?

    async def register_task(self, key_name: str, message_event: dict) -> None:
        self.stop_task(key_name)
        if self.task_catalog[key_name]['task'] == 'DONE':
            try:
                self.video_writer.save_video(key_name)
                self.task_catalog[key_name]['task'] = 'FINISHED'
            except Exception as e:
                logger.error(f"Ошибка записи видео для ключа {key_name}: {e}")
                self.task_catalog[key_name]['task'] = 'ERROR'

        try:
            data = {key_name: {'message': message_event,
                               'video_proofs': self.video_writer.video_frame_store[key_name]['video_proof_path'],
                               'image_proof_path': self.video_writer.video_frame_store[key_name]['frame_proof_path']}}
            await self.redis.zadd('events', {json.dumps(data): time.time()})
            logger.info(f"Задача зарегистрирована в редисе")
        except Exception as e:
            logger.error(f"Ошибка записи в редис: {e}")


class EventRegistrator(BasicEventRegistrator):
    async def start_process(self):
        await self.connect_nats()
        try:
            while True:
                self.process(self.inference_data)
                await asyncio.sleep(0.005)
        except KeyboardInterrupt:
            print("Остановлено пользователем")
        finally:
            await self.close()

    @abstractmethod
    def process(self, inference_data) -> None:
        pass
