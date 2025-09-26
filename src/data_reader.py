import asyncio

import cv2
from abs_src.abs_managers import AbstractReaderManager
from src.reader import OpenCVStreamReader, AVStreamReader
from utils.logger import logger
from utils.decorators import measure_latency_async


class VideoReaderFactory:
    """"Фабрика для создания объектов чтения видео."""

    _readers = {
        "opencv": OpenCVStreamReader,
        "av": AVStreamReader,
    }

    @classmethod
    def create_reader(cls, reader_type: str):
        """
        Создает экземпляр видео ридера указанного типа.

        Args:
            reader_type: Тип ридера ('opencv' или 'av')

        Returns:
            AbstractReader: Экземпляр класса ридера

        Raises:
            ValueError: Если указан неподдерживаемый тип ридера
        """
        reader_class = cls._readers.get(reader_type.lower())

        if reader_class is None:
            supported = ", ".join(cls._readers.keys())
            raise ValueError(
                f"Не поддерживаемый источник для чтения видео потока: '{reader_type}'. "
                f"поддерживаемые типы: {supported}"
            )

        return reader_class()

    @classmethod
    def register_reader(cls, reader_type: str, reader_class):
        """
        Регистрирует новый тип ридера в фабрике.

        Args:
            reader_type: Идентификатор типа ридера
            reader_class: Класс ридера (должен быть вызываемым)

        Raises:
            TypeError: Если переданный класс не вызываемый
        """
        if not callable(reader_class):
            raise TypeError("VideoReader class должен быть вызываемым объектом")
        cls._readers[reader_type.lower()] = reader_class

    @classmethod
    def supported_readers(cls):
        """"Возвращает список поддерживаемых типов ридеров."""
        return list(cls._readers.keys())


class ReaderManager(AbstractReaderManager):
    def __init__(self):
        super().__init__()

        self.reader = VideoReaderFactory.create_reader(self.reader_type)

    async def runner(self):
        try:
            with self.reader as stream:
                loop = asyncio.get_event_loop()

                while True:
                    try:
                        frames = await loop.run_in_executor(None, stream.get_frame)
                        if frames is None:
                            logger.info("Конец видеопоток")
                            break

                        message = await self.s3_cli.upload_object(frames)
                        print(message)

                        await self.publish_to_nats(message)

                    except Exception as e:
                        logger.error(f"Ошибка в цикле обработки: {e}", exc_info=True)
                        continue

        except KeyboardInterrupt:
            logger.info("Остановка по запросу пользователя")
        except Exception as e:
            logger.exception(f"Критическая ошибка в runner: {e}")


if __name__ == '__main__':
    reader = ReaderManager()
    reader.process()
