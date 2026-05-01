import asyncio
import signal
import os
from typing import Type

from common.abstractions.abs_jetstream import AbstractStreamManager, AbstractProcessor
from common.utils.logger import logger


class BaseServiceRunner:
    """
    Универсальный раннер: запускает любые комбинации
    AbstractProcessor и AbstractStreamManager.

    Использование:
        ServiceRunner(ReaderProcessor).run()
        ServiceRunner(InferenceConsumer).run()
        ServiceRunner(ReaderProcessor, InferenceConsumer).run()
    """

    def __init__(self, *processor_classes: Type[AbstractProcessor | AbstractStreamManager]) -> None:
        if not processor_classes:
            raise ValueError("Укажите хотя бы один класс процессора")
        self._processor_classes = processor_classes
        self._tasks: list[asyncio.Task] = []
        self._stop_event = asyncio.Event()

    @staticmethod
    async def _run_processor(cls: Type[AbstractProcessor | AbstractStreamManager]) -> None:
        """
        Определяет тип процессора и запускает его правильным способом:
        - AbstractStreamManager → async with cls() as p: await p.process()
        - AbstractProcessor     → p = cls(); await p.process()
        """
        name = cls.__name__

        if issubclass(cls, AbstractStreamManager):
            logger.info("[%s] Запуск через context manager (JetStream + S3)", name)
            async with cls() as processor:
                await processor.process()

        elif issubclass(cls, AbstractProcessor):
            logger.info("[%s] Запуск как processor", name)
            processor = cls()
            await processor.process()

        else:
            raise TypeError(
                f"{name} должен наследовать AbstractStreamManager или AbstractProcessor"
            )

    def _setup_signals(self) -> None:
        loop = asyncio.get_event_loop()

        def _handle_signal(sig):
            logger.info("Получен сигнал %s — останавливаю сервис...", sig.name)
            self._stop_event.set()
            for task in self._tasks:
                task.cancel()

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, _handle_signal, sig)
            except NotImplementedError:
                pass

    async def _main(self) -> None:
        self._setup_signals()
        service_name = os.getenv("SERVICE_NAME", "unknown")
        names = [cls.__name__ for cls in self._processor_classes]

        logger.info("Запуск сервиса SERVICE_NAME=%s, processors=%s", service_name, names)

        self._tasks = [
            asyncio.create_task(
                self._run_processor(cls),
                name=cls.__name__,
            )
            for cls in self._processor_classes
        ]

        try:
            await asyncio.gather(*self._tasks)
        except asyncio.CancelledError:
            logger.info("Сервис %s остановлен", service_name)
        except Exception as e:
            logger.error("Ошибка в сервисе %s: %s", service_name, e, exc_info=True)
            raise
        finally:
            for task in self._tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*self._tasks, return_exceptions=True)
            logger.info("Все таски завершены")

    def run(self) -> None:
        """Точка входа — запускает event loop."""
        asyncio.run(self._main())
