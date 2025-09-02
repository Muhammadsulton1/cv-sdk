import uuid
from abc import ABC, abstractmethod
import os

from src.singeleton.yaml_reader import YamlReader


class AbstractFrameReader(ABC):
    """
    Абстрактный базовый класс для чтения и итерирования видеокадров из различных источников.

    Отвечает за:
        - Загрузку конфигурации из YAML-файла.
        - Управление циклом открытия, чтения и закрытия потока.
        - Реализацию интерфейса iterator для постраничного (батчевого) чтения кадров.
        - Сбор и предоставление метаданных о потоке.
    """
    def __init__(self):
        """
        Инициализация абстрактного ридера.

        Из YAML-конфигурации (через YamlReader) подгружаются:
            - source: URL или путь до видеопотока.
            - batch_size: число кадров в одном батче.
            - skip_frames: число кадров для пропуска между кадрами в батче.
            - frames_skip_batch: число кадров для пропуска между батчами.

        Генерируется уникальное имя потока stream_name.

        Атрибуты:
            frame_height (int | None): высота кадра.
            frame_width (int | None): ширина кадра.
            framerate (float | None): частота кадров.
            frames_processed (int): счётчик обработанных кадров.
            stream: объект для чтения кадров (cv2.VideoCapture или PyAV container).
            container: контейнер мультимедиа (для PyAV).
            setup_settings (Dict): словарь настроек из YAML.
            reader_service (str): имя сервиса из переменной окружения SERVICE_NAME.
            skip_frames (int): пропуск кадров внутри батча.
            frames_skip_batch (int): пропуск кадров между батчами.
            need_skip_batch (bool): флаг необходимости пропуска перед первым кадром батча.
            source (str): источник видеопотока.
            batch_size (int): размер батча.
            stream_name (str): уникальное короткое имя потока ("cam_<uuid>").
        Исключения:
            ValueError: если source не задан в конфигурации.
        """

        self.frame_height = None
        self.frame_width = None
        self.framerate = None
        self.frames_processed = 0
        self.stream = None
        self.container = None

        self.setup_settings = YamlReader().reader_settings
        self.reader_service = os.getenv('SERVICE_NAME')
        service_config = self.setup_settings.get(self.reader_service, {})

        self.skip_frames = int(service_config.get('skip_frames', 0))
        self.frames_skip_batch = int(service_config.get('frames_skip_batch', 0))
        self.need_skip_batch = self.frames_skip_batch > 0

        self.source = service_config.get('source')
        self.batch_size = int(service_config.get('batch_size', 1))
        if not self.source:
            raise ValueError("Источник видео не установлен")

        self.stream_name = f"cam_{str(uuid.uuid4())[:3]}"

    def __enter__(self):
        """
        Вход в контекстный менеджер.
        Вызывает метод open() для установки соединения и подготовки к чтению кадров.
        Возвращает:
            self (AbstractFrameReader): готовый к итерации объект.
        """
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Выход из контекстного менеджера.
        Закрывает поток, вызывая метод close().
        """
        self.close()

    @abstractmethod
    def open(self) -> None:
        """
        Абстрактный метод для открытия и инициализации источника кадрирования.

        Должен установить:
            - self.stream или self.container
            - Метаданные: frame_width, frame_height, framerate
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """
        Абстрактный метод для корректного завершения работы с источником.
        Должен освобождать ресурсы (закрывать контейнер или VideoCapture).
        """
        pass

    @abstractmethod
    def get_frame(self):
        """
       Абстрактный метод для получения одного батча кадров.

       Должен возвращать:
           FrameData(caм_source, frames: List[np.ndarray] | None, meta: Dict)
       или None/FrameData с frames=None при окончании потока.
       """
        pass

    def __iter__(self):
        """
        Возвращает итератор для работы в цикле for.
        """
        return self

    def __next__(self):
        """
        Возвращает следующий батч кадров через get_frame().

        При frames=None либо None вызывает StopIteration.
        """
        frames = self.get_frame()
        if frames is None:
            raise StopIteration
        return frames

    @property
    def info(self):
        """
        Метаданные текущего состояния ридера.

        Включает:
            source, source_name, width, height, framerate,
            frames_skipped, frames_processed, batch_size.

        Возвращает:
            Dict[str, Union[str,int,float]]
        """
        return {
            "source": self.source,
            "source_name": self.stream_name,
            "width": self.frame_width,
            "height": self.frame_height,
            "framerate": self.framerate,
            "frames_skipped": self.skip_frames,
            "frames_processed": self.frames_processed,
            "batch_size": self.batch_size
        }
