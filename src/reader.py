import cv2
import av

from abs_src.abs_reader import AbstractFrameReader
from src.data_scheme import FrameData
from utils.decorators import reconnect_stream
from utils.err import StreamError, StreamDecodeError


class AVStreamReader(AbstractFrameReader):
    """
        Ридер видеопотока через PyAV (FFmpeg).

        Открывает и читает видео из источников RTSP и файловых, возвращая пакеты кадров.
        Поддерживает пропуск кадров между батчами и внутри батча.
        Автоматически восстанавливает соединение при ошибках чтения.
    """
    def __init__(self):
        """
        Инициализация AVStreamReader.

        Атрибуты от AbstractFrameReader:
            source (str): URL или путь к видео.
            batch_size (int): Количество кадров в одном пакете.
            skip_frames (int): Количество кадров для пропуска между кадрами.
            frames_skip_batch (int): Количество кадров для пропуска между пакетами.
            stream_name (str): Идентификатор потока.
            info (Dict): Метаданные о потоке (ширина, высота, fps).
            frames_processed (int): Счётчик обработанных кадров.

        Собственные атрибуты:
            frame_iterator: Итератор кадров контейнера PyAV.
            need_skip_batch (bool): Флаг необходимости пропуска пакетов.
            frame_count (int): Общее число прочитанных кадров.
        """
        super().__init__()
        self.frame_iterator = None
        self.need_skip_batch = self.frames_skip_batch > 0
        self.frame_count = 0

    @reconnect_stream(retry=5)
    def open(self) -> None:
        """
        Открывает контейнер PyAV и инициализирует параметры потока.

        Логика:
            - Проверка валидности batch_size.
            - Открытие RTSP с опцией prefer_tcp или файла.
            - Поиск видеопотока.
            - Чтение параметров width, height, framerate.
            - Инициализация frame_iterator.
        Исключения:
            ValueError   — если batch_size < 1.
            StreamError  — при проблемах открытия или поиска потока.
        """
        if self.batch_size < 1:
            raise ValueError("Длина батча не может быть меньше 1")

        if self.source.startswith('rtsp://'):
            self.container = av.open(self.source, options={"rtsp_flags": "prefer_tcp"})
        else:
            self.container = av.open(self.source)
        if not self.container:
            raise StreamError("Не удалось открыть контейнер")

        self.stream = self.container.streams.video[0]
        if not self.stream:
            raise StreamError("Не удалось найти видео поток")

        self.frame_width = self.stream.width
        self.frame_height = self.stream.height
        self.framerate = float(self.stream.average_rate)

        self.frame_iterator = self.container.decode(video=0)

    def close(self) -> None:
        """
        Закрывает контейнер PyAV при завершении работы.
        """
        if self.container is not None:
            self.container.close()

    @reconnect_stream(retry=5)
    def get_frame(self):
        """
        Считывает один пакет кадров из потока.

        Логика:
            - При необходимости пропускает frames_skip_batch кадров перед пакетом.
            - Читает до batch_size кадров, пропуская skip_frames между ними.
            - Обновляет счётчики frames_processed и frame_count.
            - После пакета отмечает необходимость пропуска следующего батча.
        Возвращает:
            FrameData: cam_source, список np.ndarray кадров или None, meta.
        Исключения:
            StreamError — при невозможности получить или пропустить кадр.
        """
        if self.need_skip_batch and self.frames_skip_batch > 0:
            skipped = 0
            while skipped < self.frames_skip_batch:
                try:
                    next(self.frame_iterator)
                    skipped += 1
                except (StopIteration, av.EOFError):
                    raise StreamError("Не удалось пропустить кадр после батча")

        self.need_skip_batch = False

        current_batch = []
        for i in range(self.batch_size):
            try:
                frame = next(self.frame_iterator)

                frame_array = frame.to_ndarray(format='rgb24')
                current_batch.append(frame_array)
                self.frames_processed += 1
                self.frame_count += 1

                if i < self.batch_size - 1 and self.skip_frames > 0:
                    skipped = 0
                    while skipped < self.skip_frames:
                        try:
                            next(self.frame_iterator)
                            skipped += 1
                            self.frame_count += 1
                        except (StopIteration, av.EOFError):
                            raise StreamError("Не удалось пропустить кадр после кадра")

            except (StopIteration, av.EOFError):
                if i == 0:
                    raise StreamError("Не удалось получить кадр")
                break

        if self.frames_skip_batch > 0:
            self.need_skip_batch = True

        if current_batch:
            return FrameData(cam_source=self.stream_name, frames=current_batch, meta=self.info)
        else:
            return FrameData(cam_source=self.stream_name, frames=None, meta=self.info)


class OpenCVStreamReader(AbstractFrameReader):
    """
        Ридер видеопотока через OpenCV VideoCapture.

        Поддерживает аналогичный интерфейс AbstractFrameReader:
            open(), get_frame(), close().
        Восстанавливает соединение при ошибках.
    """
    @reconnect_stream(retry=5)
    def open(self) -> None:
        """
        Открывает VideoCapture и инициализирует параметры потока.

        Логика:
            - Проверка валидности batch_size.
            - Открытие источника cv2.VideoCapture.
            - Чтение width, height, fps.
        Исключения:
            ValueError       — если batch_size < 1.
            StreamError      — при неудачном открытии.
        """
        if self.batch_size < 1:
            raise ValueError("Длина батча не может быть меньше 1")

        self.stream = cv2.VideoCapture(self.source)
        if not self.stream.isOpened():
            raise StreamError("Не удалось открыть стрим")

        self.frame_width = int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.framerate = self.stream.get(cv2.CAP_PROP_FPS)

    def close(self) -> None:
        """
        Освобождает ресурс VideoCapture.
        """
        if self.stream is not None:
            self.stream.release()

    def get_frame(self) -> FrameData:
        """
        Считывает один пакет кадров из VideoCapture.

        Логика:
            - При необходимости пропускает frames_skip_batch кадров перед пакетом.
            - Читает до batch_size кадров, пропуская skip_frames между ними.
            - Обновляет счётчик frames_processed.
        Возвращает:
            FrameData: cam_source, список np.ndarray кадров или None, meta.
        Исключения:
            StreamDecodeError — при невозможности получить кадр.
            StreamError       — при ошибках пропуска.
        """
        if self.need_skip_batch and self.frames_skip_batch > 0:
            for _ in range(self.frames_skip_batch):
                ret = self.stream.grab()
                if not ret:
                    raise StreamDecodeError("Не удалось пропустить кадр после батча")

        self.need_skip_batch = False

        current_batch = []
        for i in range(self.batch_size):
            ret, frame = self.stream.read()
            if not ret:
                if i == 0:
                    raise StreamDecodeError("Не удалось получить кадр")
                break

            current_batch.append(frame)
            self.frames_processed += 1

            if i < self.batch_size - 1 and self.skip_frames > 0:
                for _ in range(self.skip_frames):
                    ret = self.stream.grab()
                    if not ret:
                        raise StreamError("Не удалось пропустить кадр после кадра")

        if self.frames_skip_batch > 0:
            self.need_skip_batch = True

        if current_batch:
            return FrameData(cam_source=self.stream_name, frames=current_batch, meta=self.info)
        else:
            return FrameData(cam_source=self.stream_name, frames=None, meta=self.info)


if __name__ == "__main__":
    with AVStreamReader() as stream_reader:
        batch_count = 0
        for batch in stream_reader:
            if batch is None:
                break
            batch_count += 1
            print(stream_reader.info)

            frame_bgr_1 = cv2.cvtColor(batch[0], cv2.COLOR_RGB2BGR)
            frame_bgr_2 = cv2.cvtColor(batch[-1], cv2.COLOR_RGB2BGR)
            cv2.imshow('frame', frame_bgr_1)
            cv2.imshow('frame1', frame_bgr_2)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cv2.destroyAllWindows()

    #     with OpenCVStreamReader() as stream_reader:
    #         for batch in stream_reader:
    #             if batch is None:
    #                 break
    #             frame = batch[0]
    #             cv2.imshow('frame', frame)
    #             cv2.imshow('frame2', batch[-1])
    #             if cv2.waitKey(0) & 0xFF == ord('q'):
    #                 break
    #     cv2.destroyAllWindows()
