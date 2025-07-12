import av
import cv2

from abs_src.abs_reader import AbstractReader


class VideoStreamReader(AbstractReader):
    def __init__(self, video_source, skip_frames=0, stream_index=0):
        super().__init__(video_source, skip_frames, stream_index)
        self.source = video_source
        self.skip_frames = skip_frames
        self.stream_index = stream_index
        self.container = None
        self.video_stream = None
        self.frame_generator = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def open(self):
        self.container = av.open(
            self.source,
            options={"rtsp_flags": "prefer_tcp"}
        )

        if self.container is None:
            raise RuntimeError(f"Ошибка открытия источника: {self.source}")

        video_stream = self.container.streams.video[self.stream_index]
        self.video_stream = video_stream

        self.codec_name = video_stream.codec_context.name
        self.width = video_stream.codec_context.width
        self.height = video_stream.codec_context.height
        self.framerate = float(video_stream.average_rate)

        # Инициализация генератора кадров
        self.frame_generator = self._generate_frames()

    def close(self):
        if self.container:
            self.container.close()

    def _generate_frames(self):
        """Приватный генератор кадров"""
        frame_count = 0
        for packet in self.container.demux(self.video_stream):
            for frame in packet.decode():
                frame_count += 1
                if self.skip_frames > 0 and frame_count % (self.skip_frames + 1) != 0:
                    continue
                rgb_frame = frame.to_ndarray(format="bgr24")
                yield rgb_frame

    def get_frame(self):
        """Возвращает следующий кадр как изображение"""
        try:
            return next(self.frame_generator)
        except StopIteration:
            return None

    def __iter__(self):
        return self

    def __next__(self):
        """Поддержка итерации"""
        frame = self.get_frame()
        if frame is None:
            raise StopIteration
        return frame

    @property
    def info(self):
        return {
            "source": self.source,
            "codec": self.codec_name,
            "width": self.width,
            "height": self.height,
            "framerate": self.framerate,
            "frames_skipped": self.skip_frames
        }


class OpencvVideoReader(AbstractReader):
    def __init__(self, video_source, skip_frames=0, stream_index=0):
        super().__init__(video_source, skip_frames, stream_index)
        self.source = video_source
        self.skip_frames = skip_frames
        self.stream_index = stream_index
        self.video_stream = None
        self.codec_name = "unknown"
        self.width = 0
        self.height = 0
        self.framerate = 0
        self.frames_processed = 0

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def open(self):
        self.video_stream = cv2.VideoCapture(self.source)
        if not self.video_stream.isOpened():
            raise RuntimeError(f"Ошибка открытия источника: {self.source}")

        # Получаем свойства видео
        self.width = int(self.video_stream.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.framerate = self.video_stream.get(cv2.CAP_PROP_FPS)

        # Получаем FourCC кодек и преобразуем в строку
        fourcc_int = self.video_stream.get(cv2.CAP_PROP_FOURCC)
        if fourcc_int != 0:
            self.codec_name = "".join([
                chr(int(fourcc_int) >> 8 * i & 0xFF)
                for i in range(4)
            ])

        self.frames_processed = 0

    def close(self):
        if self.video_stream is not None and self.video_stream.isOpened():
            self.video_stream.release()
        self.video_stream = None

    def get_frame(self):
        """Возвращает следующий кадр с учетом skip_frames."""
        # Пропускаем кадры с помощью grab() для эффективности
        for _ in range(self.skip_frames):
            if not self.video_stream.grab():
                return None
            self.frames_processed += 1

        # Читаем и декодируем актуальный кадр
        ret, frame = self.video_stream.read()
        self.frames_processed += 1

        if not ret:
            return None
        return frame

    def __iter__(self):
        return self

    def __next__(self):
        """Поддержка итерации."""
        frame = self.get_frame()
        if frame is None:
            raise StopIteration
        return frame

    @property
    def info(self):
        return {
            "source": self.source,
            "codec": self.codec_name,
            "width": self.width,
            "height": self.height,
            "framerate": self.framerate,
            "frames_skipped": self.skip_frames,
            "frames_processed": self.frames_processed
        }
