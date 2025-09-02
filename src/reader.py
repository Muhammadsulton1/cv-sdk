import cv2
import av
import numpy as np

from typing import List
from abs_src.abs_reader import AbstractFrameReader
from src.data_scheme import FrameData
from utils.decorators import reconnect_stream
from utils.err import StreamError, StreamDecodeError


class AVStreamReader(AbstractFrameReader):
    def __init__(self):
        super().__init__()
        self.frame_iterator = None
        self.need_skip_batch = self.frames_skip_batch > 0
        self.frame_count = 0

    @reconnect_stream(retry=5)
    def open(self) -> None:
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
        if self.container is not None:
            self.container.close()

    @reconnect_stream(retry=5)
    def get_frame(self):
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

        # return current_batch if current_batch else None

        if current_batch:
            return FrameData(cam_source=self.stream_name, frames=current_batch, meta=self.info)
        else:
            return FrameData(cam_source=self.stream_name, frames=None, meta=self.info)


class OpenCVStreamReader(AbstractFrameReader):
    @reconnect_stream(retry=5)
    def open(self) -> None:
        if self.batch_size < 1:
            raise ValueError("Длина батча не может быть меньше 1")

        self.stream = cv2.VideoCapture(self.source)
        if not self.stream.isOpened():
            raise StreamError("Не удалось открыть стрим")

        self.frame_width = int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.framerate = self.stream.get(cv2.CAP_PROP_FPS)

    def close(self) -> None:
        if self.stream is not None:
            self.stream.release()

    def get_frame(self) -> FrameData:
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

        #return current_batch if current_batch else None


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
