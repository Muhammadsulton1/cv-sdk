import asyncio
import aiohttp
import uuid

import cv2
import nats
from nats.errors import ConnectionClosedError, TimeoutError

from abs_src.modules_interface import FileRouter
from video_reader import VideoStreamReader


class SeaweedFSRouter(FileRouter):
    def __init__(self, master_url="http://localhost:9333", volume_url=None,
                 bucket_name="video_stream"):
        self.master_url = master_url
        self.volume_url = volume_url
        self.bucket_name = bucket_name
        self._session = None

    async def __aenter__(self):
        self._session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self._session:
            await self._session.close()

    async def upload_file(self, file_data: bytes, ttl: str = "10m") -> str:
        assign_url = f"{self.master_url}/dir/assign?ttl={ttl}"
        async with self._session.get(assign_url) as resp:
            assign_data = await resp.json()
            fid = assign_data["fid"]

        upload_url = f"http://localhost:8888/video-stream/{fid}?ttl={ttl}"
        file_name = f"{uuid.uuid4()}.jpg"

        form_data = aiohttp.FormData()
        form_data.add_field(
            'file',
            file_data,
            filename=file_name,
            content_type='image/jpeg'
        )

        async with self._session.post(upload_url, data=form_data) as resp:
            resp.raise_for_status()

        return f"http://localhost:8888/video-stream/{fid}"


async def main():
    source = "video/test_maneken.mp4"
    nats_url = "nats://localhost:4222"
    topic = "inference_frames"

    try:
        nc = await nats.connect(
            servers=[nats_url],
            connect_timeout=5,
            max_reconnect_attempts=3
        )
    except (ConnectionClosedError, TimeoutError) as e:
        print(f"Failed to connect to NATS: {e}")
        return

    try:
        with VideoStreamReader(source, skip_frames=0) as stream:
            while True:
                frame = stream.get_frame()
                if frame is None:
                    break

                _, buffer = cv2.imencode('.jpg', frame)
                image_data = buffer.tobytes()

                async with SeaweedFSRouter() as router:
                    try:
                        file_url = await router.upload_file(image_data)
                        await nc.publish(topic, file_url.encode())
                    except Exception as e:
                        continue

                #await asyncio.sleep(0.01)

    except KeyboardInterrupt:
        print("Processing interrupted by user")
    finally:
        if nc.is_connected:
            await nc.drain()
            await nc.close()


if __name__ == "__main__":
    # Обработка сигналов прерывания
    # loop = asyncio.new_event_loop()
    # asyncio.set_event_loop(loop)

    asyncio.run(main())

    # try:
    #     loop.run_until_complete(main())
    # except KeyboardInterrupt:
    #     print("Program terminated")
    # finally:
    #     loop.close()
    #     sys.exit(0)