import asyncio
import json
import os
import time
from nats.aio.client import Client as NATS
from utils.logger import logger


class Router:
    def __init__(self):
        self.nats_host = os.getenv("nats_host", "nats://localhost:4222").split(",")
        self.redis_cli = os.getenv('redis_host', 'localhost')
        self.redis_port = os.getenv('redis_port', 6379)

        self.nats_cli = None
        self.topic_input = os.getenv('topic_stream')
        self.sub = None

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.close()
        return None

    async def connect(self):
        self.nats_cli = NATS()
        try:
            await self.nats_cli.connect(
                servers=self.nats_host,
                max_reconnect_attempts=-1,
                reconnect_time_wait=2
            )
            logger.info("Connected to NATS SERVER")
        except Exception as e:
            logger.error(f"Connection error: {e}")
            raise

    async def close(self):
        try:
            if self.nats_cli.is_connected:
                await self.nats_cli.drain()
        except Exception as e:
            logger.error(f"Error during drain: {e}")
        finally:
            await self.nats_cli.close()

    async def subscribe(self):
        if self.nats_cli is None or not self.nats_cli.is_connected:
            raise RuntimeError("NATS client not connected")

        # Используем callback для обработки сообщений
        self.sub = await self.nats_cli.subscribe(
            subject=f"{self.topic_input}",
            cb=self.message_handler
        )
        logger.info(f"Subscribed to {self.topic_input}")

    async def message_handler(self, msg):
        subject = msg.subject
        data_str = msg.data.decode()
        data = json.loads(data_str)
        logger.debug(f"Received message [{subject}]: {data}")

        await self.publish(data)

        # Здесь можно добавить обработку сообщения
        # Например, вызов внешних сервисов или запись в Redis

    async def publish(self, data):
        model_count = len(data["models"])
        for model in data["models"]:
            output_topic = f"inference.{model}"

            message = {
                "frame_id": data["frame_id"],
                "seaweed_url": data["seaweed_url"],
                "model": model,
                "timestamp": time.time(),
                "cached_key": f"frame_meta:{data['frame_id']}"
            }

            print(f"Publishing message: {message}")

            await self.nats_cli.publish(output_topic, json.dumps(message).encode())

        logger.info(f"Routed frame {data['frame_id']} to {model_count} models")

    async def run(self):
        await self.subscribe()
        logger.info("Listening for messages...")
        while True:
            await asyncio.sleep(0.005)  # Бесконечный цикл с минимальной нагрузкой


if __name__ == '__main__':
    async def main():
        async with Router() as router:
            await router.run()  # Запускаем постоянное прослушивание


    asyncio.run(main())

# import asyncio
# import json
# import time
# from abs_src.abs_router import AbstractRouter
#
#
# class Router(AbstractRouter):
#     def __init__(self):
#         super().__init__()
#         self.sub = None
#
#     async def connect(self):
#         try:
#             await self.nats_cli.connect(
#                 servers=self.nats_servers,
#                 max_reconnect_attempts=-1,
#                 reconnect_time_wait=2,
#             )
#             self.logger.info(f"Connected to NATS at {self.nats_servers}")
#
#             # Verify Redis connection
#             self.redis_cli.ping()
#             self.logger.info(f"Connected to Redis at {self.redis_host}:{self.redis_port}")
#         except Exception as e:
#             self.logger.error(f"Connection error: {str(e)}")
#             raise
#
#     async def close_connection(self):
#         try:
#             await self.nats_cli.drain()
#         except Exception as e:
#             self.logger.error(f"Error draining NATS: {str(e)}")
#         finally:
#             self.nats_cli.disconnect()
#
#         self.redis_cli.close()
#         self.logger.info("Router shutdown complete")
#
#     async def subscribe(self):
#         self.sub = await self.nats_cli.subscribe(f'{self.input_topic}.*')
#
#     async def cache_frame_metadata(self, data):
#         """Cache frame metadata in Redis asynchronously"""
#         loop = asyncio.get_running_loop()
#         try:
#             await loop.run_in_executor(
#                 None,
#                 self._sync_cache_frame_metadata,
#                 data
#             )
#         except Exception as e:
#             self.logger.error(f"Caching failed: {str(e)}")
#
#     def _sync_cache_frame_metadata(self, data):
#         key = f"frame_meta:{data['frame_id']}"
#         value = {
#             "seaweed_url": data["seaweed_url"],
#             "timestamp": data["timestamp"]
#         }
#         self.redis_cli.hset(key, mapping=value)
#         self.redis_cli.expire(key, 600)
#
#     async def is_duplicate(self, frame_id):
#         """Check for duplicates asynchronously"""
#         loop = asyncio.get_running_loop()
#         try:
#             return await loop.run_in_executor(
#                 None,
#                 self._sync_is_duplicate,
#                 frame_id
#             )
#         except Exception as e:
#             self.logger.error(f"Duplicate check failed: {str(e)}")
#             return True  # Treat errors as duplicates to avoid duplicates
#
#     def _sync_is_duplicate(self, frame_id):
#         key = f"frame:{frame_id}"
#         if self.redis_cli.exists(key):
#             return True
#         self.redis_cli.setex(key, 600, "1")
#         return False
#
#     async def process(self):
#         try:
#             await self.connect()
#             await self.subscribe()
#             self.logger.info("Router started")
#
#             while True:
#                 try:
#                     msg = await self.sub.next_msg(timeout=1.0)
#                     await self.message_handler(msg)
#                 except asyncio.TimeoutError:
#                     pass
#                 except Exception as e:
#                     self.logger.error(f"Error receiving message: {str(e)}")
#                     # Re-establish connection on error
#                     await self.close_connection()
#                     await self.connect()
#                     await self.subscribe()
#
#         except KeyboardInterrupt:
#             self.logger.info("Keyboard interrupt received, shutting down")
#         except Exception as e:
#             self.logger.error(f"Critical error: {str(e)}")
#         finally:
#             await self.close_connection()
#
#     async def publish(self, data):
#         model_count = len(data["models"])
#         for model in data["models"]:
#             output_topic = f"{self.output_topic}.{model}"
#
#             message = {
#                 "frame_id": data["frame_id"],
#                 "seaweed_url": data["seaweed_url"],
#                 "model": model,
#                 "timestamp": time.time(),
#                 "cached_key": f"frame_meta:{data['frame_id']}"
#             }
#
#             print(f"Publishing message: {message}")
#
#             await self.nats_cli.publish(output_topic, json.dumps(message).encode())
#
#         self.logger.info(f"Routed frame {data['frame_id']} to {model_count} models")
#
#     async def message_handler(self, msg):
#         """Handle incoming messages"""
#         try:
#             data = json.loads(msg.data.decode())
#             self.logger.debug(f"Received message: {data['frame_id']}")
#
#             # Deduplicate
#             if await self.is_duplicate(data["frame_id"]):
#                 self.logger.debug(f"Duplicate frame skipped: {data['frame_id']}")
#                 return
#
#             await self.cache_frame_metadata(data)
#
#             # Route to models
#             await self.publish(data)
#
#         except Exception as e:
#             self.logger.error(f"Error processing message: {str(e)}")
#
#
# if __name__ == "__main__":
#     router = Router()
#     asyncio.run(router.process())
