import asyncio
import time
import functools

from utils.logger import logger


def measure_latency_async():
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            result = await func(*args, **kwargs)
            end_time = time.perf_counter()
            elapsed_time = (end_time - start_time) * 1000

            print(f"[LATENCY of async function] {func.__name__}: {elapsed_time:.2f} ms")

            return result

        return wrapper

    return decorator


def measure_latency_sync():
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            elapsed_time = (end_time - start_time) * 1000
            print(f"[LATENCY] {func.__name__}: {elapsed_time:.2f} ms")

            return result

        return wrapper

    return decorator


def measure_detailed_time(func):
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        start_cpu = time.process_time()

        result = await func(*args, **kwargs)

        end_time = time.perf_counter()
        end_cpu = time.process_time()

        wall_time = (end_time - start_time) * 1000
        cpu_time = (end_cpu - start_cpu) * 1000

        print(f"📊 {func.__name__} детальная статистика:")
        print(f"   ├─ Общее время (wall clock): {wall_time:.2f} мс")
        print(f"   ├─ CPU время: {cpu_time:.2f} мс")
        print(f"   └─ I/O время: {wall_time - cpu_time:.2f} мс")

        return result

    return wrapper


def retry(retries=3, delay=1):
    """
    Декоратор для повторной попытки выполнения функции в случае неудачи.
    :param retries: количество попыток (по умолчанию 3)
    :type retries: int
    :param delay: задержка между попытками в секундах (по умолчанию 1 секунда)
    :type delay: int
    """

    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            for attempt in range(1, retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    logger.warning(f"Попытка {attempt} не удалась с ошибкой: {e}")
                    if attempt < retries:
                        await asyncio.sleep(delay)
            raise RuntimeError(f"Функция '{func.__name__}' не удалась после {retries} попыток.")

        return wrapper

    return decorator
