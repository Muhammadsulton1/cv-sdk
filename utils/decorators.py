import time
import functools


def measure_latency_async(name=None):
    def decorator(func):
        func_name = name or func.__name__

        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            result = await func(*args, **kwargs)
            end_time = time.perf_counter()
            elapsed_time = (end_time - start_time) * 1000

            print(f"[LATENCY of async function] {func_name}: {elapsed_time:.2f} ms")

            return result

        return wrapper

    return decorator


def measure_latency_sync(name=None):
    def decorator(func):
        func_name = name or func.__name__

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            elapsed_time = (end_time - start_time) * 1000
            print(f"[LATENCY] {func_name}: {elapsed_time:.2f} ms")

            return result
        return wrapper
    return decorator
