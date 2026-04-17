import time


def measure_latency(func, *args, **kwargs):
    start = time.time()
    result = func(*args, **kwargs)
    latency = time.time() - start
    return result, latency