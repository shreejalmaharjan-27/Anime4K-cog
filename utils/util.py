import torch

def timed(fn):
    if torch.cuda.is_available():
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = fn()
        end.record()
        torch.cuda.synchronize()
        return result, start.elapsed_time(end) / 1000
    else:
        from time import perf_counter
        start = perf_counter()
        result = fn()
        end = perf_counter()
        return result, end - start