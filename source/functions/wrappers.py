import time

from typing import Callable, Any


def print_status_and_time(func: Callable) -> Callable:
    """
    A decorator to measure and print the execution time of a function.

    This decorator wraps a function and computes the time taken for its execution.
    It prints the function name along with the elapsed time in minutes and seconds.

    Args:
        func (Callable): The function to be wrapped and timed.

    Returns:
        Callable: The wrapped function with added timing and status printing.
    """

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        total_time = end_time - start_time
        sec = int(total_time % 60)
        min = int(total_time // 60)
        print(f"{func.__name__:<25} -- OK. Time elapsed: {min} min {sec} sec")
        return result

    return wrapper
