import functools
import time
from typing import Callable, Any


# todo: убрать typing из сигнатуры
class PerformanceMeasurer:
    """Класс для измерения времени выполнения функций."""

    @staticmethod
    def measure_time(func: Callable[..., Any], *args, **kwargs) -> tuple:
        """
        Измеряет время выполнения функции и возвращает результат и время.

        Args:
            func: функция для выполнения
            *args: позиционные аргументы функции
            **kwargs: именованные аргументы функции

        Returns:
            tuple: (результат функции, время выполнения в секундах, имя функции)
        """
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()

        execution_time = end_time - start_time
        function_name = func.__name__

        print(f"Функция '{function_name}' выполнена за: {execution_time:.6f} секунд")

        return result, execution_time, function_name

    @staticmethod
    def measure_time_decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        """
        Декоратор для автоматического измерения времени выполнения функции.

        Usage:
        @PerformanceMeasurer.measure_time_decorator
        def my_function():
            # код функции
        """

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()

            execution_time = end_time - start_time
            print(f"Функция '{func.__name__}' выполнена за: {execution_time:.6f} секунд")

            return result

        return wrapper
