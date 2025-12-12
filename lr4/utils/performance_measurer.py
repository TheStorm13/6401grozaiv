import asyncio
import functools
import inspect
import logging
import time
from typing import Callable, Any

logger = logging.getLogger(__name__)


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
        if inspect.iscoroutinefunction(func):
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                # Внутри уже работающего цикла событий нельзя вызвать asyncio.run.
                # Подсказка: используйте measure_time_async в асинхронном коде.
                raise RuntimeError("Для асинхронных функций используйте measure_time_async внутри async-кода.")

            async def _runner():
                start_time = time.perf_counter()
                result = await func(*args, **kwargs)
                end_time = time.perf_counter()
                return result, end_time - start_time

            result, execution_time = asyncio.run(_runner())
            function_name = func.__name__
            logger.info("Функция '%s' выполнена за: %.6f секунд", function_name, execution_time)
            return result, execution_time, function_name

        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()

        execution_time = end_time - start_time
        function_name = func.__name__
        logger.info("Функция '%s' выполнена за: %.6f секунд", function_name, execution_time)

        return result, execution_time, function_name

    @staticmethod
    async def measure_time_async(func: Callable[..., Any], *args, **kwargs) -> tuple:
        """
        Асинхронный вариант измерения времени. Вызывать внутри async-кода.

        Returns:
            tuple: (результат функции, время выполнения в секундах, имя функции)
        """
        start_time = time.perf_counter()
        if inspect.iscoroutinefunction(func):
            result = await func(*args, **kwargs)
        else:
            # Вызов синхронной функции без блокировки цикла событий.
            result = await asyncio.to_thread(func, *args, **kwargs)
        end_time = time.perf_counter()

        execution_time = end_time - start_time
        function_name = func.__name__
        logger.info("Функция '%s' выполнена за: %.6f секунд", function_name, execution_time)
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

        if inspect.iscoroutinefunction(func):
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                start_time = time.perf_counter()
                result = await func(*args, **kwargs)
                end_time = time.perf_counter()

                execution_time = end_time - start_time
                logger.info("Функция '%s' выполнена за: %.6f секунд", func.__name__, execution_time)

                return result

            return wrapper

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()

            execution_time = end_time - start_time
            logger.info("Функция '%s' выполнена за: %.6f секунд", func.__name__, execution_time)

            return result

        return wrapper
