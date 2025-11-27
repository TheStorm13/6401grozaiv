from collections import defaultdict
from typing import Dict, Generator

import matplotlib.pyplot as plt
import pandas as pd


class ThirdPipelines:
    @staticmethod
    def extract_rating_data(chunks: Generator[pd.DataFrame]) -> Generator[pd.DataFrame]:
        """Извлечение данных о годе выпуска и рейтинге"""
        for chunk in chunks:
            # Фильтруем только нужные рейтинги
            filtered_chunk = chunk[['Release.Year', 'Release.Rating']]
            filtered_chunk = filtered_chunk[filtered_chunk['Release.Rating'].isin(['E', 'T', 'M'])]
            filtered_chunk = filtered_chunk.dropna()
            yield filtered_chunk

    @staticmethod
    def count_games_by_rating_year(chunks: Generator[pd.DataFrame]) -> Generator[Dict[int, Dict[str, int]]]:
        """Подсчет количества игр по рейтингам и годам"""
        rating_year_count = defaultdict(lambda: defaultdict(int))

        for chunk in chunks:
            for year, rating in zip(chunk['Release.Year'], chunk['Release.Rating']):
                rating_year_count[year][rating] += 1

        # Преобразуем в удобный для визуализации формат
        result = {}
        for year, ratings in rating_year_count.items():
            result[year] = dict(ratings)

        yield result

    @staticmethod
    def plot_rating_trends(rating_data: Dict[int, Dict[str, int]]) -> None:
        """Визуализация трендов по рейтингам."""
        # Собираем данные в удобный формат
        years = sorted(rating_data.keys())
        ratings = ['E', 'T', 'M']

        # Создаем словарь для хранения скользящих средних
        moving_avgs = {rating: [] for rating in ratings}

        plt.figure(figsize=(12, 6))

        for rating in ratings:
            counts = [rating_data[year].get(rating, 0) for year in years]

            # Линейный график исходных данных
            plt.plot(years, counts, label=f'Rating {rating}', alpha=0.5)

            # Скользящее среднее (окно = 3 года)
            window = 3
            for i in range(len(counts)):
                start = max(0, i - window + 1)
                end = i + 1
                moving_avg = sum(counts[start:end]) / (end - start)
                moving_avgs[rating].append(moving_avg)

            # Линейный график скользящего среднего
            plt.plot(years, moving_avgs[rating], label=f'{rating} (скользящее среднее)', linewidth=2)

        plt.xlabel('Год')
        plt.ylabel('Количество игр')
        plt.title('Количество выпущенных игр по возрастным рейтингам')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
