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
    def count_games_by_rating_year(
            chunks: Generator[pd.DataFrame],
    ) -> Generator[pd.DataFrame]:
        """
        Подсчёт количества игр по рейтингам и годам.
        Возвращает генератор с одним DataFrame: ['year', 'rating', 'count'].
        """
        # Инициализируем Series с правильным MultiIndex и именами уровней
        total_counts = pd.Series(
            dtype='int64',
            index=pd.MultiIndex.from_tuples([], names=['Release.Year', 'Release.Rating'])
        )

        for chunk in chunks:
            sub = chunk[['Release.Year', 'Release.Rating']]

            # Преобразуем и очищаем данные
            sub['Release.Year'] = pd.to_numeric(sub['Release.Year'], errors='coerce')
            sub = sub.dropna(subset=['Release.Year', 'Release.Rating'])

            sub['Release.Year'] = sub['Release.Year'].astype('int64')
            sub['Release.Rating'] = sub['Release.Rating'].astype('string')

            # Агрегация в чанке
            chunk_counts = sub.groupby(['Release.Year', 'Release.Rating']).size()

            # Инкрементальное накопление
            total_counts = total_counts.add(chunk_counts, fill_value=0).astype('int64')

        # Формируем итоговый DataFrame
        result_df = total_counts.reset_index()
        result_df.columns = ['year', 'rating', 'count']
        result_df = result_df.astype({
            'year': 'int64',
            'rating': 'string',
            'count': 'int64'
        })

        yield result_df

    @staticmethod
    def plot_rating_trends(df: pd.DataFrame) -> None:
        """
        Визуализация трендов по возрастным рейтингам.
        """
        if df.empty:
            print("Нет данных для визуализации.")
            return

        required = {'year', 'rating', 'count'}
        if not required.issubset(df.columns):
            raise ValueError(f"DataFrame должен содержать колонки: {required}")

        # Сортируем по году для корректного порядка на графике
        df = df.sort_values('year').reset_index(drop=True)
        years_all = sorted(df['year'].unique())

        plt.figure(figsize=(12, 6))

        # Группируем по рейтингу и строим отдельно для каждого
        for rating, group in df.groupby('rating'):
            # Приводим к полной временной шкале: если в каком-то году нет рейтинга — count = 0
            series = group.set_index('year')['count']
            full_series = series.reindex(years_all, fill_value=0).sort_index()

            years = full_series.index.tolist()
            counts = full_series.values

            # Исходные данные
            plt.plot(years, counts, label=f'Rating {rating}', alpha=0.5)

            # Скользящее среднее (окно = 3 года)
            moving_avg = full_series.rolling(window=3, min_periods=1).mean()
            plt.plot(years, moving_avg, label=f'{rating} (скользящее среднее)', linewidth=2)

        plt.xlabel('Год')
        plt.ylabel('Количество игр')
        plt.title('Количество выпущенных игр по возрастным рейтингам')
        plt.legend()
        plt.xticks(years_all, rotation=45)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()
