import math
from collections import defaultdict
from typing import Tuple, Generator

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class SecondPipelines:
    @staticmethod
    def extract_review_scores(
            chunks: Generator[pd.DataFrame],
    ) -> Generator[pd.DataFrame]:
        """Извлечение данных об издателях и оценках"""
        for chunk in chunks:
            # Проверяем, что колонки существуют
            if 'Metadata.Publishers' not in chunk.columns or 'Metrics.Review Score' not in chunk.columns:
                continue
            clean_chunk = chunk[['Metadata.Publishers', 'Metrics.Review Score']].dropna()
            yield clean_chunk

    # @staticmethod
    # def calculate_publisher_std(
    #         chunks: Generator[pd.DataFrame],
    # ) -> Generator[pd.DataFrame]:
    #     """
    #     Вычисляет стандартное отклонение оценок по каждому издателю на основе всех чанков.
    #     """
    #     publisher_stats = defaultdict(lambda: {'n': 0, 'sum': 0.0, 'sum_sq': 0.0})
    #
    #     for chunk in chunks:
    #         sub = chunk[['Metadata.Publishers', 'Metrics.Review Score']].copy()
    #         sub['Metrics.Review Score'] = pd.to_numeric(sub['Metrics.Review Score'], errors='coerce')
    #         sub = sub.dropna()
    #
    #         for publisher, group in sub.groupby('Metadata.Publishers'):
    #             scores = group['Metrics.Review Score']
    #             print(type(scores))
    #             n = len(scores)
    #             s = scores.sum()
    #             s2 = (scores ** 2).sum()
    #
    #             # todo: df
    #             publisher_stats[publisher]['n'] += n
    #             publisher_stats[publisher]['sum'] += s
    #             publisher_stats[publisher]['sum_sq'] += s2
    #
    #     # Финальный расчёт статистик
    #     # todo: df
    #     results = {}
    #     for pub, stats in publisher_stats.items():
    #         n, s, s2 = stats['n'], stats['sum'], stats['sum_sq']
    #         if n < 2:
    #             continue  # std не определён
    #
    #         mean = s / n
    #         # Несмещённая дисперсия
    #         variance = (s2 - (s ** 2) / n) / (n - 1)
    #         if variance < 0:
    #             variance = 0.0
    #         std = math.sqrt(variance)
    #
    #         results[pub] = {
    #             'n': n,
    #             'mean': mean,
    #             'std': std,
    #         }
    #
    #     df = pd.DataFrame.from_dict(results, orient='index')
    #
    #     yield df

    @staticmethod
    def calculate_publisher_std(
            chunks: Generator[pd.DataFrame, None, None],
    ) -> Generator[pd.DataFrame, None, None]:
        """
        Вычисляет стандартное отклонение оценок по каждому издателю.
        Используется один агрегирующий DataFrame. Без dict, без лишних groupby.
        """
        # Инициализируем пустой DataFrame с нужными столбцами и правильными типами
        agg_df = pd.DataFrame(
            columns=['n', 'sum', 'sum_sq'],
            dtype='float64'
        ).astype({'n': 'int64'})

        for chunk in chunks:
            sub = chunk[['Metadata.Publishers', 'Metrics.Review Score']].copy()
            sub['Metrics.Review Score'] = pd.to_numeric(sub['Metrics.Review Score'], errors='coerce')
            sub = sub.dropna()

            if sub.empty:
                continue

            # Агрегация внутри чанка — один groupby
            chunk_agg = sub.groupby('Metadata.Publishers')['Metrics.Review Score'].agg(
                n='size',
                sum='sum',
                sum_sq=lambda x: (x ** 2).sum()
            ).astype({'n': 'int64'})

            # Используем add с fill_value для корректного объединения
            agg_df = agg_df.add(chunk_agg, fill_value=0)


        # Фильтрация: только издатели с n >= 2
        agg_df = agg_df[agg_df['n'] >= 2]
        if agg_df.empty:
            yield pd.DataFrame(columns=['publisher', 'n', 'mean', 'std'])

        print(agg_df.keys())

        # Векторизованный расчёт
        n = agg_df['n']
        s = agg_df['sum']
        s2 = agg_df['sum_sq']

        mean = s / n
        variance = (s2 - (s ** 2) / n) / (n - 1)
        variance = variance.clip(lower=0)
        std = variance ** 0.5

        result_df = pd.DataFrame({
            'publisher': agg_df.index,
            'n': n,
            'mean': mean,
            'std': std
        })

        yield result_df

    @staticmethod
    def get_top_bottom_publishers(
            stats_df: pd.DataFrame,
            top_n: int = 3
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Возвращает топ-N издателей с наибольшим и наименьшим std."""
        if stats_df.empty:
            return stats_df.iloc[0:0], stats_df.iloc[0:0]

        # Сортируем по std
        sorted_df = stats_df.sort_values(by='std').reset_index(drop=True)
        bottom = sorted_df.head(top_n)
        top = sorted_df.tail(top_n).iloc[::-1].reset_index(drop=True)  # от большего к меньшему
        return top, bottom

    @staticmethod
    def plot_publisher_variance(stats_df: pd.DataFrame, top_n: int = 3):
        """Столбчатая диаграмма: top-N издателей с max/min std."""

        top_n = min(top_n, len(stats_df))
        top, bottom = SecondPipelines.get_top_bottom_publishers(stats_df, top_n=top_n)

        selected_df = pd.concat([top, bottom], ignore_index=True)
        publishers = selected_df['publisher'].tolist()
        means = selected_df['mean'].tolist()
        stds = selected_df['std'].tolist()

        plt.figure(figsize=(12, 6))
        y_pos = np.arange(len(publishers))
        plt.bar(y_pos, means, yerr=stds, capsize=5, alpha=0.7, color='skyblue')
        plt.xlabel('Издатель')
        plt.ylabel('Средняя оценка (с разбросом = std)')
        plt.title(f'{top_n} издателя с наибольшим и {top_n} с наименьшим разбросом оценок')
        plt.xticks(y_pos, publishers, rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

        print(f"{top_n} издателей с наибольшим разбросом оценок (std):")
        for _, row in top.iterrows():
            print(f"  {row['publisher']}: std = {row['std']:.2f}, mean = {row['mean']:.2f}, n = {int(row['n'])}")

        print(f"\n{top_n} издателей с наименьшим разбросом оценок (std):")
        for _, row in bottom.iterrows():
            print(f"  {row['publisher']}: std = {row['std']:.2f}, mean = {row['mean']:.2f}, n = {int(row['n'])}")
