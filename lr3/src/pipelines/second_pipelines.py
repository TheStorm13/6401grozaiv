import math
from collections import defaultdict
from typing import Dict, Tuple, Generator

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

    @staticmethod
    def calculate_publisher_std(
            chunks: Generator[pd.DataFrame],
    ) -> Generator[pd.DataFrame]:
        """
        Вычисляет стандартное отклонение оценок по каждому издателю на основе всех чанков.
        """
        publisher_stats = defaultdict(lambda: {'n': 0, 'sum': 0.0, 'sum_sq': 0.0})

        for chunk in chunks:
            sub = chunk[['Metadata.Publishers', 'Metrics.Review Score']].copy()
            sub['Metrics.Review Score'] = pd.to_numeric(sub['Metrics.Review Score'], errors='coerce')
            sub = sub.dropna()

            for publisher, group in sub.groupby('Metadata.Publishers'):
                scores = group['Metrics.Review Score']
                n = len(scores)
                s = scores.sum()
                s2 = (scores ** 2).sum()

                publisher_stats[publisher]['n'] += n
                publisher_stats[publisher]['sum'] += s
                publisher_stats[publisher]['sum_sq'] += s2

        # Финальный расчёт статистик
        results = {}
        for pub, stats in publisher_stats.items():
            n, s, s2 = stats['n'], stats['sum'], stats['sum_sq']
            if n < 2:
                continue  # std не определён

            mean = s / n
            # Несмещённая дисперсия
            variance = (s2 - (s ** 2) / n) / (n - 1)
            if variance < 0:
                variance = 0.0
            std = math.sqrt(variance)

            results[pub] = {
                'n': n,
                'mean': mean,
                'std': std,
            }

        df = pd.DataFrame.from_dict(results, orient='index')
        if not df.empty:
            df = df[['n', 'mean', 'std']]
        yield df

    @staticmethod
    def get_top_bottom_publishers(
            stats_df: pd.DataFrame,
            top_n: int = 3
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Возвращает топ-N с наибольшим и наименьшим std"""
        sorted_df = stats_df.sort_values(by='std')
        bottom = sorted_df.head(top_n)
        top = sorted_df.tail(top_n).iloc[::-1]  # от большего к меньшему
        return top, bottom

    @staticmethod
    def plot_publisher_variance(stats_df: pd.DataFrame, top_n: int = 3):
        """Столбчатая диаграмма: 3 издателя с max и 3 с min std (если хватает данных)."""
        from matplotlib import pyplot as plt
        import numpy as np

        if stats_df.empty:
            print("Нет данных для визуализации.")
            return

        # Гарантия наличия нужных колонок
        required_cols = {'mean', 'std', 'n'}
        if not required_cols.issubset(stats_df.columns):
            print("Некорректный формат stats_df.")
            return

        top_n = min(top_n, len(stats_df))
        top, bottom = SecondPipelines.get_top_bottom_publishers(stats_df, top_n=top_n)

        selected_df = pd.concat([top, bottom])
        publishers = selected_df.index.tolist()
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
        for pub, row in top.iterrows():
            print(f"  {pub}: std = {row['std']:.2f}, mean = {row['mean']:.2f}, n = {int(row['n'])}")

        print(f"\n{top_n} издателей с наименьшим разбросом оценок (std):")
        for pub, row in bottom.iterrows():
            print(f"  {pub}: std = {row['std']:.2f}, mean = {row['mean']:.2f}, n = {int(row['n'])}")
