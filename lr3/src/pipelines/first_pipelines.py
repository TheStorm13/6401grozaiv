from typing import Dict, Generator

import matplotlib.pyplot as plt
import pandas as pd


class FirstPipelines:
    @staticmethod
    def extract_sales_data(
            chunks: Generator[pd.DataFrame],
    ) -> Generator[pd.DataFrame]:
        """Извлечение данных о продажах и годе выпуска"""

        for chunk in chunks:
            # Удаляем строки с пропущенными значениями
            clean_chunk = chunk[['Release.Year', 'Metrics.Sales']].dropna()
            yield clean_chunk

    @staticmethod
    def aggregate_sales_by_year(
            chunks: Generator[pd.DataFrame],
    ) -> Generator[pd.Series]:
        """Агрегация продаж по годам"""
        agg_series = pd.Series(dtype='float64')  # Индекс — год, значение — сумма продаж

        for chunk in chunks:
            # Группируем чанк по году
            chunk_agg = chunk.groupby('Release.Year')['Metrics.Sales'].sum()
            # Объединяем с накопленной серией и суммируем
            agg_series = agg_series.add(chunk_agg, fill_value=0.0)

        yield agg_series

    @staticmethod
    def plot_sales_by_year(sales_series: pd.Series) -> None:
        """Визуализация продаж по годам."""
        # Убедимся, что индекс — это годы (int), и отсортируем по году
        sales_series = sales_series.sort_index()

        years = sales_series.index.astype(int)
        sales = sales_series.values

        plt.figure(figsize=(12, 6))
        plt.bar(years, sales, color='skyblue')
        plt.xlabel('Год')
        plt.ylabel('Общие продажи')
        plt.title('Продажи игр по годам')
        plt.xticks(years, rotation=45)
        plt.tight_layout()
        plt.show()

        # Находим лучший и худший годы
        best_year = sales_series.idxmax()
        worst_year = sales_series.idxmin()
        print(f"Лучший год для продаж: {best_year} (продажи: {sales_series[best_year]:.2f})")
        print(f"Худший год для продаж: {worst_year} (продажи: {sales_series[worst_year]:.2f})")