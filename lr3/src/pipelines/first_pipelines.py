from collections import defaultdict
from typing import Iterator, Dict, Any, List
import pandas as pd
import matplotlib.pyplot as plt


class FirstPipelines:
    @staticmethod
    def extract_sales_data(
        chunks: Iterator[pd.DataFrame],
    ) -> Iterator[pd.DataFrame]:
        """Извлечение данных о продажах и годе выпуска"""

        for chunk in chunks:
            # Удаляем строки с пропущенными значениями
            clean_chunk = chunk[['Release.Year', 'Metrics.Sales']].dropna()
            yield clean_chunk

    @staticmethod
    def aggregate_sales_by_year(
        chunks: Iterator[pd.DataFrame],
    ) -> Iterator[Dict[int, float]]:
        """Агрегация продаж по годам"""
        sales_by_year = defaultdict(float)

        for chunk in chunks:
            for year, sales in zip(chunk['Release.Year'], chunk['Metrics.Sales']):
                sales_by_year[year] += sales

        yield dict(sales_by_year)

    @staticmethod
    def plot_sales_by_year(
        sales_data: Dict[int, float]
    ) -> None:
        """Визуализация продаж по годам"""
        years = list(sales_data.keys())
        sales = list(sales_data.values())

        plt.figure(figsize=(12, 6))
        plt.bar(years, sales)
        plt.xlabel('Год')
        plt.ylabel('Общие продажи')
        plt.title('Продажи игр по годам')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        # Находим лучший и худший годы
        best_year = max(sales_data, key=sales_data.get)
        worst_year = min(sales_data, key=sales_data.get)
        print(f"Лучший год для продаж: {best_year} (продажи: {sales_data[best_year]:.2f})")
        print(f"Худший год для продаж: {worst_year} (продажи: {sales_data[worst_year]:.2f})")

