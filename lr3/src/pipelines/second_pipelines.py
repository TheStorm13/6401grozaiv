import math
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


class SecondPipelines:
    @staticmethod
    def extract_review_scores(chunks):
        """Извлечение данных об издателях и оценках"""
        for chunk in chunks:
            clean_chunk = chunk[['Metadata.Publishers', 'Metrics.Review Score']].dropna()
            yield clean_chunk

    @staticmethod
    def calculate_review_stats(chunks):
        """Расчет статистики по оценкам для каждого издателя"""
        publisher_stats = defaultdict(lambda: {'scores': [], 'count': 0})

        for chunk in chunks:
            for publisher, score in zip(chunk['Metadata.Publishers'], chunk['Metrics.Review Score']):
                publisher_stats[publisher]['scores'].append(score)
                publisher_stats[publisher]['count'] += 1

        # Рассчитываем дисперсию и доверительные интервалы
        result = {}
        for publisher, stats in publisher_stats.items():
            if len(stats['scores']) >= 2:  # Нужно как минимум 2 точки для дисперсии
                scores = stats['scores']
                mean = np.mean(scores)
                variance = np.var(scores, ddof=1)  # несмещенная дисперсия
                n = len(scores)
                # 95% доверительный интервал
                se = math.sqrt(variance / n)
                ci_lower = mean - 1.96 * se
                ci_upper = mean + 1.96 * se

                result[publisher] = {
                    'variance': variance,
                    'mean_score': mean,
                    'ci_lower': ci_lower,
                    'ci_upper': ci_upper,
                    'count': n
                }

        yield result

    @staticmethod
    def plot_publisher_variance(publisher_stats):
        """Визуализация разброса оценок по издателям"""
        # Сортируем издателей по дисперсии
        sorted_publishers = sorted(publisher_stats.items(),
                                   key=lambda x: x[1]['variance'],
                                   reverse=True)

        # Берем топ-3 и последние 3
        top_3 = sorted_publishers[:3]
        bottom_3 = sorted_publishers[-3:]

        selected = top_3 + bottom_3
        publishers = [p[0] for p in selected]
        means = [p[1]['mean_score'] for p in selected]
        ci_lower = [p[1]['ci_lower'] for p in selected]
        ci_upper = [p[1]['ci_upper'] for p in selected]

        plt.figure(figsize=(12, 6))
        y_pos = np.arange(len(publishers))

        plt.bar(y_pos, means, yerr=[means[i] - ci_lower[i] for i in range(len(means))],
                capsize=5, alpha=0.7)
        plt.xlabel('Издатель')
        plt.ylabel('Средняя оценка')
        plt.title('3 издателя с наибольшим и 3 с наименьшим разбросом оценок')
        plt.xticks(y_pos, publishers, rotation=45)
        plt.tight_layout()
        plt.show()

        print("3 издателя с наибольшим разбросом оценок:")
        for publisher, stats in top_3:
            print(f"  {publisher}: дисперсия = {stats['variance']:.2f}")

        print("\n3 издателя с наименьшим разбросом оценок:")
        for publisher, stats in bottom_3:
            print(f"  {publisher}: дисперсия = {stats['variance']:.2f}")
