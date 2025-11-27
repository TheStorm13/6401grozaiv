from lr3.config import PROJECT_ROOT
from lr3.src.pipelines.first_pipelines import FirstPipelines
from lr3.src.pipelines.second_pipelines import SecondPipelines
from lr3.src.pipelines.third_pipelines import ThirdPipelines
from lr3.src.storage.storage_manager import StorageManager


def main(filename):
    # Задача 1: Агрегация продаж по годам
    chunks = StorageManager.read_csv_file_chunks(filename)
    sales_chunks = FirstPipelines.extract_sales_data(chunks)
    sales_aggregated = FirstPipelines.aggregate_sales_by_year(sales_chunks)
    sales_data = next(sales_aggregated)
    FirstPipelines.plot_sales_by_year(sales_data)

    # Задача 2: Разброс оценок по издателям
    chunks = StorageManager.read_csv_file_chunks(filename)
    clean_chunks = SecondPipelines.extract_review_scores(chunks)
    stats = SecondPipelines.calculate_publisher_std(clean_chunks)
    stats_data = next(stats)
    SecondPipelines.plot_publisher_variance(stats_data)

    # Задача 3: Количество игр по рейтингам и годам
    chunks = StorageManager.read_csv_file_chunks(filename)
    rating_chunks = ThirdPipelines.extract_rating_data(chunks)
    rating_counts = ThirdPipelines.count_games_by_rating_year(rating_chunks)
    rating_data = next(rating_counts)
    ThirdPipelines.plot_rating_trends(rating_data)


if __name__ == "__main__":
    filename = PROJECT_ROOT / "video_games.csv"
    main(filename)
