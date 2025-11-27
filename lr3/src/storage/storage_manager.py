import pandas as pd


class StorageManager:

    @staticmethod
    def read_csv_file_chunks(filename, chunksize=100):
        """Генератор для чтения CSV файла порциями"""
        for chunk in pd.read_csv(filename, chunksize=chunksize):
            yield chunk
    @staticmethod
    def read_csv_file(filename):
        """Чтение всего CSV файла"""
        return pd.read_csv(filename)