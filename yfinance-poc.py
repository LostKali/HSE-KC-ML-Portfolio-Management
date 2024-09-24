import yfinance as yf
import pandas as pd

# Функция для загрузки данных по акциям
def download_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

# Пример использования
ticker = 'AAPL'  # Символ компании (например, Apple)
start_date = '2020-01-01'
end_date = '2023-01-01'

# Загружаем данные
data = download_stock_data(ticker, start_date, end_date)

# Сохраняем данные в CSV файл для дальнейшей обработки
data.to_csv(f'{ticker}_stock_data.csv')

print(data.head())