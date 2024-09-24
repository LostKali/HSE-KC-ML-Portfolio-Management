import yfinance as yf
import pandas as pd

# Определяем тикер для индекса S&P 500
ticker = "^GSPC"

# Указываем период данных (можно изменить под свои нужды)
start_date = "2020-01-01"
end_date = "2024-01-01"

# Скачиваем данные по индексу S&P 500
sp500_data = yf.download(ticker, start=start_date, end=end_date)

# Сохраняем данные в CSV файл
sp500_data.to_csv('sp500_index_data.csv')

print("Данные по S&P 500 успешно загружены и сохранены в 'sp500_historical_data.csv'")