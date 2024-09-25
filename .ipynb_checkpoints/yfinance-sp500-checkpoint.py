import yfinance as yf
import pandas as pd

# Шаг 1: Загрузить список компаний из S&P 500 с Wikipedia
url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
table = pd.read_html(url)  # Получаем все таблицы с указанной страницы
sp500_table = table[0]     # S&P 500 обычно находится в первой таблице

# Извлекаем тикеры компаний
tickers = sp500_table['Symbol'].tolist()

# Шаг 2: Функция для скачивания данных по акциям
def download_stock_data(ticker, start_date, end_date):
    try:
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        stock_data['Ticker'] = ticker  # Добавляем столбец с тикером компании
        return stock_data
    except Exception as e:
        print(f"Не удалось загрузить данные для {ticker}: {e}")
        return None

# Шаг 3: Загрузка данных по всем тикерам S&P 500
start_date = '2020-01-01'
end_date = '2023-01-01'
all_data = []  # Список для хранения всех DataFrame

for ticker in tickers:
    print(f"Загружаем данные для {ticker}...")
    stock_data = download_stock_data(ticker, start_date, end_date)
    if stock_data is not None:
        all_data.append(stock_data)  # Добавляем DataFrame в список

# Шаг 4: Объединение всех данных в один DataFrame
all_data_df = pd.concat(all_data)

# Шаг 5: Сохранение всех данных в CSV файл
all_data_df.to_csv('sp500_historical_data.csv')
print("Все данные загружены и сохранены в файл sp500_historical_data.csv")