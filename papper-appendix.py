########## Imports ##########
# Import standard libraries
import pandas as pd
import numpy as np
import random

import os

# For data splitting and preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# For building the LSTM model
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input

# For the genetic algorithm
from deap import base, creator, tools, algorithms

# For model evaluation
from sklearn.metrics import mean_squared_error, mean_absolute_error

# For visualization
import matplotlib.pyplot as plt
%matplotlib inline

# Additional libraries
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")

import joblib
import yfinance as yf
from scipy.stats import shapiro, ttest_rel

import tensorflow.keras.backend as K


########## Part 0: Download Data ##########
# Define the ticker for the S&P 500 index
ticker = "^GSPC"

# Specify the date range for the data (can be modified as needed)
start_date = "2020-01-01"
end_date = "2024-01-01"

# Download the data for the S&P 500 index
sp500_data = yf.download(ticker, start=start_date, end=end_date)

# Save the data to a CSV file
sp500_data.to_csv('sp500_index_data.csv')

print("S&P 500 data successfully downloaded and saved to 'sp500_historical_data.csv'")

# Step 1: Download the list of S&P 500 companies from Wikipedia
url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
table = pd.read_html(url)  # Get all tables from the specified page
sp500_table = table[0]     # The S&P 500 table is usually the first one

# Extract the tickers of the companies
tickers = sp500_table['Symbol'].tolist()

# Step 2: Function to download stock data
def download_stock_data(ticker, start_date, end_date):
    try:
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        stock_data['Ticker'] = ticker  # Add a column with the company ticker
        return stock_data
    except Exception as e:
        print(f"Failed to download data for {ticker}: {e}")
        return None

# Step 3: Download data for all S&P 500 tickers
start_date = '2020-01-01'
end_date = '2023-01-01'
all_data = []  # List to store all DataFrames

for ticker in tickers:
    print(f"Downloading data for {ticker}...")
    stock_data = download_stock_data(ticker, start_date, end_date)
    if stock_data is not None:
        all_data.append(stock_data)  # Add the DataFrame to the list

# Step 4: Combine all the data into one DataFrame
all_data_df = pd.concat(all_data)

# Step 5: Save all the data to a CSV file
all_data_df.to_csv('sp500_historical_data.csv')
print("All data downloaded and saved to the file sp500_historical_data.csv")

########## Main Section ##########

########## Part 1: Data Preparation ##########

data = pd.read_csv('data/sp500_historical_data.csv')
data['Date'] = pd.to_datetime(data['Date'])
data = data[data['Date'].dt.weekday < 5]
full_date_range = pd.date_range(start=data['Date'].min(), end=data['Date'].max(), freq='B')  # 'B' stands for business day

data_pivot = data.pivot_table(index='Date', columns='Ticker', values='Adj Close')
missing_percent = data_pivot.isnull().mean() * 100
tickers_to_remove = missing_percent[missing_percent > 5].index.tolist()
clean_data = data[~data['Ticker'].isin(tickers_to_remove)]

clean_data = clean_data.drop_duplicates()

volume_threshold = clean_data['Volume'].quantile(0.01)
low_volume_data = clean_data[clean_data['Volume'] <= volume_threshold]
low_volume_data.to_csv('data/low_volume_records.csv', index=False)

data_pivot_clean = clean_data.pivot_table(index='Date', columns='Ticker', values='Adj Close')
missing_dates_per_ticker = data_pivot_clean.isnull().sum()
tickers_with_missing_dates = missing_dates_per_ticker[missing_dates_per_ticker > 0].index.tolist()
clean_data = clean_data[~clean_data['Ticker'].isin(tickers_with_missing_dates)]

clean_tickers = clean_data['Ticker'].unique()
cleaned_data = data.sort_values(['Ticker', 'Date']).reset_index(drop=True)
clean_data.to_csv('data/cleaned_sp500_data.csv', index=False)

# Getting the minimum and maximum dates
min_date = data['Date'].min()
max_date = data['Date'].max()

# Total time period
total_days = (max_date - min_date).days

# Split date (80% of the total period)
split_date = min_date + pd.Timedelta(days=int(total_days * 0.8))

print(f"Data split date: {split_date.date()}")
# Training set: data up to and including the split date
train_data = data[data['Date'] <= split_date]

# Test set: data after the split date
test_data = data[data['Date'] > split_date]
print(f"Training set size: {len(train_data)} records")
print(f"Test set size: {len(test_data)} records")

# Save the training set
train_data.to_csv('data/sp500_train_data.csv', index=False)

# Save the test set
test_data.to_csv('data/sp500_test_data.csv', index=False)

# Get the list of unique tickers
tickers = data['Ticker'].unique()

# Initialize a dictionary to store sector information
ticker_sectors = {}

# Retrieve sector information for each ticker
for ticker in tickers:
    try:
        stock_info = yf.Ticker(ticker).info
        sector = stock_info.get('sector', 'Unknown')
        ticker_sectors[ticker] = sector
    except Exception as e:
        print(f"Failed to retrieve data for {ticker}: {e}")
        ticker_sectors[ticker] = 'Unknown'

# Create a DataFrame from the dictionary
sectors_df = pd.DataFrame.from_dict(ticker_sectors, orient='index', columns=['Sector'])

sectors_df.to_csv('data/sectors.csv')

# Get the list of unique tickers before and after filtering
original_tickers = data['Ticker'].unique()
filtered_tickers = clean_data['Ticker'].unique()

# Volume filtering
low_volume_tickers = low_volume_data['Ticker'].unique()

# Create a table with filtering information
summary_data = {
    "Total Tickers Before Filtering": [len(original_tickers)],
    "Tickers Removed Due to Missing Data": [len(tickers_to_remove)],
    "Tickers Removed Due to Low Volume": [len(low_volume_tickers)],
    "Tickers Remaining for Analysis": [len(filtered_tickers)]
}

summary_df = pd.DataFrame(summary_data)

# Save the table to a CSV file
summary_df.to_csv('data/summary_preprocessing.csv', index=False)

# Print the table
print(summary_df)

########## Part 2: LSTM + GA ##########
# Loading the training data
train_data = pd.read_csv('data/sp500_train_data.csv')

# Converting the 'Date' column to datetime format
train_data['Date'] = pd.to_datetime(train_data['Date'])

# Sorting the data
train_data = train_data.sort_values(['Ticker', 'Date']).reset_index(drop=True)

# Defining parameters
TIME_STEPS = 30  # Sequence length for input into LSTM

# Initializing dictionaries to store results
models = {}
scalers = {}
volatility = {}
performance = {}

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

def create_sequences(data, time_steps=TIME_STEPS):
    X = []
    y = []
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps)])
        y.append(data[i + time_steps])
    return np.array(X), np.array(y)

def evaluate_model(individual, X_train, y_train, X_val, y_val):
    # Unpacking hyperparameters
    n_layers = individual[0]
    n_neurons = individual[1]
    dropout_rate = individual[2]
    learning_rate = individual[3]

    # Building the model
    K.clear_session()
    model = Sequential()
    model.add(Input(shape=(X_train.shape[1], X_train.shape[2])))

    if n_layers == 1:
        # Single LSTM layer
        model.add(LSTM(units=n_neurons, return_sequences=False))
        model.add(Dropout(dropout_rate))
    else:
        # First LSTM layer
        model.add(LSTM(units=n_neurons, return_sequences=True))
        model.add(Dropout(dropout_rate))
        # Middle LSTM layers
        for _ in range(n_layers - 2):
            model.add(LSTM(units=n_neurons, return_sequences=True))
            model.add(Dropout(dropout_rate))
        # Last LSTM layer
        model.add(LSTM(units=n_neurons, return_sequences=False))
        model.add(Dropout(dropout_rate))

    model.add(Dense(1))

    # Compiling the model
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss='mean_squared_error', optimizer=optimizer)

    # Training the model
    history = model.fit(X_train, y_train, epochs=5, batch_size=32,
                        validation_data=(X_val, y_val), verbose=0)

    # Evaluating validation loss
    val_loss = history.history['val_loss'][-1]

    return val_loss,

def custom_mutation(individual, indpb):
    for i in range(len(individual)):
        if random.random() < indpb:
            if i == 0:  # n_layers (integer)
                individual[i] = random.randint(1, 2)  # Maximum 2 layers
            elif i == 1:  # n_neurons (integer)
                individual[i] = random.randint(32, 128)  # From 32 to 128 neurons
            elif i == 2:  # dropout_rate (float)
                individual[i] = random.uniform(0.0, 0.3)  # From 0.0 to 0.3
            elif i == 3:  # learning_rate (float)
                individual[i] = random.uniform(0.0005, 0.005)  # Reduced range
    return individual,

def optimize_model_with_ga(X_train, y_train, X_val, y_val, n_generations=3, population_size=5):
    # Defining individual and fitness function
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # Minimize validation error
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()

    # Defining attributes for each hyperparameter
    toolbox.register("attr_n_layers", random.randint, 1, 2)
    toolbox.register("attr_n_neurons", random.randint, 32, 128)
    toolbox.register("attr_dropout_rate", random.uniform, 0.0, 0.3)
    toolbox.register("attr_learning_rate", random.uniform, 0.0005, 0.005)

    # Creating an individual
    toolbox.register("individual", tools.initCycle, creator.Individual,
                     (toolbox.attr_n_layers, toolbox.attr_n_neurons, toolbox.attr_dropout_rate, toolbox.attr_learning_rate), n=1)

    # Creating a population
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Defining evaluation function
    def eval_function(individual):
        return evaluate_model(individual, X_train, y_train, X_val, y_val)

    toolbox.register("evaluate", eval_function)
    toolbox.register("mate", tools.cxUniform, indpb=0.5)
    toolbox.register("mutate", custom_mutation, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Initializing the population
    pop = toolbox.population(n=population_size)

    # Using Hall of Fame to track the best solution
    hof = tools.HallOfFame(1)

    # Parallel evaluation of the population
    def eval_population(population):
        fitnesses = Parallel(n_jobs=-1)(delayed(toolbox.evaluate)(ind) for ind in population)
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

    # Main GA cycle
    for gen in range(n_generations):
        print(f"Generation {gen+1}/{n_generations}")

        # Evaluating the population
        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        if invalid_ind:
            eval_population(invalid_ind)

        # Updating the Hall of Fame
        hof.update(pop)

        # Applying GA operators
        offspring = algorithms.varAnd(pop, toolbox, cxpb=0.5, mutpb=0.2)

        # Evaluating offspring
        invalid_offspring = [ind for ind in offspring if not ind.fitness.valid]
        if invalid_offspring:
            eval_population(invalid_offspring)
        
        # Outputting the best validation error in the current generation
        best = tools.selBest(pop, k=1)[0]
        print(f"Best validation error: {best.fitness.values[0]}")

        # Selecting a new population
        pop = toolbox.select(pop + offspring, k=len(pop))

    # Returning the best found individual
    best_individual = hof[0]
    return best_individual

# Creating directories to save models and plots
if not os.path.exists('models'):
    os.makedirs('models')
if not os.path.exists('plots'):
    os.makedirs('plots')
if not os.path.exists('scalers'):
    os.makedirs('scalers')

tickers = train_data['Ticker'].unique()

# Loop for each ticker
for ticker in tickers:
    if not os.path.isfile(f'models/{ticker}_model.keras'):
        print(f"Processing ticker: {ticker}")

        # Getting data for the ticker
        df_ticker = train_data[train_data['Ticker'] == ticker].sort_values('Date')

        # Using 'Adj Close' as the target variable
        close_prices = df_ticker['Adj Close'].values.reshape(-1, 1)

        # Checking if there is enough data
        if len(close_prices) <= TIME_STEPS:
            print(f"Not enough data for ticker {ticker}")
            continue

        # Scaling the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(close_prices)
        scalers[ticker] = scaler
        
        # Saving the scaler
        scaler_filename = f'scalers/{ticker}_scaler.save'
        joblib.dump(scaler, scaler_filename)

        # Creating sequences
        X, y = create_sequences(scaled_data)

        # Checking if there are enough sequences
        if len(X) == 0:
            print(f"Not enough sequences for ticker {ticker}")
            continue

        # Splitting data into training and testing sets (80% for training)
        split_index = int(0.8 * len(X))
        X_train_full, X_test = X[:split_index], X[split_index:]
        y_train_full, y_test = y[:split_index], y[split_index:]

        # Further splitting the training data into training and validation sets (80% for training)
        val_split_index = int(0.8 * len(X_train_full))
        X_train, X_val = X_train_full[:val_split_index], X_train_full[val_split_index:]
        y_train, y_val = y_train_full[:val_split_index], y_train_full[val_split_index:]

        # Optimizing the model with GA
        best_hyperparams = optimize_model_with_ga(X_train, y_train, X_val, y_val, n_generations=3, population_size=5)

        # Extracting the best hyperparameters
        n_layers = best_hyperparams[0]
        n_neurons = best_hyperparams[1]
        dropout_rate = best_hyperparams[2]
        learning_rate = best_hyperparams[3]

        # Building the final model with the best hyperparameters
        K.clear_session()
        model = Sequential()
        model.add(Input(shape=(X_train_full.shape[1], X_train_full.shape[2])))

        if n_layers == 1:
            # Single LSTM layer
            model.add(LSTM(units=n_neurons, return_sequences=False))
            model.add(Dropout(dropout_rate))
        else:
            # First LSTM layer
            model.add(LSTM(units=n_neurons, return_sequences=True))
            model.add(Dropout(dropout_rate))
            # Middle LSTM layers
            for _ in range(n_layers - 2):
                model.add(LSTM(units=n_neurons, return_sequences=True))
                model.add(Dropout(dropout_rate))
            # Last LSTM layer
            model.add(LSTM(units=n_neurons, return_sequences=False))
            model.add(Dropout(dropout_rate))

        model.add(Dense(1))

        # Compiling the model
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(loss='mean_squared_error', optimizer=optimizer)

        # Training the final model on the full training set
        model.fit(X_train_full, y_train_full, epochs=20, batch_size=32, verbose=0)

        # Evaluating the model on the test data
        y_pred = model.predict(X_test)
        y_pred_inverse = scaler.inverse_transform(y_pred)
        y_test_inverse = scaler.inverse_transform(y_test.reshape(-1, 1))

        # Calculating evaluation metrics
        mse = mean_squared_error(y_test_inverse, y_pred_inverse)
        mae = mean_absolute_error(y_test_inverse, y_pred_inverse)

        performance[ticker] = {'MSE': mse, 'MAE': mae}

        # Saving the model
        model.save(f'models/{ticker}_model.keras')
        models[ticker] = model

        # Visualizing the results
        plt.figure(figsize=(10, 6))
        plt.plot(y_test_inverse, label='Real Price')
        plt.plot(y_pred_inverse, label='Predicted Price')
        plt.title(f'Price Prediction for {ticker}')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        plt.savefig(f'plots/{ticker}_prediction.png')
        plt.close()
    else:
        print(f"Model for ticker: {ticker} already exists")
        
# Converting the performance dictionary into a DataFrame
performance_df = pd.DataFrame.from_dict(performance, orient='index')
performance_df.to_csv('data/model_performance.csv')

########## Part 3: Portfolio Management ##########
# Loading the list of tickers from model files
model_files = os.listdir('models')
clean_tickers = [filename.split('_model.keras')[0] for filename in model_files if filename.endswith('_model.keras')]

models = {}
scalers = {}

for ticker in clean_tickers:
    model_path = f'models/{ticker}_model.keras'
    scaler_path = f'scalers/{ticker}_scaler.save'
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        models[ticker] = load_model(model_path)
        scalers[ticker] = joblib.load(scaler_path)
    else:
        print(f"Model or scaler for ticker {ticker} not found.")
        
# Loading test data
test_data = pd.read_csv('data/sp500_test_data.csv')
test_data['Date'] = pd.to_datetime(test_data['Date'])
test_data = test_data.sort_values(['Date', 'Ticker']).reset_index(drop=True)

# Loading S&P 500 index data
sp500_data = pd.read_csv('data/sp500_index_data.csv')
sp500_data['Date'] = pd.to_datetime(sp500_data['Date'])
sp500_data = sp500_data.sort_values('Date').reset_index(drop=True)

# Getting the unique trading dates from the test dataset
test_dates = sorted(test_data['Date'].unique())

# Selecting every 5th date for portfolio review
review_dates = test_dates[::5]

portfolio_history = []
portfolio_returns = []
index_returns = []
dates = []

TIME_STEPS = 30  # Sequence length for LSTM

# Creating directory for saving predictions
if not os.path.exists('predictions'):
    os.makedirs('predictions')

for i in range(len(review_dates) - 1):
    current_date = review_dates[i]
    print(f"\nPredicting prices for the date: {current_date.date()}")
    
    # Checking if predictions for the current date are already saved
    prediction_file = f'predictions/predictions_{current_date.date()}.csv'
    if os.path.exists(prediction_file):
        print(f"Predictions for {current_date.date()} are already saved.")
        continue
    
    # Dictionary for storing predictions
    predictions_dict = {'Ticker': [], 'Predicted Price': []}
    
    for ticker in clean_tickers:
        if ticker not in models or ticker not in scalers:
            continue

        # Getting data for the ticker up until the current date
        df_ticker = test_data[(test_data['Ticker'] == ticker) & (test_data['Date'] <= current_date)].sort_values('Date')

        # Checking if there is enough data
        if len(df_ticker) < TIME_STEPS + 7:
            continue

        # Getting the last TIME_STEPS of data for prediction
        last_data = df_ticker.iloc[-TIME_STEPS:]
        last_close_prices = last_data['Adj Close'].values.reshape(-1, 1)

        # Scaling the data
        scaler = scalers[ticker]
        scaled_data = scaler.transform(last_close_prices)

        # Preparing input data for the model
        X_input = np.array([scaled_data])

        # Making a 7-day prediction
        # Recursive prediction
        predictions = []
        current_input = X_input.copy()
        for _ in range(7):
            predicted_price = models[ticker].predict(current_input)
            predictions.append(predicted_price[0, 0])

            # Reshaping predicted_price into the correct format
            predicted_price_reshaped = predicted_price.reshape((1, 1, 1))

            # Updating current_input
            current_input = np.concatenate((current_input[:, 1:, :], predicted_price_reshaped), axis=1)

        # Inversely scaling the predicted prices
        predicted_prices = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

        # Current price
        current_price = last_close_prices[-1, 0]

        # Predicted price in 7 days
        predicted_price_7d = predicted_prices[-1, 0]

        # Saving the predicted price
        predictions_dict['Ticker'].append(ticker)
        predictions_dict['Predicted Price'].append(predicted_price_7d)

    # Saving predictions to CSV
    predictions_df = pd.DataFrame(predictions_dict)
    predictions_df.to_csv(prediction_file, index=False)
    print(f"Predictions saved to {prediction_file}")
    
portfolio_history = []
portfolio_returns = []
index_returns = []
dates = []

for i in range(len(review_dates) - 1):
    current_date = review_dates[i]
    next_date = review_dates[i + 1]
    print(f"\nPortfolio review for the date: {current_date.date()}")

    # Loading predicted prices
    prediction_file = f'predictions/predictions_{current_date.date()}.csv'
    if not os.path.exists(prediction_file):
        print(f"Predictions for {current_date.date()} not found. Skipping date.")
        continue

    predictions_df = pd.read_csv(prediction_file)
    predictions_df.set_index('Ticker', inplace=True)

    # Dictionary to store expected return and volatility
    expected_returns = {}

    for ticker in predictions_df.index:
        if ticker not in scalers:
            continue

        # Get data for the ticker up to the current date
        df_ticker = test_data[(test_data['Ticker'] == ticker) & (test_data['Date'] <= current_date)].sort_values('Date')

        # Check if there is enough data
        if len(df_ticker) < TIME_STEPS:
            continue

        # Current price
        current_price = df_ticker.iloc[-1]['Adj Close']

        # Predicted price after 7 days
        predicted_price_7d = predictions_df.loc[ticker]['Predicted Price']

        # Calculate expected return
        expected_return = (predicted_price_7d - current_price) / current_price

        # Calculate volatility based on historical data
        df_ticker['Return'] = df_ticker['Adj Close'].pct_change()
        volatility = df_ticker['Return'].std()

        # Store expected return and volatility
        expected_returns[ticker] = {
            'Expected Return': expected_return,
            'Volatility': volatility
        }

    # Convert the dictionary to a DataFrame
    expected_returns_df = pd.DataFrame.from_dict(expected_returns, orient='index').dropna()

    # Continue only if there is data
    if expected_returns_df.empty:
        print("No available data to form a portfolio for this date.")
        continue

    # Select the top 10 stocks by expected return
    expected_returns_df = expected_returns_df.sort_values(by='Expected Return', ascending=False)
    selected_tickers = expected_returns_df.head(10).index.tolist()

    # Create a DataFrame with the selected stocks
    portfolio_df = expected_returns_df.loc[selected_tickers]

    # Calculate investment fractions using the Kelly criterion
    portfolio_df['Kelly Fraction'] = portfolio_df.apply(
        lambda row: row['Expected Return'] / (row['Volatility'] ** 2) if row['Volatility'] != 0 else 0, axis=1
    )

    # Remove negative values
    portfolio_df['Kelly Fraction'] = portfolio_df['Kelly Fraction'].clip(lower=0)

    # Normalize fractions if the total is greater than 1
    total_fraction = portfolio_df['Kelly Fraction'].sum()
    if total_fraction > 1:
        portfolio_df['Kelly Fraction'] = portfolio_df['Kelly Fraction'] / total_fraction
    else:
        # Handle the case where the total_fraction is zero (all fractions are zero)
        portfolio_df['Investment Fraction'] = 0

    portfolio_df['Investment Fraction'] = portfolio_df['Kelly Fraction']

    # Evaluate the actual returns of the portfolio for the period
    actual_returns = {}
    for ticker in selected_tickers:
        df_ticker = test_data[test_data['Ticker'] == ticker].sort_values('Date').reset_index(drop=True)
        idx_current = df_ticker[df_ticker['Date'] == current_date].index
        idx_next = df_ticker[df_ticker['Date'] == next_date].index
        if len(idx_current) == 0 or len(idx_next) == 0:
            continue
        idx_current = idx_current[0]
        idx_next = idx_next[0]
        current_price = df_ticker.iloc[idx_current]['Adj Close']
        next_price = df_ticker.iloc[idx_next]['Adj Close']
        actual_return = (next_price - current_price) / current_price
        actual_returns[ticker] = actual_return

    # Add the actual return to the portfolio DataFrame
    portfolio_df['Actual Return'] = portfolio_df.index.map(actual_returns).fillna(0)

    # Calculate portfolio return for the period
    portfolio_return = (portfolio_df['Actual Return'] * portfolio_df['Investment Fraction']).sum()
    portfolio_returns.append(portfolio_return)

    # Calculate index return for the period
    sp500_period_data = sp500_data[(sp500_data['Date'] >= current_date) & (sp500_data['Date'] <= next_date)].reset_index(drop=True)
    if not sp500_period_data.empty:
        sp500_start_price = sp500_period_data.iloc[0]['Adj Close']
        sp500_end_price = sp500_period_data.iloc[-1]['Adj Close']
        sp500_return = (sp500_end_price - sp500_start_price) / sp500_start_price
    else:
        sp500_return = 0
    index_returns.append(sp500_return)

    # Save the date
    dates.append(next_date)

    # Save the current portfolio with additional information
    portfolio_info = {
        'Date': current_date,
        'Portfolio': portfolio_df[['Investment Fraction']].to_dict('index'),
        'Return': portfolio_return * 100,
        'SP500 Return': sp500_return * 100
    }
    portfolio_history.append(portfolio_info)

    print(f"Portfolio return for the period: {portfolio_return * 100:.2f}%")
    print(f"S&P 500 return for the period: {sp500_return * 100:.2f}%")

# Creating a DataFrame for comparison
comparison_df = pd.DataFrame({
    'Date': dates,  # The dates, which should already be in the `dates` variable
    'Portfolio Return': portfolio_returns,  # Your portfolio's returns data
    'Index Return': index_returns  # Returns for the S&P 500 or another index
})

# Calculate cumulative return
comparison_df['Portfolio Cumulative Return'] = (1 + comparison_df['Portfolio Return']).cumprod() - 1
comparison_df['Index Cumulative Return'] = (1 + comparison_df['Index Return']).cumprod() - 1

# Output final cumulative return values
final_portfolio_cumulative_return = comparison_df['Portfolio Cumulative Return'].iloc[-1]
final_index_cumulative_return = comparison_df['Index Cumulative Return'].iloc[-1]

print(f"Cumulative return for Kelly + LTSM + GA: {final_portfolio_cumulative_return * 100:.2f}%")
print(f"Cumulative return for S&P 500: {final_index_cumulative_return * 100:.2f}%")

# Visualizing the results
plt.figure(figsize=(12, 6))
plt.plot(comparison_df['Date'], comparison_df['Portfolio Cumulative Return'] * 100, label='Your Portfolio')
plt.plot(comparison_df['Date'], comparison_df['Index Cumulative Return'] * 100, label='S&P 500')
plt.xlabel('Date')
plt.ylabel('Cumulative Return (%)')
plt.title('Portfolio Return Comparison')
plt.legend()
plt.grid(True)
plt.show()

# For the report: save to the 'reports' folder
if not os.path.exists('reports'):
    os.makedirs('reports')

plt.figure(figsize=(12, 6))
plt.plot(comparison_df['Date'], comparison_df['Portfolio Cumulative Return'] * 100, label='Kelly + LTSM + GA Portfolio')
plt.plot(comparison_df['Date'], comparison_df['Index Cumulative Return'] * 100, label='S&P 500')
plt.xlabel('Date')
plt.ylabel('Cumulative Return (%)')
plt.title('Returns Comparison')
plt.legend()
plt.grid(True)

# Save the image
plt.savefig(f'reports/portfolio_return_comparision.png')
plt.close()

print("Chart saved to 'reports/portfolio_return_comparision.png'")

# Calculate weekly returns
comparison_df['Week'] = comparison_df['Date'].dt.to_period('W')  # Convert dates to weekly periods
weekly_returns = comparison_df.groupby('Week').agg({
    'Portfolio Return': 'sum',
    'Index Return': 'sum'
}).reset_index()

# Calculate the mean and standard deviation of weekly returns
mean_weekly_portfolio_return = weekly_returns['Portfolio Return'].mean()
std_weekly_portfolio_return = weekly_returns['Portfolio Return'].std()

mean_weekly_index_return = weekly_returns['Index Return'].mean()
std_weekly_index_return = weekly_returns['Index Return'].std()

# Output mean values and standard deviation of weekly returns
print(f"Mean Weekly Return for Kelly + LTSM + GA: {mean_weekly_portfolio_return * 100:.2f}%")
print(f"Standard Deviation of Weekly Returns for Kelly + LTSM + GA: {std_weekly_portfolio_return * 100:.2f}%")
print(f"Mean Weekly Return for S&P 500: {mean_weekly_index_return * 100:.2f}%")
print(f"Standard Deviation of Weekly Returns for S&P 500: {std_weekly_index_return * 100:.2f}%")

# Saving portfolio history to a CSV file
portfolio_records = []

for entry in portfolio_history:
    date = entry['Date']
    portfolio_return = entry['Return']
    sp500_return = entry['SP500 Return']
    for ticker, info in entry['Portfolio'].items():
        portfolio_records.append({
            'Date': date,
            'Ticker': ticker,
            'Investment Fraction': info['Investment Fraction'],
            'Portfolio Return': portfolio_return,
            'SP500 Return': sp500_return
        })

portfolio_df = pd.DataFrame(portfolio_records)
portfolio_df.to_csv('data/portfolio_history.csv', index=False)
print("Portfolio information saved in 'portfolio_history.csv'")

# Function to calculate maximum drawdown
def max_drawdown(return_series):
    cumulative = (1 + return_series).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    return drawdown

# Function to calculate CAGR
def calculate_cagr(return_series, periods_per_year=52):
    total_return = (1 + return_series).cumprod().iloc[-1] - 1
    num_years = len(return_series) / periods_per_year
    cagr = (1 + total_return) ** (1 / num_years) - 1
    return cagr

# Creating a DataFrame for comparison
comparison_df = pd.DataFrame({
    'Date': dates,  # The dates, which should already be in the `dates` variable
    'Portfolio Return': portfolio_returns,  # Your portfolio's returns data
    'Index Return': index_returns  # Returns for S&P 500 or another index
})

# Calculating cumulative return for both portfolios
comparison_df['Portfolio Cumulative Return'] = (1 + comparison_df['Portfolio Return']).cumprod() - 1
comparison_df['Index Cumulative Return'] = (1 + comparison_df['Index Return']).cumprod() - 1

# Calculating maximum drawdown at each point in time
comparison_df['Portfolio Drawdown'] = max_drawdown(comparison_df['Portfolio Return'])
comparison_df['Index Drawdown'] = max_drawdown(comparison_df['Index Return'])

# Visualizing cumulative returns and drawdowns on one chart
plt.figure(figsize=(14, 8))

# Cumulative return (line chart)
plt.plot(comparison_df['Date'], comparison_df['Portfolio Cumulative Return'] * 100, label='Portfolio Cumulative Return', color='blue')
plt.plot(comparison_df['Date'], comparison_df['Index Cumulative Return'] * 100, label='S&P 500 Cumulative Return', color='orange')

# Drawdowns (bar chart)
plt.bar(comparison_df['Date'], comparison_df['Portfolio Drawdown'] * 100, label='Portfolio Drawdown', color='blue', alpha=0.3)
plt.bar(comparison_df['Date'], comparison_df['Index Drawdown'] * 100, label='S&P 500 Drawdown', color='orange', alpha=0.3)

# Adding a horizontal axis at the 0 level (dark gray color)
plt.axhline(0, color='darkgray', linewidth=1.5)

# Chart settings
plt.title('Cumulative Return and Drawdowns Over Time')
plt.ylabel('Cumulative Return / Drawdown (%)')
plt.xlabel('Date')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Saving the chart to a file
plt.figure(figsize=(14, 8))

# Cumulative return (line chart)
plt.plot(comparison_df['Date'], comparison_df['Portfolio Cumulative Return'] * 100, label='Portfolio Cumulative Return', color='blue')
plt.plot(comparison_df['Date'], comparison_df['Index Cumulative Return'] * 100, label='S&P 500 Cumulative Return', color='orange')

# Drawdowns (bar chart)
plt.bar(comparison_df['Date'], comparison_df['Portfolio Drawdown'] * 100, label='Portfolio Drawdown', color='blue', alpha=0.3)
plt.bar(comparison_df['Date'], comparison_df['Index Drawdown'] * 100, label='S&P 500 Drawdown', color='orange', alpha=0.3)

# Adding a horizontal axis at the 0 level (dark gray color)
plt.axhline(0, color='darkgray', linewidth=1.5)

# Chart settings
plt.title('Cumulative Return and Drawdowns Over Time')
plt.ylabel('Cumulative Return / Drawdown (%)')
plt.xlabel('Date')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('reports/cumulative_and_drawdowns.png')
plt.close()

print("Cumulative return and drawdown chart saved to 'reports/cumulative_and_drawdowns.png'")

# Converting returns into numpy arrays
portfolio_returns_array = np.array(portfolio_returns)
index_returns_array = np.array(index_returns)

# Testing normality of return distributions
stat_portfolio, p_portfolio = shapiro(portfolio_returns_array)
stat_index, p_index = shapiro(index_returns_array)

print(f"P-value of the normality test for portfolio returns: {p_portfolio}")
print(f"P-value of the normality test for S&P 500 returns: {p_index}")

# Selecting the appropriate statistical test
if p_portfolio > 0.05 and p_index > 0.05:
    # Both distributions are normal
    t_stat, p_value = ttest_rel(portfolio_returns_array, index_returns_array)
    print(f"P-value of the paired t-test: {p_value}")
else:
    # Using a non-parametric test
    stat, p_value = wilcoxon(portfolio_returns_array - index_returns_array)
    print(f"P-value of the Wilcoxon test: {p_value}")

# Calmar ratio calculation
def calculate_cagr(returns, periods_per_year):
    total_return = np.prod([1 + r for r in returns]) - 1
    num_periods = len(returns)
    cagr = (1 + total_return) ** (periods_per_year / num_periods) - 1
    return cagr

########## Illustration needs ##########

# Load data from CSV file
df = pd.read_csv('data/portfolio_history.csv')

# Convert the date column to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Get a list of unique dates and tickers
unique_dates = df['Date'].unique()
unique_tickers = df['Ticker'].unique()

# Number of unique dates
num_dates = len(unique_dates)

# Create a unified chart using subplots
fig, ax = plt.subplots(figsize=(10, num_dates * 0.5))

# Convert tickers to indices for the Y-axis
ticker_to_index = {ticker: i for i, ticker in enumerate(unique_tickers)}

# Create a Gantt chart
for i, date in enumerate(unique_dates):
    # Select data for the current date
    date_data = df[df['Date'] == date]

    # Add horizontal bars for each ticker
    for ticker, frac in zip(date_data['Ticker'], date_data['Investment Fraction']):
        ax.barh(ticker_to_index[ticker], frac, left=i, height=0.8, edgecolor='black')

        # Display the value of the investment fraction next to the bar, slightly shifted to the right and rounded to two decimal places
        ax.text(i + frac + 0.01, ticker_to_index[ticker], f'{frac:.2f}', va='center', ha='left', color='black', fontsize=9)

# Setting up axis labels
ax.set_yticks(range(len(unique_tickers)))
ax.set_yticklabels(unique_tickers)
ax.set_xticks(range(len(unique_dates)))
ax.set_xticklabels([date.strftime('%Y-%m-%d') for date in unique_dates], rotation=45)

# Adding title and axis labels
ax.set_xlabel('Time (Dates)')
ax.set_ylabel('Ticker')
ax.set_title('Gantt Chart of Portfolio Rebalancing with Investment Fractions')

# Saving the Gantt chart
plt.tight_layout()
if not os.path.exists('reports'):
    os.makedirs('reports')

plt.savefig('reports/portfolio_rebalancing_gantt.png')
plt.show()

print(f"Gantt chart saved to 'reports/portfolio_rebalancing_gantt.png'")