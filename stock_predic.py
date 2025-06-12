# Install the yfinance library
!pip install yfinance

# Import necessary libraries
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


# Select a stock ticker and define the time period
ticker_symbol = 'TSLA' # You can change this to 'AAPL', 'GOOGL', etc.
start_date = '2019-01-01'
end_date = '2024-01-01'

# Load the historical data using yfinance
stock_data = yf.download(ticker_symbol, start=start_date, end=end_date)

# Display the first few rows of the dataset
print(f"Displaying data for {ticker_symbol}")
stock_data.head()


# Create the target variable: the next day's closing price
# We use .shift(-1) to move the 'Close' price of the next day to the current row
stock_data['Target'] = stock_data['Close'].shift(-1)

# Drop the last row, as it will have a NaN value in the 'Target' column
stock_data.dropna(inplace=True)

# Define our features (X) and target (y)
features = ['Open', 'High', 'Low', 'Close', 'Volume']
X = stock_data[features]
y = stock_data['Target']

# Display the last few rows to see the 'Target' column
X.tail()
y.tail()


# Split the data into 80% training and 20% testing
# IMPORTANT: For time-series data, we set shuffle=False
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

print(f"Training set size: {len(X_train)} samples")
print(f"Testing set size: {len(X_test)} samples")

# --- Model 1: Linear Regression ---
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# --- Model 2: Random Forest Regressor ---
# We use RandomForestRegressor as we are predicting a continuous value (price)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)


# Make predictions with both models
lr_predictions = lr_model.predict(X_test)
rf_predictions = rf_model.predict(X_test)

# Evaluate the models
# We will use Mean Squared Error (MSE) and R-squared (R2) score
lr_mse = mean_squared_error(y_test, lr_predictions)
lr_r2 = r2_score(y_test, lr_predictions)

rf_mse = mean_squared_error(y_test, rf_predictions)
rf_r2 = r2_score(y_test, rf_predictions)

print("--- Linear Regression Evaluation ---")
print(f"Mean Squared Error (MSE): {lr_mse:.2f}")
print(f"R-squared (R2) Score: {lr_r2:.4f}")
print("\n")
print("--- Random Forest Evaluation ---")
print(f"Mean Squared Error (MSE): {rf_mse:.2f}")
print(f"R-squared (R2) Score: {rf_r2:.4f}")

# The R2 score is very high for both, indicating that today's price is a very strong
# predictor of tomorrow's price. The Random Forest model is slightly better.


# Create a new DataFrame for plotting the results
results_df = pd.DataFrame({'Actual': y_test, 'Predicted': rf_predictions})
results_df.index = y_test.index # Use the original dates as the index

plt.figure(figsize=(14, 7))
plt.plot(results_df.index, results_df['Actual'], label='Actual Closing Price', color='blue', linewidth=2)
plt.plot(results_df.index, results_df['Predicted'], label='Predicted Closing Price (Random Forest)', color='red', linestyle='--', linewidth=2)

plt.title(f'{ticker_symbol} Stock Price Prediction (Short-Term)', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Stock Price (USD)', fontsize=12)
plt.legend()
plt.grid(True)
plt.show()


# Get the most recent row of data from our original feature set
last_day_features = X.iloc[-1:]

# Use the Random Forest model to predict the next day's closing price
next_day_prediction = rf_model.predict(last_day_features)

print("--- Future Price Prediction ---")
print(f"Most recent data available (for {last_day_features.index[0].date()}):")
print(last_day_features)
print("\n")
print(f"Predicted Closing Price for the next trading day: ${next_day_prediction[0]:.2f}")
