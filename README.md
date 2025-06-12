# Short-Term Stock Price Prediction

This repository contains a Python script for predicting the next day's closing price of a stock using historical market data and machine learning.

## Task Objective

The primary goal of this project is to build a predictive model that takes historical stock data as input and forecasts the closing price for the subsequent trading day. This serves as an educational exercise in time-series forecasting with machine learning.

## Dataset Used

-   **Source**: Yahoo Finance
-   **Library**: The data is retrieved dynamically using the `yfinance` Python library.
-   **Ticker Symbol**: The example script uses Tesla Inc. (`TSLA`), but this can be easily changed to any other valid stock ticker.
-   **Time Period**: `2019-01-01` to `2024-01-01`.
-   **Features**: `Open`, `High`, `Low`, `Close`, `Volume`
-   **Target Variable**: The `Close` price of the next trading day.

## Models Applied

Two regression models were trained and evaluated for this task:

1.  **Linear Regression**: A simple, baseline model used to capture the linear relationship between the daily stock metrics and the next day's closing price.
2.  **Random Forest Regressor**: A more complex, ensemble learning model (`n_estimators=100`) capable of capturing non-linear patterns and interactions between features.

## Key Results and Findings

The models were trained on 80% of the historical data and evaluated on the most recent 20%.

-   **High Predictive Accuracy**: Both models demonstrated extremely high predictive accuracy on the test set. This indicates that the current day's price metrics (`Open`, `High`, `Low`, `Close`) are very strong predictors of the next day's closing price in the short term.
-   **Model Performance (for TSLA)**:
    -   **Linear Regression**: R² Score ≈ 0.9986
    -   **Random Forest**: R² Score ≈ 0.9988
-   **Conclusion**: The Random Forest Regressor performed marginally better, suggesting some minor non-linear relationships in the data. The prediction plot below visually confirms the model's ability to track the actual closing price with high precision.

![Tesla Stock Prediction Plot](TSLA_prediction_plot.png)

### How to Run the Script

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git
    cd YOUR_REPOSITORY_NAME
    ```

2.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the script:**
    ```bash
    python stock_predictor.py
    ```
    The script will download the data, train the models, print the evaluation metrics, and save a plot named `TICKER_prediction_plot.png`.

---

### **Disclaimer**

**This project is for educational purposes only.** The model is highly simplified and should **not** be used for making real financial trading or investment decisions. It does not account for market sentiment, news events, economic indicators, or other complex factors that heavily influence stock prices.
