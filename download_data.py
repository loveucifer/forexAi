import yfinance as yf
import pandas as pd

# Define the currency pair and a valid time period
ticker = "EURUSD=X"
start_date = "2024-06-01"
end_date = "2025-06-17"

print("--- Starting Data Download ---")
# Download the data using yfinance
data = yf.download(ticker, start=start_date, end=end_date, interval="1h")

if data.empty:
    print(f"No data found for {ticker}. Please check the ticker symbol or date range.")
else:
    # THE FIX: Select the correct level (0) of the column index
    data.columns = data.columns.get_level_values(0)
    
    # Best Practice: Ensure standard column names (lowercase) for compatibility
    data.rename(columns={
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume'
    }, inplace=True)

    # Ensure the standard OHLCV order
    data = data[['open', 'high', 'low', 'close', 'volume']]

    # Save the cleaned data to a new CSV file
    data.to_csv("eurusd_clean_data.csv")
    
    print("\n--- Data Download and Cleaning Complete! ---")
    print("Clean data saved to 'eurusd_clean_data.csv'")
    print("\nClean Data Head:")
    print(data.head())