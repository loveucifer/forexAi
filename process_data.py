import pandas as pd
import pandas_ta as ta

print("--- Starting Data Processing from Local File ---")

# Load the CLEAN data from our local CSV file
# We set 'Datetime' as the index column
try:
    df = pd.read_csv("eurusd_clean_data.csv", index_col="Datetime", parse_dates=True)
except FileNotFoundError:
    print("\n[ERROR] 'eurusd_clean_data.csv' not found.")
    print("Please run the 'download_data.py' script first to create the file.")
    exit()


print("Calculating technical indicators...")
# The 'df.ta' accessor will now work perfectly
df.ta.sma(length=20, append=True)
df.ta.ema(length=50, append=True)
df.ta.rsi(length=14, append=True)
df.ta.macd(fast=12, slow=26, signal=9, append=True)
df.ta.bbands(length=20, std=2, append=True)

print("Identifying candlestick patterns...")
# This will now find the 'open', 'high', 'low', 'close' columns
# Note: You may need to install TA-Lib for full pattern support, but many work without it.
df.ta.cdl_pattern(name="all", append=True)

print("Cleaning up data...")
df.dropna(inplace=True)

# Save the final, feature-rich dataset
df.to_csv("eurusd_features.csv")

print("\n--- Feature Engineering Complete! ---")
print("Final dataset with all features saved to 'eurusd_features.csv'")
print("\nFinal Data Head:")
print(df.head())