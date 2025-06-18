import pandas as pd
import pandas_ta as ta

print("--- Starting Data Processing from Local File (V4 Features) ---")

# Load the CLEAN data from our local CSV file
try:
    df = pd.read_csv("eurusd_clean_data.csv", index_col="Datetime", parse_dates=True)
except FileNotFoundError:
    print("\n[ERROR] 'eurusd_clean_data.csv' not found.")
    print("Please run the 'download_data.py' script first to create the file.")
    exit()

print("Calculating technical indicators...")
# Existing Indicators
df.ta.sma(length=20, append=True)
df.ta.ema(length=50, append=True)
df.ta.rsi(length=14, append=True)
df.ta.macd(fast=12, slow=26, signal=9, append=True)
df.ta.bbands(length=20, std=2, append=True)

# --- NEW V4 FEATURES ---
print("Adding new V4 features: ATR and Stochastic Oscillator...")
df.ta.atr(length=14, append=True)      # Average True Range
df.ta.stoch(k=14, d=3, smooth_k=3, append=True) # Stochastic Oscillator
# -------------------------

print("Identifying candlestick patterns...")
df.ta.cdl_pattern(name="all", append=True)

print("Cleaning up data...")
df.dropna(inplace=True)

# Save the final, feature-rich dataset
df.to_csv("eurusd_features.csv")

print("\n--- Feature Engineering Complete! ---")
print("V4 dataset with new features saved to 'eurusd_features.csv'")
print("\nFinal Data Head:")
print(df.head())