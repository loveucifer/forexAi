import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

print("--- Loading Pre-trained Model and Data ---")
try:
    # Make sure to load the V2 model
    model = tf.keras.models.load_model("flux_trade_model_v5.keras")
except IOError:
    print("\n[ERROR] 'flux_trade_model_v2.keras' not found.")
    print("Please run 'train_model.py' first to create the model file.")
    exit()

try:
    df = pd.read_csv("eurusd_features.csv", index_col="Datetime", parse_dates=True)
except FileNotFoundError:
    print("\n[ERROR] 'eurusd_features.csv' not found.")
    print("Please run 'process_data.py' first.")
    exit()

# This section re-creates the data exactly as it was for training
N_FUTURE_PERIODS = 1
PRICE_THRESHOLD = 0.0005
df['Future_Close'] = df['close'].shift(-N_FUTURE_PERIODS)
df['Price_Change'] = (df['Future_Close'] - df['close']) / df['close']
df['target'] = 1
df.loc[df['Price_Change'] > PRICE_THRESHOLD, 'target'] = 2
df.loc[df['Price_Change'] < -PRICE_THRESHOLD, 'target'] = 0
df.dropna(inplace=True)

features = df.drop(['Future_Close', 'Price_Change', 'target'], axis=1)
target = df['target']

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_features = scaler.fit_transform(features)

SEQUENCE_LENGTH = 24
X_sequences, y_sequences = [], []
for i in range(len(scaled_features) - SEQUENCE_LENGTH):
    X_sequences.append(scaled_features[i:i + SEQUENCE_LENGTH])
    y_sequences.append(target.iloc[i + SEQUENCE_LENGTH - 1])
X, y = np.array(X_sequences), np.array(y_sequences)

TRAIN_SPLIT = 0.8
split_index = int(len(X) * TRAIN_SPLIT)
X_test = X[split_index:]
y_test = y[split_index:]

print("\n--- Evaluating Model on Unseen Test Data ---")
predictions_proba = model.predict(X_test)
predictions = np.argmax(predictions_proba, axis=1)

print("\n--- Performance Metrics ---")
accuracy = accuracy_score(y_test, predictions)
print(f"Test Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
print("-" * 50)

print("Classification Report:")
print(classification_report(y_test, predictions, target_names=['Sell (0)', 'Hold (1)', 'Buy (2)']))
print("-" * 50)

print("Confusion Matrix:")
cm = confusion_matrix(y_test, predictions)
print(cm)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Sell', 'Hold', 'Buy'], yticklabels=['Sell', 'Hold', 'Buy'])
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()