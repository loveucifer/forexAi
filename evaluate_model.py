import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# --- Part 1: Load Model and Re-create Test Data ---

print("--- Loading Pre-trained Model and Data ---")
# Load the trained model
try:
    model = tf.keras.models.load_model("flux_trade_model.h5")
except IOError:
    print("\n[ERROR] 'flux_trade_model.h5' not found.")
    print("Please run 'train_model.py' first to create the model file.")
    exit()

# Load the features dataset to re-create the test set
try:
    df = pd.read_csv("eurusd_features.csv", index_col="Datetime", parse_dates=True)
except FileNotFoundError:
    print("\n[ERROR] 'eurusd_features.csv' not found.")
    print("Please run 'process_data.py' first to create the feature file.")
    exit()


# --- This section is identical to train_model.py to ensure the data is processed in the exact same way ---

# Create the target variable
N_FUTURE_PERIODS = 1
PRICE_THRESHOLD = 0.0005
df['Future_Close'] = df['close'].shift(-N_FUTURE_PERIODS)
df['Price_Change'] = (df['Future_Close'] - df['close']) / df['close']
df['target'] = 1
df.loc[df['Price_Change'] > PRICE_THRESHOLD, 'target'] = 2
df.loc[df['Price_Change'] < -PRICE_THRESHOLD, 'target'] = 0
df.dropna(inplace=True)

# Select features and target
features = df.drop(['Future_Close', 'Price_Change', 'target'], axis=1)
target = df['target']

# Scale features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_features = scaler.fit_transform(features)

# Create sequences
SEQUENCE_LENGTH = 24
X_sequences, y_sequences = [], []
for i in range(len(scaled_features) - SEQUENCE_LENGTH):
    X_sequences.append(scaled_features[i:i + SEQUENCE_LENGTH])
    y_sequences.append(target.iloc[i + SEQUENCE_LENGTH - 1])
X, y = np.array(X_sequences), np.array(y_sequences)

# Split the data into training and testing sets (the exact same split as in training)
TRAIN_SPLIT = 0.8
split_index = int(len(X) * TRAIN_SPLIT)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]


# --- Part 2: Make Predictions and Evaluate ---

print("\n--- Evaluating Model on Unseen Test Data ---")
# Make predictions on the test data
predictions_proba = model.predict(X_test)
# The output is probabilities, so we take the index of the highest probability
predictions = np.argmax(predictions_proba, axis=1)


# --- Part 3: Display Performance Metrics ---

# 1. Accuracy Score
accuracy = accuracy_score(y_test, predictions)
print(f"\nTest Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
print("-" * 50)


# 2. Classification Report
# This shows precision, recall, and f1-score for each class.
# - Precision: Of all the 'Buy' predictions, how many were actually correct?
# - Recall: Of all the actual 'Buys', how many did the model find?
print("Classification Report:")
print(classification_report(y_test, predictions, target_names=['Sell (0)', 'Hold (1)', 'Buy (2)']))
print("-" * 50)


# 3. Confusion Matrix
# This shows where the model got confused.
print("Confusion Matrix:")
cm = confusion_matrix(y_test, predictions)
print(cm)

# For a better visual, let's plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Sell', 'Hold', 'Buy'], yticklabels=['Sell', 'Hold', 'Buy'])
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()