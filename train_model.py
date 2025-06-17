import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import class_weight
# NEW: Import EarlyStopping
from tensorflow.keras.callbacks import EarlyStopping

print("--- Loading Feature-Rich Data ---")
try:
    df = pd.read_csv("eurusd_features.csv", index_col="Datetime", parse_dates=True)
except FileNotFoundError:
    print("\n[ERROR] 'eurusd_features.csv' not found.")
    print("Please run 'process_data.py' first.")
    exit()

# This part remains the same
# ... (code for creating target, scaling, sequencing, splitting) ...
print("--- Creating the Target Variable (y) ---")
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

print("\n--- Scaling Data and Creating Sequences ---")
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
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

print("\n--- Calculating Class Weights to handle imbalance ---")
class_weights_calculated = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = {i: class_weights_calculated[i] for i in range(len(class_weights_calculated))}
print("Calculated Weights:", class_weights)

print("\n--- Building the Bidirectional LSTM Model ---")
model = tf.keras.models.Sequential([
    tf.keras.layers.InputLayer(input_shape=(X_train.shape[1], X_train.shape[2])),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=100, return_sequences=True)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=50)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=3, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

# --- NEW: Define the EarlyStopping callback ---
# This will monitor the validation loss and stop if it doesn't improve for 5 epochs.
# 'restore_best_weights=True' ensures we keep the model from its best epoch.
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
# --------------------------------------------

print("\n--- Training the Model (Longer session with Early Stopping) ---")
history = model.fit(X_train, y_train,
                    # NEW: Increased epochs to give the model more time to learn
                    epochs=100,
                    batch_size=32,
                    validation_split=0.1,
                    shuffle=False,
                    class_weight=class_weights,
                    # NEW: Add the callback to the training process
                    callbacks=[early_stopping])

print("\n--- Model Training Complete! ---")
model.save("flux_trade_model_v3.keras")
print("Model saved to 'flux_trade_model_v3.keras'")
