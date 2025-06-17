import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# --- Part 1: Load and Prepare the Data ---

print("--- Loading Feature-Rich Data ---")
try:
    # Load the features we created in the previous step
    df = pd.read_csv("eurusd_features.csv", index_col="Datetime", parse_dates=True)
except FileNotFoundError:
    print("\n[ERROR] 'eurusd_features.csv' not found.")
    print("Please run 'process_data.py' first to create the feature file.")
    exit()

# --- Part 2: Create the Target Variable (y) ---

print("--- Creating the Target Variable (y) ---")
# The 'target' is what we want to predict.
# We'll predict the price direction in the next N periods.
N_FUTURE_PERIODS = 1

# A small percentage threshold to define 'Hold'
PRICE_THRESHOLD = 0.0005  # 0.05% change

# Calculate the future price change
df['Future_Close'] = df['close'].shift(-N_FUTURE_PERIODS)
df['Price_Change'] = (df['Future_Close'] - df['close']) / df['close']

# Define the target classes
# 2 = Buy (price will increase by more than the threshold)
# 1 = Hold (price change is within the threshold)
# 0 = Sell (price will decrease by more than the threshold)
df['target'] = 1  # Default to 'Hold'
df.loc[df['Price_Change'] > PRICE_THRESHOLD, 'target'] = 2  # Buy
df.loc[df['Price_Change'] < -PRICE_THRESHOLD, 'target'] = 0 # Sell

# Drop rows with NaN values created by the shift operation
df.dropna(inplace=True)

# Select our features (X) and target (y)
features = df.drop(['Future_Close', 'Price_Change', 'target'], axis=1)
target = df['target']

# We need to see the class distribution to ensure it's not skewed
print("\nTarget Class Distribution:")
print(target.value_counts(normalize=True))


# --- Part 3: Scale and Create Sequences ---

print("\n--- Scaling Data and Creating Sequences ---")
# Scale features to be between 0 and 1, which is optimal for neural networks
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_features = scaler.fit_transform(features)

# Create sequences of data for the LSTM model
# The model will look at a sequence of past data to make a prediction.
SEQUENCE_LENGTH = 24  # e.g., look at the last 24 hours of data

X_sequences, y_sequences = [], []
for i in range(len(scaled_features) - SEQUENCE_LENGTH):
    X_sequences.append(scaled_features[i:i + SEQUENCE_LENGTH])
    y_sequences.append(target.iloc[i + SEQUENCE_LENGTH -1]) # Target corresponds to the end of the sequence

X, y = np.array(X_sequences), np.array(y_sequences)


# --- Part 4: Split Data into Training and Testing Sets ---

# Split the data, but DO NOT shuffle for time-series data
# We want to train on older data and test on newer data.
TRAIN_SPLIT = 0.8
split_index = int(len(X) * TRAIN_SPLIT)

X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

print(f"\nTraining data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")


# --- Part 5: Build the LSTM Model with TensorFlow/Keras ---

print("\n--- Building the LSTM Model ---")
model = tf.keras.models.Sequential([
    # Input Layer: Shape is (sequence_length, num_features)
    tf.keras.layers.InputLayer(input_shape=(X_train.shape[1], X_train.shape[2])),
    
    # LSTM Layer 1: The main memory layer. `return_sequences=True` is needed
    # when stacking LSTM layers.
    tf.keras.layers.LSTM(units=100, return_sequences=True),
    tf.keras.layers.Dropout(0.2), # Dropout layer to prevent overfitting
    
    # LSTM Layer 2
    tf.keras.layers.LSTM(units=50),
    tf.keras.layers.Dropout(0.2),
    
    # Dense Layer: A standard fully connected layer for final processing
    tf.keras.layers.Dense(units=32, activation='relu'),
    
    # Output Layer: 3 units for our 3 classes (Sell, Hold, Buy).
    # 'softmax' gives a probability distribution for each class.
    tf.keras.layers.Dense(units=3, activation='softmax')
])

# Compile the model
# `sparse_categorical_crossentropy` is used when the target is an integer (0, 1, 2)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()


# --- Part 6: Train the Model ---

print("\n--- Training the Model ---")
# This is where the model learns from the training data.
# `epochs` is the number of times the model will see the entire training dataset.
# `batch_size` is the number of sequences processed at a time.
history = model.fit(X_train, y_train,
                    epochs=20,  # Start with a smaller number of epochs
                    batch_size=32,
                    validation_split=0.1, # Use 10% of training data for validation
                    shuffle=False) # Important for time series

print("\n--- Model Training Complete! ---")

# Save the trained model for future use
model.save("flux_trade_model.h5")
print("Model saved to 'flux_trade_model.h5'")
