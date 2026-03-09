import sys
import os

# get project root
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

sys.path.append(base_dir)

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from config.config import set_config
set_config()


# --------------------------------------------------------------
# Load processed dataset safely
# --------------------------------------------------------------

data_path = os.path.join(base_dir, "data", "processed", "smart_irrigation_processed.csv")

print("Loading processed dataset from:", data_path)

data = pd.read_csv(data_path)

data['Date & Time'] = pd.to_datetime(data['Date & Time'])

data.set_index('Date & Time', inplace=True)

print("Dataset loaded successfully")


# --------------------------------------------------------------
# Normalize data
# --------------------------------------------------------------

features = ['temperature', 'pressure', 'soilmiosture']

scaler = MinMaxScaler()

scaled_data = scaler.fit_transform(data[features])

scaled_df = pd.DataFrame(scaled_data, columns=features, index=data.index)

print("Data normalized")


# --------------------------------------------------------------
# Create LSTM sequences
# --------------------------------------------------------------

window_size = 10

X = []
y = []

values = scaled_df.values

for i in range(len(values) - window_size):

    X.append(values[i:i+window_size])

    y.append(values[i+window_size][2])  # soil moisture prediction


X = np.array(X)
y = np.array(y)

print("Sequence creation complete")

print("X shape:", X.shape)
print("y shape:", y.shape)


# --------------------------------------------------------------
# Save features
# --------------------------------------------------------------

interim_dir = os.path.join(base_dir, "data", "interim")

os.makedirs(interim_dir, exist_ok=True)

np.save(os.path.join(interim_dir, "X_lstm.npy"), X)
np.save(os.path.join(interim_dir, "y_lstm.npy"), y)

scaled_df.to_pickle(os.path.join(interim_dir, "03_data_features.pkl"))

print("Features saved successfully")