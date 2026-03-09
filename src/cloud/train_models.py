import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input

# -------------------------
# Load Data
# -------------------------

crop_data = pd.read_csv("data/cropdata_updated.csv")
sensor_data = pd.read_csv("data/SmartIrrigationDataDerive.csv")

# -------------------------
# RANDOM FOREST MODEL
# -------------------------

le = LabelEncoder()

crop_data['soil_type'] = le.fit_transform(crop_data['soil_type'])
crop_data['Seedling Stage'] = le.fit_transform(crop_data['Seedling Stage'])
crop_data['crop ID'] = le.fit_transform(crop_data['crop ID'])

X = crop_data[['soil_type','Seedling Stage','MOI','temp','humidity']]
y = crop_data['result']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

rf = RandomForestClassifier(n_estimators=100)

rf.fit(X_train, y_train)

pred = rf.predict(X_test)

print("Random Forest Accuracy:", accuracy_score(y_test, pred))

# -------------------------
# LSTM MODEL
# -------------------------

moisture = sensor_data['soilmiosture'].values

sequence_length = 5

X_seq = []
y_seq = []

for i in range(len(moisture) - sequence_length):
    X_seq.append(moisture[i:i+sequence_length])
    y_seq.append(moisture[i+sequence_length])

X_seq = np.array(X_seq)
y_seq = np.array(y_seq)

X_seq = X_seq.reshape((X_seq.shape[0], X_seq.shape[1], 1))

# Build LSTM model
model = Sequential([
    Input(shape=(sequence_length,1)),
    LSTM(16),
    Dense(1)
])

model.compile(loss='mse', optimizer='adam')

model.fit(X_seq, y_seq, epochs=10, batch_size=8)

print("LSTM training completed")

# -------------------------
# Save Models
# -------------------------

import joblib

joblib.dump(rf, "models/random_forest_model.pkl")
model.save("models/lstm_model.h5")

print("Models saved successfully")

# -------------------------
# Convert to TensorFlow Lite (for ESP32 edge deployment)
# -------------------------

import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Fix for LSTM conversion
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]

converter._experimental_lower_tensor_list_ops = False

tflite_model = converter.convert()

with open("models/lstm_model.tflite", "wb") as f:
    f.write(tflite_model)

print("Lightweight ESP32 model created")