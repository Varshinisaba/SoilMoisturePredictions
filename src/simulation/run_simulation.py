import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

print("SMART IRRIGATION SYSTEM SIMULATION")

# Load trained models
rf = joblib.load("models/random_forest_model.pkl")
lstm = load_model("models/lstm_model.h5", compile=False)

# Load dataset
sensor_data = pd.read_csv("data/SmartIrrigationDataDerive.csv")

moisture = sensor_data['soilmiosture'].values

sequence_length = 5

# ---------------- EDGE NODE ----------------
print("\nEDGE NODE: Predicting soil moisture using LSTM")

last_sequence = moisture[-sequence_length:]

X = np.array(last_sequence).reshape((1,sequence_length,1))

predicted_moisture = lstm.predict(X)

predicted_moisture = predicted_moisture[0][0]

print("Predicted Soil Moisture:", predicted_moisture)

# ---------------- FOG NODE ----------------
print("\nFOG NODE: Making irrigation decision")

# Example environmental values
temperature = 30
humidity = 70

sample = np.array([[0,0,predicted_moisture,temperature,humidity]])

decision = rf.predict(sample)

if decision[0] == 1:
    print("IRRIGATION REQUIRED")
else:
    print("NO IRRIGATION NEEDED")

print("\nSimulation Complete")