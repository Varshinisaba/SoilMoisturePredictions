import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("models/lstm_model.h5")

sensor_data = pd.read_csv("data/SmartIrrigationDataDerive.csv")

moisture = sensor_data['soilmiosture'].values

sequence_length = 5

def predict_next_moisture(data):

    seq = np.array(data[-sequence_length:])
    seq = seq.reshape((1,sequence_length,1))

    prediction = model.predict(seq)

    return prediction[0][0]

pred = predict_next_moisture(moisture)

print("Predicted soil moisture:",pred)