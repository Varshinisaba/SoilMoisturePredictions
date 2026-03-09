import joblib
import numpy as np

rf = joblib.load("models/random_forest_model.pkl")

def irrigation_decision(temp,humidity,moisture):

    sample = np.array([[0,0,moisture,temp,humidity]])

    result = rf.predict(sample)

    if result[0] == 1:
        print("Irrigation Required")
    else:
        print("No Irrigation Needed")

irrigation_decision(30,70,40)