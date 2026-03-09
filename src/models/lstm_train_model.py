import sys
import os
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_path)

from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam
from keras import backend as K

from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold, TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

from data.data_utils import resistance_to_moisture
from config.config import set_config
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

set_config()
vectorized_res_to_moist = np.vectorize(resistance_to_moisture)

# ------------------ Utils ------------------

def dataset_splitter(X, y, train_size=0.8):
    train_size = int(len(X) * train_size)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    return X_train, X_test, y_train, y_test

def train_evaluate_simple_lstm(X_train, y_train, X_test, epochs=30, batch_size=32):
    X_train_reshaped = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test_reshaped = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

    model = Sequential([
        LSTM(16, activation='relu', input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2])),
        Dense(2)  # irrigation + fertilizer
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    model.fit(X_train_reshaped, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

    y_train_preds = model.predict(X_train_reshaped)
    y_test_preds = model.predict(X_test_reshaped)
    return y_train_preds, y_test_preds

def train_and_plot(X, y, prediction_features, title):
    if prediction_features is None or len(prediction_features) == 0:
        print(f"Skipping {title} (no features)")
        return

    pretraining_set_X = X[prediction_features]
    X_train, X_test, y_train, y_test = dataset_splitter(pretraining_set_X, y)

    scaler_X = MinMaxScaler()
    X_train_scale = scaler_X.fit_transform(X_train)
    X_test_scale = scaler_X.transform(X_test)

    scaler_y = MinMaxScaler()
    y_train_scale = scaler_y.fit_transform(y_train)
    y_test_scale = scaler_y.transform(y_test)

    y_train_preds, y_test_preds = train_evaluate_simple_lstm(X_train_scale, y_train_scale, X_test_scale)

    mae = mean_absolute_error(y_test_scale, y_test_preds)
    rmse = np.sqrt(mean_squared_error(y_test_scale, y_test_preds))
    print(f"\n{title}: MAE={mae:.4f}, RMSE={rmse:.4f}")

# ------------------ Data ------------------

efficient_feature_df = pd.read_pickle('../../data/interim/03_data_features.pkl')

basic_features = [
    'Barometer - hPa', 'Temp - C', 'High Temp - C', 'Low Temp - C',
    'Hum - %', 'Dew Point - C', 'Wet Bulb - C', 'Wind Speed - km/h', 'Heat Index - C',
    'THW Index - C', 'Rain - mm', 'Heating Degree Days', 'Cooling Degree Days'
]

lag_features = [col for col in efficient_feature_df.columns if "_lag_" in col]
pca_features = [col for col in efficient_feature_df.columns if "pca_" in col]

print(f'Basic Features: {len(basic_features)}')
print(f'Lag Features: {len(lag_features)}')
print(f'PCA Features: {len(pca_features)}')

feature_set_basic_lag = list(set(basic_features + lag_features))

# ------------------ Sensor 1 ------------------

sensor1_df = efficient_feature_df.copy()
X = sensor1_df.drop(['Sensor1 (Ohms)', 'Sensor2 (Ohms)'], axis=1)

soil_resistance = sensor1_df['Sensor1 (Ohms)'].values
soil_moisture = vectorized_res_to_moist(soil_resistance)

irrigation_need = (soil_moisture < 30).astype(int)
fertilizer_need = (soil_moisture < 40).astype(int)

y = np.column_stack((irrigation_need, fertilizer_need))

train_and_plot(X, y, feature_set_basic_lag, 'Sensor 1: Basic + Lag Features')

# ------------------ Sensor 2 ------------------

sensor2_df = efficient_feature_df.copy()
X2 = sensor2_df.drop(['Sensor2 (Ohms)', 'Sensor1 (Ohms)'], axis=1)

soil_resistance2 = sensor2_df['Sensor2 (Ohms)'].values
soil_moisture2 = vectorized_res_to_moist(soil_resistance2)

irrigation_need2 = (soil_moisture2 < 30).astype(int)
fertilizer_need2 = (soil_moisture2 < 40).astype(int)

y2 = np.column_stack((irrigation_need2, fertilizer_need2))

train_and_plot(X2, y2, feature_set_basic_lag, 'Sensor 2: Basic + Lag Features')

print("\n✅ Training completed successfully (Lightweight LSTM + Multi-task).")