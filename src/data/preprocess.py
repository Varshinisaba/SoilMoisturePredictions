import sys
import os

# add project root to path
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(root_path)

import pandas as pd
import numpy as np
from glob import glob
from data.data_utils import set_datetime_as_index
import data.remove_outliers as remove_outliers

from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

from config.config import set_config
set_config()


# --------------------------------------------------------------
# File Loader
# --------------------------------------------------------------

class FileLoader(BaseEstimator, TransformerMixin):

    def __init__(self, pattern):
        self.pattern = pattern

    def fit(self, X=None, y=None):
        return self

    def transform(self, X=None, y=None):

        load_files = glob(self.pattern)

        if not load_files:
            raise ValueError(f"No files found for pattern: {self.pattern}")

        load_file = load_files[0]

        print(f"Loading file: {load_file}")

        with open(load_file, 'r', encoding='utf-8', errors='ignore') as file:
            df = pd.read_csv(file)

        return df


# --------------------------------------------------------------
# Column Cleaning
# --------------------------------------------------------------

class ColumnSpaceCleaner(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X.columns = X.columns.str.strip()
        return X


# --------------------------------------------------------------
# Datetime Index Setter
# --------------------------------------------------------------

class DateTimeIndexSetter(BaseEstimator, TransformerMixin):

    def __init__(self, datetime_column):
        self.datetime_column = datetime_column

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):

        X = set_datetime_as_index(X, self.datetime_column)
        return X


# --------------------------------------------------------------
# Saver
# --------------------------------------------------------------

class Saver(BaseEstimator, TransformerMixin):

    def __init__(self, pickle_name, csv_name):
        self.pickle_name = pickle_name
        self.csv_name = csv_name

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):

        X.to_pickle(self.pickle_name)
        X.to_csv(self.csv_name)

        return X


# --------------------------------------------------------------
# Remove extreme values
# --------------------------------------------------------------

class ExtremeValueRemover(BaseEstimator, TransformerMixin):

    def __init__(self, column_name, max_value):
        self.column_name = column_name
        self.max_value = max_value

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):

        X = X[X[self.column_name] < self.max_value]

        return X


# --------------------------------------------------------------
# Remove outliers with LOF
# --------------------------------------------------------------

class RemoveOutliersWithLOF(BaseEstimator, TransformerMixin):

    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):

        outlier_safe_df, outliers, X_scores = remove_outliers.mark_outliers_lof(
            X, self.columns
        )

        for column in self.columns:
            outlier_safe_df.loc[outlier_safe_df['outlier_lof'], column] = np.nan

        outlier_safe_df.interpolate(method='linear', inplace=True)

        outlier_safe_df.drop('outlier_lof', axis=1, inplace=True)

        return outlier_safe_df


# --------------------------------------------------------------
# PREPROCESS YOUR SMART IRRIGATION DATASET
# --------------------------------------------------------------

def preprocess_smart_irrigation_dataset(file_path):

    print("Loading Smart Irrigation dataset...")

    df = pd.read_csv(file_path)

    # combine date and time
    df['Date & Time'] = pd.to_datetime(df['date'] + " " + df['time'])

    # select useful columns
    df = df[['Date & Time', 'temperature', 'pressure', 'soilmiosture']]

    # set datetime index
    df.set_index('Date & Time', inplace=True)

    # resample every 15 minutes
    df = df.resample('15T').mean()

    # interpolate missing values
    df.interpolate(method='linear', inplace=True)

    # create output folders if not exist
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

    interim_dir = os.path.join(base_dir, "data", "interim")
    processed_dir = os.path.join(base_dir, "data", "processed")

    os.makedirs(interim_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    # save files
    df.to_pickle(os.path.join(interim_dir, "smart_irrigation_processed.pkl"))
    df.to_csv(os.path.join(processed_dir, "smart_irrigation_processed.csv"))

    print("Preprocessing complete!")
    print("Saved processed dataset.")

    return df


# --------------------------------------------------------------
# MAIN
# --------------------------------------------------------------

if __name__ == "__main__":

    # build correct dataset path automatically
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    data_path = os.path.join(base_dir, "data", "raw", "SmartIrrigationDataDerive.csv")

    print("Dataset path:", data_path)

    preprocess_smart_irrigation_dataset(data_path)