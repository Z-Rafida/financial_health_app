import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from config import (
    DATA_PATH, TARGET_COL, DROP_COLS,
    CATEGORICAL_COLS, NUMERICAL_COLS,
    TARGET_MAPPING, TEST_SIZE, RANDOM_STATE
)


def load_data(path=DATA_PATH):
    """Load the SME financial health dataset."""
    # TODO: Read the CSV file and return a DataFrame
    df = pd.read_csv(path)
    return df



def clean_data(df):
    df = df.copy()

    # TODO: Step 1 — drop ID column
    df = df.drop(columns = DROP_COLS, errors= 'ignore')

    # TODO: Step 2 — fill missing numerical with median
    df[NUMERICAL_COLS] = df[NUMERICAL_COLS].fillna(df[NUMERICAL_COLS].median())

    # TODO: Step 3 — fill missing categorical with 'Unknown'
    df[CATEGORICAL_COLS] = df[CATEGORICAL_COLS].fillna('Unknown')

    return df


def encode_features(df):
    """
    Encode all categorical columns using LabelEncoder.
    """
    df = df.copy()
    le = LabelEncoder()

    # TODO: Loop through CATEGORICAL_COLS and encode each one
    for col in CATEGORICAL_COLS:
        df[col] = le.fit_transform(df[col])
    return df


def prepare_features(df):
    """
    Full pipeline — returns X_train, X_test, y_train, y_test.
    Encode target: Low=0, Medium=1, High=2
    """
    # TODO: Step 1 — clean
    df = clean_data(df)

    # TODO: Step 2 — encode target using TARGET_MAPPING
    df[TARGET_COL] = df[TARGET_COL].map(TARGET_MAPPING)

    # Hint: df[TARGET_COL] = df[TARGET_COL].map(TARGET_MAPPING)
    # TODO: Step 3 — encode features
    df = encode_features(df)

    # TODO: Step 4 — split X and y, then train_test_split with stratify=y
    X = df.drop(columns = [TARGET_COL])
    y = df[TARGET_COL]

    #Train test split
    X_train, X_test, y_train, y_test =train_test_split (
        X, y, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_STATE, 
        stratify=y
    )
    return X_train, X_test , y_train , y_test
   

