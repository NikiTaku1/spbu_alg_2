import pandas as pd
import numpy as np
import random
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

def detect_categorical_columns(df):
    """Identifies categorical columns based on data type."""
    categorical_cols = []
    if df is not None:
        for col in df.columns:
            if pd.api.types.is_object_dtype(df[col]):  # Check if object (string)
                categorical_cols.append(col)
    return categorical_cols

def create_mappings(df, categorical_cols):
    """Creates mappings (dictionaries) for categorical columns."""
    mappings = {}
    for col in categorical_cols:
        unique_values = df[col].unique()
        mappings[col] = {val: i for i, val in enumerate(unique_values)}
    return mappings


def apply_mappings(df, mappings):
    """Applies the mappings to convert categorical columns to numerical."""
    df_mapped = df.copy()
    for col, mapping in mappings.items():
        df_mapped[col] = df_mapped[col].map(mapping)
    return df_mapped


def reverse_mappings(df, mappings):
    """Reverts the mappings (numerical to original labels)."""
    df_reverted = df.copy()
    for col, mapping in mappings.items():
        # Invert the mapping dictionary
        inverse_mapping = {v: k for k, v in mapping.items()}
        if col in df_reverted.columns:  # Handle potential missing columns
            df_reverted[col] = df_reverted[col].map(inverse_mapping)
    return df_reverted


def inject_missing_values(df, percent, cols_to_modify):
    """Injects missing values as NaN into the specified columns."""
    df_missing = df.copy()
    for col in cols_to_modify:
        num_missing = int(len(df) * percent)
        indices = random.sample(range(len(df)), num_missing)
        df_missing.loc[indices, col] = np.nan  # Introduce NaN
    return df_missing


def impute_missing_values(df, method, numeric_cols):
    """Imputes missing values using the specified method."""

    if method == "Median":
        imputer = SimpleImputer(strategy='median')

        for col in numeric_cols:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[[col]] = imputer.fit_transform(df[[col]]) #try imputing each value alone

    elif method == "Linear Regression":
        for col in numeric_cols:
            if df[col].isnull().any() and len(df) > 0:
                if len(df[[col]].dropna()) < 2:
                    continue
                df_temp = df[[col]].copy()
                df_temp['index'] = df.index  # Use index as a feature
                known = df_temp[df_temp[col].notnull()]
                unknown = df_temp[df_temp[col].isnull()]

                if len(known) > 2 and len(unknown) > 0:
                    # Train the model
                    try:
                        # Scale the features
                        scaler = StandardScaler()
                        known[['index']] = scaler.fit_transform(known[['index']])
                        # Train the model
                        model = SGDRegressor(loss='squared_error', penalty=None, max_iter=1000, tol=1e-3)
                        model.fit(known[['index']], known[col])
                        # Predict missing values
                        unknown[['index']] = scaler.transform(unknown[['index']])
                        predicted = model.predict(unknown[['index']])

                        #Get max min values
                        min_val = df[col].min() #get min value
                        max_val = df[col].max() #get max value

                        predicted = np.clip(predicted, min_val, max_val) #Clip to set value
                        df.loc[unknown.index, col] = predicted.astype(int) #Округляем значения
                    except Exception as e:
                        print(e)
    # Hot-Deck Imputation to fill any remaining NaN values
    for col in numeric_cols:
        valid_values = df[col].dropna().values  # Remove NaN values
        if len(valid_values) > 0:  # Check data before imputing values
            df[col] = df[col].fillna(pd.Series(np.random.choice(valid_values, size=len(df))))  # Set if data is correct

    return df