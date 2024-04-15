from pathlib import Path
import pickle


import pandas as pd
import pandas.testing as pd_testing

# ML
from sklearn.preprocessing import MinMaxScaler

# App
import streamlit as st





###### Load Data ##############
def load_data():
    """
    Provude the user with the option to upload a csv File.
    Return pandas  file
    """
    uploaded_file = st.file_uploader("")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df
        
        return df
    
###### Validate input df ##############
def validate_input(df:pd.DataFrame):
    """
    Validate input data.
    Check if the input file contains the required columns and data types
    """
    if df is None:
        return "Upload Data"
    
    df_val = pd.read_csv(Path(__file__).parents[1] / "data/support" / "df_vali.csv", usecols=lambda x: x not in ["ID","Diagnosis"]) # usecols: https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html

    # Sort columns alphabetically
    df_val_sorted = df_val.sort_index(axis=1)
    df_sorted = df.sort_index(axis=1)

    # Check if column names and dtypes are the same
    ## https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.testing.assert_frame_equal.html
    if df_sorted.columns.equals(df_val_sorted.columns) and all(df_sorted.dtypes == df_val_sorted.dtypes):
        return "Table validation passed!"
    elif not (df_sorted.columns.equals(df_val_sorted.columns) and all(df_sorted.dtypes == df_val_sorted.dtypes)):
        return "Table validation not passed. Table contains different columns and dtypes"
    elif not (df_sorted.columns.equals(df_val_sorted.columns)):
        return "Table validation not passed. Table contains different columns"
    elif not (all(df_sorted.dtypes == df_val_sorted.dtypes)):
        return "Table validation not passed. Table contains dtypes"


###### Normalized Data ##############

def scale_data(df:pd.DataFrame):
    """
    Scale input data with saved scaler from ML development process
    """

    if df is None:
            return ""
    X_input = df.copy()

    # Load Scaler
    with open(Path(__file__).parents[1] / "machine_learning/scalers/min_max_scaler.pickle", "rb") as file:
        scaler = pickle.load(file)
    
    # Scaler Input Data
    X_input_norm = scaler.transform(X_input)

    X_input_norm = pd.DataFrame(X_input_norm, columns = X_input.columns)

    return X_input_norm





###### Drop Features ##############

def drop_cols(df:pd.DataFrame):
    """
    Drop columns from the input dataframe to match the feature engineered
    X_train_reduced set
    """
    

    if df is None:
            return ""
    df_filtered = df.copy()

    col_list = pd.read_csv(Path(__file__).parents[1] / "data/Train_Set" / "X_train_reduced.csv", nrows=0).columns.tolist()
    
    df_filtered = df_filtered.filter(items=col_list)

    return df_filtered



###### Update Weigts ##############
def update_w2_value(w1_value):
            """
            Update weights for multiple model deployment
            """
            return 100 - w1_value