from pathlib import Path
import pickle


import pandas as pd
import pandas.testing as pd_testing
import numpy as np
# ML
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import cohen_kappa_score

# App
import streamlit as st





############## Load Data ##############
def load_data():
    """
    Provude the user with the option to upload a csv File.
    Return pandas  file
    """
    uploaded_file = st.file_uploader("")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df
        
        
        # Extract filename
        filename = uploaded_file.name
        st.session_state.filename = filename

        return df, filename
    else:
         st.error("Please upload file")
         return None, None
    
############## Validate input df ##############
def validate_input(df:pd.DataFrame):
    """
    Validate input data.
    Check if the input file contains the required columns and data types
    """
    if df is None:
        return "Upload Data"
    
    df_val = pd.read_csv(Path(__file__).resolve().parent.parent / "data/support" / "df_vali.csv", usecols=lambda x: x not in ["ID","Diagnosis"]) # usecols: https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html
    st.write("Resolved path:", Path(__file__).resolve().parent.parent / "data/support" / "df_vali.csv") # debugging

    # Sort columns alphabetically
    df_val_sorted = df_val.sort_index(axis=1)
    df_sorted = df.sort_index(axis=1)

    # Check if column names and dtypes are the same
    ## https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.testing.assert_frame_equal.html
    if df_sorted.columns.equals(df_val_sorted.columns) and all(df_sorted.dtypes == df_val_sorted.dtypes):
        return "Table validation passed!"
    elif not (df_sorted.columns.equals(df_val_sorted.columns) and not (all(df_sorted.dtypes == df_val_sorted.dtypes))):
        return "Table validation not passed. Table contains different columns and dtypes"
    elif not (df_sorted.columns.equals(df_val_sorted.columns)):
        return "Table validation not passed. Table contains different columns"
    elif not (all(df_sorted.dtypes == df_val_sorted.dtypes)):
        return "Table validation not passed. Table contains different dtypes"


############## Scale Data ##############

def scale_data(df:pd.DataFrame):
    """
    Scale input data with saved scaler from ML development process
    """

    if df is None:
            return ""
    X_input = df.copy()

    # Load Scaler
    with open(Path(__file__).resolve().parent.parent / "machine_learning/scalers/min_max_scaler.pickle", "rb") as file:
        scaler = pickle.load(file)
    
    # Scaler Input Data
    X_input_norm = scaler.transform(X_input)

    X_input_norm = pd.DataFrame(X_input_norm, columns = X_input.columns)

    return X_input_norm





############## Drop Features ##############

def drop_cols(df:pd.DataFrame):
    """
    Drop columns from the input dataframe to match the feature engineered
    X_train_reduced set
    """
    

    if df is None:
            return ""
    df_filtered = df.copy()

    col_list = pd.read_csv(Path(__file__).resolve().parent.parent / "data/Train_Set" / "X_train_reduced.csv", nrows=0).columns.tolist()
    
    df_filtered = df_filtered.filter(items=col_list)

    return df_filtered



############## Load ML Models ##############

def ml_model_loader(model_list:list):
    """
    The function loads models via a list of model names
    """

    ml_list = []
    

    mapping_dict = {"AdaBoost(RF)":"ada_class.pickle",
               'Logistic Regression':"log_reg.pickle", 
               'Random Forrest':"random_forest.pickle", 
               'SVM':"svm.pickle"}

    """
    for key, value in mapping_dict.items():
        if key in model_list:

            # Load Model
            with open(Path(__file__).parents[1] / "machine_learning/models" / value, "rb") as file:
                value = pickle.load(file)
    
                ml_list.append(value)
    """

    for key in model_list:
        if key in mapping_dict:
            # Load Model
            with open(Path(__file__).resolve().parent.parent / "machine_learning/models" / mapping_dict[key], "rb") as file:
                value = pickle.load(file)
    
                ml_list.append(value)

    return ml_list


############## Run Predictions ##############
def ml_predictor(df:pd.DataFrame, ml_list:list):
    
    if df is None or not ml_list:
        return ""
    
    # Copy of test_set
    X_test = df.copy()
    #st.write(X_test)

    # Initiate empty DataFrame to store predictions
    df_pred = pd.DataFrame()




    # Add predictions to array
    for i, model in enumerate(ml_list):
        pred = model.predict(X_test)
        #st.write(pred)

        # Convert to array to pandas df
        series_pred = pd.Series(pred, name='model_'+str(i))
        

        # Add Series to df_pred   
        df_pred = pd.concat([df_pred, series_pred], axis=1)

    # Get most frequent values: value_counts
    value_counts_df = df_pred.apply(pd.value_counts, axis=1)
    #st.write("value_counts_df", value_counts_df)
    #st.dataframe(value_counts_df.duplicated(keep=False))
    #st.write("idxmax", value_counts_df.idxmax(axis=1))
    #st.write(df_pred.iloc[:,0])
    

    # Get Mode: select first value_count in case of a tie
    mode_pred = value_counts_df.idxmax(axis=1)
    row_ties = value_counts_df.apply(lambda x: x.eq(x.max()), axis=1).sum(axis=1) > 1
    #st.write("row ties", row_ties)
    mode_pred_first = mode_pred.mask(row_ties, 
                                     df_pred.iloc[:,0])
    
    

    #st.write("mode_pred",mode_pred)
    #st.write("mode_pred_first",mode_pred_first)
   

    X_test['prediction'] = mode_pred_first
    


    return X_test, df_pred, df_pred[df_pred.nunique(1).ne(1)]  # https://stackoverflow.com/questions/56061033/compare-multiple-columns-in-a-pandas-dataframe


############## Convert df to csv ##############
# https://docs.streamlit.io/knowledge-base/using-streamlit/how-download-pandas-dataframe-csv

@st.cache_data
def convert_df(df):
   """
   Convert pandas dataframe to csv for download.
   """
   return df.to_csv(index=False).encode('utf-8')


############## Model Performance For Validation File ##############
def validation_model_performer(df:pd.DataFrame):
     
     pred = df['prediction'].copy()

     # Load validation_target
     validation_target = pd.read_csv(Path(__file__).resolve().parent.parent / "data/Validation_Set" / "validation_target.csv") 


     #st.dataframe(validation_target) 
     #st.dataframe(validation_target) 

     # Evaluate Model Performance - Ground Truth (Kappa Coef)
     

     return round(cohen_kappa_score(validation_target, pred), 4)