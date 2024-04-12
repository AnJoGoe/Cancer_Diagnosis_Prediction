
################################ FUNCTIONS ###########################

## Libraries
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import scipy
import scipy.stats as st
from scipy.stats import chi2_contingency
from scipy.stats.contingency import association


################################ univariate_feat_selection ###########################
# The function calculates the class mean for each feature and cvalidates if the difference is significant or not

def univariate_feat_selection(df:pd.DataFrame, alpha:int = 0.05):
    """
    The function calculates the class mean for each feature and validates if the difference is significant or not. 
    Features for which the difference is not significant will be stored in a list.
    The list will be returned.
    Required:
        Dataframe with target column called "Diagnosis"
    Optional:
        value for alpha
    """

    df = df.copy()
    alpha =  alpha
    
    univar_drop_cols = []
    univar_drop_p = []
    
    for feat in df.columns:
        if feat != 'Diagnosis':
            mean_0 = df[df['Diagnosis']==0][feat]
            mean_1 = df[df['Diagnosis']==1][feat]
    
            t_statistic, p_value = st.ttest_ind(mean_0, mean_1, equal_var=False)
    
            if p_value > alpha:
                univar_drop_cols.append(feat)
                univar_drop_p.append(p_value)
    
            #print(f"Feature: {feat} with mean benign:{mean_0.mean(): .2f} and mean malignant:{mean_0.mean(): .2f} and p-value {p_value}")
    
    for i in range(len(univar_drop_cols)):
        print(f"The feature >>{univar_drop_cols[i]}<< does NOT show a significant difference between the class means (p-value:{univar_drop_p[i]: .4f})")


    return univar_drop_cols




################################ keep_unique_corr_pairs ###########################

# Function to remove duplicate correlation pairs from a stacked correlation matrix.

def keep_unique_corr_pairs(corr_matrix_stacked:pd.DataFrame):
    """
    Function to remove duplicate correlation pairs from a stacked correlation matrix.
    Returns dataframe with unique correlation pairs
    Required columns:
        feature_1
        feature_2
        correlation
    """

    corr_matrix = corr_matrix_stacked.copy()

    # Empty df to store pairs
    df_corr_unique = pd.DataFrame(columns=['feature_1', 'feature_2', 'correlation'])
    
    # Iterate over df 
    for index, row in corr_matrix_stacked.iterrows():
        #print(row)
        feat1 = row['feature_1']
        feat2 = row['feature_2']
        correl = row['correlation']
    
        # check if corr pair is already in df_corr_unique
        if not (((df_corr_unique['feature_1'] == feat1) & (df_corr_unique['feature_2'] == feat2)).any() or ((df_corr_unique['feature_1'] == feat2) & (df_corr_unique['feature_2'] == feat1)).any()):
            #df_corr_unique = df_corr_unique.append({'feature_1': feat1, 'feature_2': feat2, 'correlation': corr}, ignore_index=True)
    
            # Create a temporary dataframe with the current row
            temp_df = pd.DataFrame([[feat1, feat2, correl]], columns=['feature_1', 'feature_2', 'correlation'])
            
            # Concatenate the temporary dataframe with the new dataframe
            df_corr_unique = pd.concat([df_corr_unique, temp_df], ignore_index=True)

    return df_corr_unique



################################ get_feat_with_high_cross_corr ###########################

# Get feature that has the lowest correlation with the target from the the high correlation feature pair.
def get_feat_with_high_cross_corr(corr_matrix:pd.DataFrame, corr_matrix_stacked:pd.DataFrame):
    """
    Get feature that has the lowest correlation with the target from the the high correlation feature pair.
    Returns list with with features to DROP.
    Required dataframes:
        - corrleation matrix
        - stacked correlation matrix
    Required columns:
        feature_1
        feature_2
        correlation
    """
    # Make a copy of the dataframe
    df_copy = corr_matrix_stacked.copy()
    
    # list with features
    drop_cols_list = []
    
    # Iterate over the dataframe
    while len(df_copy) > 0:
        # Get the row with the lowest correlation
        row = df_copy.iloc[0]  # Select the first row after each iteration
        #print(row)
    
        # Extract the features and their correlations
        feat1 = row['feature_1']
        feat2 = row['feature_2']
        
        # Append the feature with the lowest correlation to the target to lowest_values list
        if corr_matrix['Diagnosis'].loc[feat1] < corr_matrix['Diagnosis'].loc[feat2]:
            #print(feat1,feat2)
            #print(corr_matrix['Diagnosis'].loc[feat1],corr_matrix['Diagnosis'].loc[feat2])
            drop_cols_list.append(feat1)
            df_copy = df_copy[~df_copy[['feature_1', 'feature_2']].isin([feat1]).any(axis=1)]
        else:
            drop_cols_list.append(feat2)
            df_copy = df_copy[~df_copy[['feature_1', 'feature_2']].isin([feat2]).any(axis=1)]
    
    # Return the list of lowest values
    #print("Lowest values:", lowest_values)

    return drop_cols_list



################################ rmv_feat_with_low_corr_to_target ###########################

# Remove features that have the lowest correlation with the target. 
def rmv_feat_with_low_corr_to_target(corr_matrix:pd.DataFrame):
    """
    Remove features that have the lowest correlation with the target. 
    Returns list with with features to KEEP.
    Required dataframes:
        - corrleation matrix
    """

    df_copy = corr_matrix.copy()
    
    keep_cols_list = df_copy[df_copy['Diagnosis'] >= 0.1].index.tolist()


    return keep_cols_list