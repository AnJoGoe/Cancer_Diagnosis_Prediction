
# Breast Cancer Diagnosis Prediction


## ABSTRACT
This Streamlit app allows users to upload a dataset containing the following features related to breast tumor characteristics:

- **Additional Variable Information**

    * ID number
    * Diagnosis (M = malignant, B = benign) 3-32)
    * Ten real-valued features are computed for each cell nucleus:

    a) radius (mean of distances from center to points on the perimeter)
    b) texture (standard deviation of gray-scale values)
    c) perimeter
    d) area
    e) smoothness (local variation in radius lengths)
    f) compactness (perimeter^2 / area - 1.0)
    g) concavity (severity of concave portions of the contour)
    h) concave points (number of concave portions of the contour)
    i) symmetry 
    j) fractal dimension ("coastline approximation" - 1)


- **APP Usage**
    * Upload Data: Load a dataset template with the required features.
    * Select Models: Choose one or combine several ML models for prediction.
    * Get Prediction: Models are loaded, and predictions are made for the input data.
    * Download: Download the file with associated predictions for diagnosis.

*Note*:To validate the APP the validation_features file and validation_target (see folder ../data/Validation_Set/) can be used. This data subset has not been used for training or testing in the ML Development process.


## DATASET

<a href="https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic">Breast Cancer Wisconsin (Diagnostic)</a>

<a href="https://github.com/uci-ml-repo/ucimlrepo/tree/main">github: ucimlrepo package</a>


Citation: Wolberg,William, Mangasarian,Olvi, Street,Nick, and Street,W.. (1995). Breast Cancer Wisconsin (Diagnostic). UCI Machine Learning Repository. https://doi.org/10.24432/C5DW2B.



## 1. Pre-Processing
- Jupyter Notebook
- Location: notebooks/01_CDP_Preprocessing.ipynb
- Summary:
    The Pre-Processing step provides a first overview of the dataset, checking for unique and missing values and provides initial descriptive stats. 



## 2. Exploratory Data Analysis
- Jupyter Notebook
- Location: notebooks/02_CDP_EDA.ipynb
- Summary:
    The EDA step provides a first glance at the distribution of the dataset and validates the relationship between feature and target. 

## 3. Machine Learning Model Development
- Jupyter Notebook
- Location: notebooks/03_CDP_ML_Feat_Engin_Model_Dev
- Summary:
    The ML section contains steps for Feature Engineering (scaling, selection), Model Training and Performance validation.
    The following models are deployed:
    * Logistic Regression
    * Random Forest Classifier
    * AdaBoost (Estimator:Random Forest Classifier)
    * Support Vector Machines (SVM)

## 4. Streamlit App
    - 2 Python Scripts:
    - Location 1: app/app.py
    - Location 2: backend.py
    - Abstract:
        The app uses two python scripts app.py containing the Streamlit code defining the interface and backend.py containing the process logic
        For validation of the Streamlit app run the command "Streamlit run app.py" in your command line tool.
        The file "data/Validation_Set/validation_features.csv" can be used to validate the app (using this file provides a Cohens_Kappa metric at the end).

    - Process:
    1. Upload File
        - Only csv files with the required columns and datatypes are allowed
        - The file will be validated against a standard data table
    2. Automatic Scaling and Feature selection
        - The user can validate each step by displaying the data table under each dropdown menue
    3. Select a ML model
        - The user can select one to multiple ML models from a dropdown list
        - If multiple models are selected the mode is taken as the predictive outcome (in case of a tie the first selected model will be taken into account)
    4. Download the results as a csv file.
