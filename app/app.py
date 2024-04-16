import streamlit as st
from backend import *

from pathlib import Path


# set parent path of the script
script_dir = Path(__file__).parents[1]

st.title("Cancer Diagnosis Predictor")

st.write('Please select the pathalogical cancer report file for analysis')
st.write('Only *.csv files in the correct format are allowed!')

def main():


    ################## Load Data ##################
    
    ### Select File (*.csv)
    with st.container():
        df, filename = load_data()

    
    if df is not None:

        st.warning(validate_input(df))
          

        st.write("")
        st.write("")


        ################## Input Table ##################
        with st.expander("Input Table"):
            st.header("Input Table")
            st.write("""
                The table shows the original input data.
            """)
            st.dataframe(df)
    

        st.write("")
        st.write("")


        ################## Feature Scaling ##################
        with st.expander("Feature Scaling"):
            st.header("Scaled Data Table")
            st.write("""
                The table shows the scaled data.
            """)
            X_input_norm = scale_data(df)
            st.dataframe(X_input_norm)
    

        st.write("")
        st.write("")

        
        ################## Feature Selection ##################
        # Drop columns from input df to match feature engineered X_train set

        with st.expander("Feature Selection"):
            st.header("Filtered Data Table")
            st.write("""
                The table shows the filtered data colums.
            """)
    
            st.header("Filtered Table")

            X_input_reduced = drop_cols(X_input_norm)
            st.dataframe(X_input_reduced)
   
    
        st.write("")
        st.write("")
        

        ################## Select Model ##################
        # Select Available Models:
        with st.expander("Select Models"):
            st.header("Filtered Data Table")


            options = st.multiselect(
            'Combine multiple models!',
            ['Logistic Regression', 'Random Forrest', 'AdaBoost(RF)', 'SVM'],
            ['AdaBoost(RF)'])

            st.text('Selected Models:')

            for ele in options:  
                st.write(ele)


            st.write("Model Performance")

            tab1, tab2, tab3, tab4 = st.tabs(["Accuracy", "Precision", "Sensitivity", "Kappa"])

            with tab1:
                st.header("Accuracy")
                st.image(str(script_dir)+"/resources/accuracy_scores.PNG", caption="Accuracy")

            with tab2:
                st.header("Precision")
                st.image(str(script_dir)+"/resources/precision_scores.PNG", caption="Precision")

            with tab3:
                st.header("Sensitivity")
                st.image(str(script_dir)+"/resources/sensitivity.PNG", caption="Recall")

            with tab4:
                st.header("Kappa")
                st.image(str(script_dir)+"/resources/accuracy_scores.PNG", caption="Kappa")
  
    
        st.write("")
        st.write("")
        


        ################## Run Prediction ##################

        # Run Prediction:
        with st.expander("Run Prediction"):
            
            
            # Load Selected Models:
            #st.write(ml_model_loader(options))
            ml_list = ml_model_loader(options)
            st.write(ml_list)

            # Run Prediction
            X_result, X_Pred, X_Pred_Diff = ml_predictor(X_input_reduced, ml_list)
            
            st.header("Result Table")
            st.dataframe(X_result)    
            
            col1, col2 = st.columns(2)

            with col1:
                st.header("Prediction Table")
                st.dataframe(X_Pred)

            with col2:
                st.header("Prediction Differences")
                st.dataframe(X_Pred_Diff)
        
            ##Model Performance For Validation File
            if filename == "validation_features.csv":
                st.write(validation_model_performer(X_result))

        ################## Download Resuls ##################

        csv = convert_df(X_result)

        st.download_button(
            "Download Results",
            csv,
            "predictions.csv",
            "text/csv",
            key='download-csv'
            )   



        


if __name__ == '__main__':
    main()