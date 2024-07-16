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
        st.divider()  # 👈 Draws a horizontal rule
        st.write("")


        ################## Input Table ##################
        with st.expander("Input Table"):
            st.header("Input Table", divider=True)
            st.write("""
                The table shows the original input data.
            """)
            st.dataframe(df)
    

        st.write("")
        st.divider()  # 👈 Draws a horizontal rule
        st.write("")


        ################## Feature Scaling Result##################
        with st.expander("Feature Scaling Results"):
            st.header("Scaled Data Table", divider=True)
            st.write("""
                The table shows the scaled data.
            """)
            X_input_norm = scale_data(df)
            st.dataframe(X_input_norm)

            st.markdown("Min-Max Scaler was used for scaling.")
    

        st.write("")
        st.divider()  # 👈 Draws a horizontal rule
        st.write("")

        
        ################## Feature Selection Result##################
        # Drop columns from input df to match feature engineered X_train set

        with st.expander("Feature Selection Results"):
            st.header("Filtered Data Table", divider=True)
            st.write("""
                The table shows the filtered data colums.
            """)


            X_input_reduced = drop_cols(X_input_norm)
            st.dataframe(X_input_reduced)
            
            st.subheader("Selected columns", divider=True)
            col1, col2 = st.columns(2)
            with col1:
                st.write(X_input_reduced.columns)

            with col2:
                bullet_points = ["Univariate Feature Selection", "Cross-Correlation with other features", "Low Correlation with Target"]
                st.write("Features were selected based on :")
                for point in bullet_points:
                    st.write(f"- {point}")
    
        st.write("")
        st.divider()  # 👈 Draws a horizontal rule
        st.write("")
        

        ################## Select Model ##################
        # Select Available Models:
        with st.expander("Model Selection"):
            st.header("Select Models", divider=True)
            st.write("""
                Select Models from the dropdown menue. Combine multiple models. In case multiple models are selected the result will be the mode of the models. In case of a tie, the value of the first selected model contributing to the tie will be regarded.
            """)

            options = st.multiselect(
            '',
            ['Logistic Regression', 'Random Forrest', 'AdaBoost(RF)', 'SVM'],
            ['AdaBoost(RF)'])
            
            st.write("")
            st.divider()  # 👈 Draws a horizontal rule
            st.write("")

            st.text('Selected Models:')

            for ele in options:  
                st.write(ele)
           
            st.write("")
            st.divider()  # 👈 Draws a horizontal rule
            st.write("")
            
            st.write("Model Performance")

            tab1, tab2, tab3, tab4 = st.tabs(["Accuracy", "Precision", "Sensitivity", "Kappa"])

            with tab1:
                st.subheader("Accuracy", divider=True)
                st.image(str(script_dir)+"/resources/accuracy_scores.png", caption="Accuracy")

            with tab2:
                st.subheader("Precision", divider=True)
                st.image(str(script_dir)+"/resources/precision_scores.png", caption="Precision")

            with tab3:
                st.subheader("Sensitivity", divider=True)
                st.image(str(script_dir)+"/resources/sensitivity.png", caption="Recall")

            with tab4:
                st.subheader("Kappa", divider=True)
                st.image(str(script_dir)+"/resources/cohens_kappa.png", caption="Kappa")
  
    
        st.write("")
        st.divider()  # 👈 Draws a horizontal rule
        st.write("")
        


        ################## Run Prediction ##################

        # Run Prediction:
        with st.expander("Run Prediction"):
            
            
            # Load Selected Models:
            #st.write(ml_model_loader(options))
            ml_list = ml_model_loader(options)
            

            # Run Prediction
            if ml_list:
                st.write(ml_list)
                X_result, X_Pred, X_Pred_Diff = ml_predictor(X_input_reduced, ml_list)
            
                st.header("Result Table", divider=True)
                st.dataframe(X_result)    
                
                col3, col4 = st.columns(2)

                with col3:
                    st.subheader("Prediction Table", divider=True)
                    st.dataframe(X_Pred)

                with col4:
                    st.subheader("Prediction Differences", divider=True)
                    st.dataframe(X_Pred_Diff)
            
                ##Model Performance For Validation File
                if filename == "validation_features.csv":
                    st.metric(label="Cohens-Kappa", value=validation_model_performer(X_result))
            else:
                st.text("Please select a model!")
        ################## Download Resuls ##################

        if ml_list:
        
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