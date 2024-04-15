import streamlit as st
from backend import *

#from pathlib import Path



st.title("Load Data")

st.write('Please select the pathalogical cancer report file (.csv)')
st.write('See example data table below')

def main():


    ################## Load Data ##################
    
    ### Select File (*.csv)
    with st.container():
        st.header("Select File (*.csv)")
        df = load_data()

    
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
            'Combine up to 2 models!',
            ['Logistic Regression', 'Random Forrest', 'AdaBoost(RF)', 'SVM'],
            ['AdaBoost(RF)'],
            max_selections=2)

            st.text('Selected Models:')
            for ele in options:  
                st.write(ele)

            if len(options) > 1:
                st.write("Set weights for multiple model deployment")

                w1_value = st.slider("Weight Slider for w1", 0.0, 100.0, 50.0, step=0.1)
                w2_value = update_w2_value(w1_value)

                st.write("Weight for w1:", w1_value)
                st.write("Weight for w2:", round(w2_value,1))

            tab1, tab2, tab3, tab4 = st.tabs(["Accuracy", "Precision", "Sensitivity", "Kappa"])

            st.write("Model Performance")
            with tab1:
                st.header("Accuracy")
                st.image("../resources/accuracy_scores.PNG", caption="Accuracy")

            with tab2:
                st.header("Precision")
                st.image("../resources/precision_scores.PNG", caption="Precision")

            with tab3:
                st.header("Sensitivity")
                st.image("../resources/sensitivity.PNG", caption="Recall")

            with tab4:
                st.header("Kappa")
                st.image("../resources/accuracy_scores.PNG", caption="Kappa")






    # Model combination: c1(RF), c2(SVM) > w1 = c1/(c+1c2), w2 = c2/(c1+c2) y_pred = w1 * y_pred_RF + w2 * y_pred_SVM
    ## e.g. c1=75% -> 0.75
    ## Dispaly Performance of trainset on combination
    ## Library in scipy

    # Download Prediciton

    # IF 
if __name__ == '__main__':
    main()