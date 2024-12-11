import model_utils as mu
import poison_utils as pu 
import preprocessing_utils as ppu 
import pandas as pd 
import pipeline_utils as pppu
import streamlit as st

def main():
    # Title for the application
    st.markdown("<h1 style='text-align: center;'>Data Poisoning</h1>", unsafe_allow_html=True)

    # Top section with image and text
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image("poison_2.png",  use_container_width=True)
    with col2:
        st.write("CSC 674 Machine Learning Final Project")
        st.write("Nathan Whitener and William Bailey")

    # Dropdown menu for options
    options = ["Heart", "Cancer", "Loan","Machine"]
    selected_option = st.selectbox("Choose a dataset:", options)

    # Button to confirm selection
    if st.button("Confirm Selection"):
        if selected_option == "Heart":
            data = ppu.heart_load()
        
            data2 = pppu.test_poison_demo(data, 'TAMPER', 'HEART')
            
            data3 = pppu.test_poison_demo(data, 'INJECT','HEART')
            data4 = pppu.test_poison_demo(data, 'FLIP', "HEART")
            
            combined = pd.DataFrame({'Tamper':data2, "Inject":data3, "Flip": data4})
            st.table(combined)
            
            # Add logic for path 1 here
        elif selected_option == "Machine":
            data = ppu.machine_load()
    
        
            data2 = pppu.test_poison_demo(data, 'TAMPER', 'HEART')
            
            data3 = pppu.test_poison_demo(data, 'INJECT','HEART')
            data4 = pppu.test_poison_demo(data, 'FLIP', "HEART")
            
            combined = pd.DataFrame({'Tamper':data2, "Inject":data3, "Flip": data4})
            st.table(combined)
        elif selected_option == "Loan":
            data = ppu.loan_load()

        
            data2 = pppu.test_poison_demo(data, 'TAMPER', 'HEART')
            
            data3 = pppu.test_poison_demo(data, 'INJECT','HEART')
            data4 = pppu.test_poison_demo(data, 'FLIP', "HEART")
            
            combined = pd.DataFrame({'Tamper':data2, "Inject":data3, "Flip": data4})
            st.table(combined)
        elif selected_option == "Cancer":
            data = ppu.cancer_load()

        
            data2 = pppu.test_poison_demo(data, 'TAMPER', 'HEART')
            
            data3 = pppu.test_poison_demo(data, 'INJECT','HEART')
            data4 = pppu.test_poison_demo(data, 'FLIP', "HEART")
            
            combined = pd.DataFrame({'Tamper':data2, "Inject":data3, "Flip": data4})
            st.table(combined)
            # Add logic for path 3 here

if __name__ == "__main__":
    main()
