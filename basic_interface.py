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
    st.markdown("<h1 style='text-align: center;'>CSC 674 Final Project</h1>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center;'>William Bailey and Nathan Whitener</h1>", unsafe_allow_html=True)

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
            print(data4)
            
            combined = pd.DataFrame({'Tamper':data2, "Inject":data3, "Flip": data4})
            st.table(combined)
            
            # Add logic for path 1 here
        elif selected_option == "Machine":
            data = ppu.machine_load()
    
        
            data2 = pppu.test_poison_demo(data, 'TAMPER', 'MACHINE')
            
            data3 = pppu.test_poison_demo(data, 'INJECT','MACHINE')
            data4 = pppu.test_poison_demo(data, 'FLIP', "MACHINE")
            
            combined = pd.DataFrame({'Tamper':data2, "Inject":data3, "Flip": data4})
            st.table(combined)
        elif selected_option == "Loan":
            data = ppu.loan_load()
            data2 = pppu.test_poison_demo(data, 'TAMPER', 'LOAN')
            data3 = pppu.test_poison_demo(data, 'INJECT','LOAN')
            data4 = pppu.test_poison_demo(data, 'FLIP', "LOAN")
            
            combined = pd.DataFrame({'Tamper':data2, "Inject":data3, "Flip": data4})
            st.table(combined)
        elif selected_option == "Cancer":
            data = ppu.cancer_load()

        
            data2 = pppu.test_poison_demo(data, 'TAMPER', 'CANCER')
            
            data3 = pppu.test_poison_demo(data, 'INJECT','CANCER')
            data4 = pppu.test_poison_demo(data, 'FLIP', "CANCER")
            
            combined = pd.DataFrame({'Tamper':data2, "Inject":data3, "Flip": data4})
            st.table(combined)
            # Add logic for path 3 here

if __name__ == "__main__":
    main()
