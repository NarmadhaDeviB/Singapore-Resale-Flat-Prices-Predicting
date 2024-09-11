import streamlit as st # type: ignore
import numpy as np # type: ignore
from streamlit_option_menu import option_menu # type: ignore
import pickle as pkl
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Singapore Resale Flat Prices Predicting",
                   layout="wide",
                   initial_sidebar_state="auto",
                   menu_items={'About': "This was done by Narmadha Devi B"})

with st.sidebar:
    selected = option_menu(None, ["Home", "Price Prediction"],
                           icons=['house', 'tag'],
                           default_index=0,
                           orientation="vertical",
                           styles={"nav-link-selected": {"background-color": "#EE2536"}})

if selected == "Home":
    st.title(':red[Singapore Resale Flat Prices Predicting]')
    st.subheader(':blue[Domain:] Real Estate')
    st.subheader(':blue[Overview:] The project focus on building a machine learning model to predict the resale prices of flats in Singapore based on historical data. It includes the process as Data Collection and Preprocessing, Feature Engineering, Model Selection and Evaluation. Once the evaluation is completed then develop a user-friendly web application using Streamlit application.')
    st.subheader(':blue[Skills Take Away:] Data Wrangling, EDA(Exploratory Data Analysis), Model Building, Model Deployment')



if selected == "Price Prediction":

    st.title(':red[Price Prediction]')
    

    month_values = {"January" : 1,"February" : 2,"March" : 3,"April" : 4,"May" : 5,"June" : 6,"July" : 7,"August" : 8,"September" : 9,
            "October" : 10 ,"November" : 11,"December" : 12}

    town_values = {'ANG MO KIO' : 0 ,'BEDOK' : 1,'BISHAN' : 2,'BUKIT BATOK' : 3,'BUKIT MERAH' : 4,'BUKIT PANJANG' : 5,'BUKIT TIMAH' : 6,
        'CENTRAL AREA' : 7,'CHOA CHU KANG' : 8,'CLEMENTI' : 9,'GEYLANG' : 10,'HOUGANG' : 11,'JURONG EAST' : 12,'JURONG WEST' : 13,
        'KALLANG/WHAMPOA' : 14,'LIM CHU KANG' : 15,'MARINE PARADE' : 16,'PASIR RIS' : 17,'PUNGGOL' : 18,'QUEENSTOWN' : 19,
        'SEMBAWANG' : 20,'SENGKANG' : 21,'SERANGOON' : 22,'TAMPINES' : 23,'TOA PAYOH' : 24,'WOODLANDS' : 25,'YISHUN' : 26}
    
    flat_type_values = {'1 ROOM': 0,'2 ROOM' : 1,'3 ROOM' : 2,'4 ROOM' : 3,'5 ROOM' : 4,'EXECUTIVE' : 5,'MULTI-GENERATION' : 6}

    flat_model_values = {'2-ROOM' : 0,'3GEN' : 1,'ADJOINED FLAT' : 2,'APARTMENT' : 3,'DBSS' : 4,'IMPROVED' : 5,'IMPROVED-MAISONETTE' : 6,
                'MAISONETTE' : 7,'MODEL A' : 8,'MODEL A-MAISONETTE' : 9,'MODEL A2': 10,'MULTI GENERATION' : 11,'NEW GENERATION' : 12,
                'PREMIUM APARTMENT' : 13,'PREMIUM APARTMENT LOFT' : 14,'PREMIUM MAISONETTE' : 15,'SIMPLIFIED' : 16,'STANDARD' : 17,
                'TERRACE' : 18,'TYPE S1' : 19,'TYPE S2' : 20}


    with st.form('prediction'):

        col1,col2=st.columns(2)

    with col1:

            option_month = st.selectbox('Select the Month',month_values,key=1)

            option_town = st.selectbox('Select the Town',town_values,key = 2)

            option_flat_type = st.selectbox('Select the Flat Type',flat_type_values,key = 3)

            option_flat_model = st.selectbox('Select the Flat Model',flat_model_values,key = 4)

            floor_area_sqm = st.number_input('Enter the Floor Area sqm (min:28.0 & max :366.0)')
            
    with col2:

            year = st.text_input("Enter the year", max_chars=4)
            
            lease_commence_date = st.text_input('Enter the Lease Commence Date',max_chars=4)
            
            storey_start = st.text_input("Enter the Storey start (Min:1 & Max:49)")
            
            storey_end = st.text_input('Enter the Storey end (Min:3 & Max:51)')

            button = st.form_submit_button('PREDICT PRICE',use_container_width=True)

           
    if button:
                
        if not all([option_month, option_town, option_flat_type, option_flat_model, floor_area_sqm, lease_commence_date, storey_start, storey_end]):
            st.error("Please fill in all required fields.")

        else:

            month = month_values[option_month]
            town = town_values[option_town]
            flat_type = flat_type_values[option_flat_type]
            flat_model = flat_model_values[option_flat_model]
            floor_area_sqm_log = np.log(floor_area_sqm)
                
            with open('DT.pkl','rb') as file:
                model_DT = pkl.load(file)

             
            new_sample = np.array([[month, town, flat_type, flat_model, lease_commence_date, year, storey_start, storey_end, floor_area_sqm_log]])
            new_pred = model_DT.predict(new_sample)
            resale_price = np.exp(new_pred[0])
         
            st.subheader(f"Predicted Resale price : :red[{resale_price:.2f}]")