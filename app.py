import streamlit as st
import streamlit.components.v1 as stc
from ml_app import run_ml_app
from visualisasi import visualisasi


html_temp = """
            <div style="background-color:#1f77b4;padding:10px;border-radius:10px;">
		    <h1 style="color:white;text-align:center;font-family: 'Arial', sans-serif;">Cyberbullying Prediction App </h1>
		    <h4 style="color:white;text-align:center;">Laskar Lima Muda Team </h4>
		    </div>
            """

desc_temp = """            
            This application is used to determine whether text on the internet constitutes cyberbullying or not
            #### Data Source
            - https://www.kaggle.com/datasets/andrewmvd/cyberbullying-classification
            #### App Content
            - Exploratory Data Analysis
            - Machine Learning Section
            """


def main():
    stc.html(html_temp)

    menu = ['Home', 'Machine Learning', 'Exploratory Data Analysis']
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == 'Home':
        st.subheader("Welcome to Homepage")
        st.markdown(desc_temp)
    elif choice == "Machine Learning":
        st.subheader("Welcome to Machine learning")
        run_ml_app()
    elif choice == "Exploratory Data Analysis":
        st.subheader("Exploratory Data Analysis")
        visualisasi()


if __name__ == '__main__':
    main()
