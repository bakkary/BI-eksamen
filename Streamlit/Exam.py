# exam.py
import streamlit as st
import Graphs  # This imports the graphs module you've created
import Predictions  # This imports the predictions module you've created
import Geomaps
import DataPrep
import placeholder

# Function to display the homepage content
def show_homepage():
    st.title('Homepage')
    st.write('This Project is an analasys of pollution data in different countries and some mesurements for different categories of pollution.')
    
    st.write("Made by: Tobias, XiaoXuan, Andreas og Chris")

# Main function that runs the app
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", ["Homepage", "Data Visualisation","AI Training and predicitons","GeoMaps","DataPrep","placeholder"])

    if page == "Homepage":
        show_homepage()
    elif page == "Data Visualisation":
        Graphs.show_graphs()  
    elif page == "AI Training and predicitons":
        Predictions.show_Predictions()
    elif page == "GeoMaps":
        Geomaps.show_Geomaps()
    elif page == "DataPrep":
        DataPrep.DataPreparation()
    elif page == "placeholder":
        placeholder.show_placeholder()



if __name__ == "__main__":
    main()