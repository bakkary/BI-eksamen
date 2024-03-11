# exam.py
import streamlit as st
import Graphs  # This imports the graphs module you've created
import Predictions  # This imports the predictions module you've created
import Geomaps
import DataPrep
# Function to display the homepage content
def show_homepage():
    st.title('Homepage')
    st.write('This Project is an analasys of pollution data in different countries and some mesurements for different categories of pollution.')
    
    st.write("Made by: Tobias, XiaoXuan, Andreas og Chris")

# Main function that runs the app
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", ["Homepage", "Generate Graphs","AI Training and predicitons","GeoMaps","DataPrep"])

    if page == "Homepage":
        show_homepage()
    elif page == "Generate Graphs":
        Graphs.show_graphs()  
    elif page == "AI Training and predicitons":
        Predictions.show_Predictions()
    elif page == "GeoMaps":
        Geomaps.show_Geomaps()
    elif page == "DataPrep":
        DataPrep.DataPreparation()



if __name__ == "__main__":
    main()