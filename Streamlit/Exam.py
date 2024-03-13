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
    page = st.sidebar.selectbox("Choose a page", ["Homepage","DataPrep","Data Visualisation","GeoMaps","AI Training and predicitons"])

    if page == "Homepage":
        show_homepage()
    elif page == "DataPrep":
        DataPrep.DataPreparation()    
    elif page == "Data Visualisation":
        Graphs.show_graphs()  
    elif page == "GeoMaps":
        Geomaps.show_Geomaps()        
    elif page == "AI Training and predicitons":
        Predictions.Show_Predictions()
    




if __name__ == "__main__":
    main()