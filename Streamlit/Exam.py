# exam.py
import streamlit as st
import Graphs  # This imports the graphs module you've created
import Predictions  # This imports the predictions module you've created
import GeoMaps as GeoMaps
# Function to display the homepage content
def show_homepage():
    st.title('Homepage')
    st.write("Welcome to the Streamlit App!")

# Main function that runs the app
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", ["Homepage", "Generate Graphs","AI Training and predicitons","GeoMaps"])

    if page == "Homepage":
        show_homepage()
    elif page == "Generate Graphs":
        Graphs.show_graphs()  # This calls a function from graphs.py to display its content
    elif page == "AI Training and predicitons":
        Predictions.show_Predictions()
    elif page == "GeoMaps":
        GeoMaps.show_Geomaps()



if __name__ == "__main__":
    main()