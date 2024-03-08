import streamlit as st
from streamlit_folium import st_folium
import folium
import json  # Import json for loading GeoJSON data
from DataLoader import load_data

def show_Geomaps():
    st.title('Geomaps')
    st.write("This page allows you to visualize data on a map based on the selected category.")

    df = load_data()  # Load your DataFrame

    # Filter DataFrame for European countries
    df = df[df['Continent'] == 'Europe']

    # Fetching all column names for the dropdown options
    category_names = df.columns.tolist()

    # Creating a dropdown menu for the user to select a category
    selected_category = st.selectbox('Select a category:', category_names, index=category_names.index('AQI Category'))  # Adjust default index as needed

    # Creating a Folium map centered on Europe
    m = folium.Map(location=[54, 15], zoom_start=4)  # Latitude and longitude roughly centered on Europe

    # Load and add the GeoJSON layer, specifying UTF-8 encoding
    europe_geojson_path = '../GeoMaps/EU Map.json'  # Relative path from your Streamlit app to the GeoJSON file
    with open(europe_geojson_path, 'r', encoding='utf-8') as f:
        europe_geojson = json.load(f)
    folium.GeoJson(europe_geojson, name='geojson').add_to(m)

    # Add markers for each data point based on the selected category
    for index, row in df.iterrows():
        folium.Marker(
            [row['City Latitude'], row['City Longitude']],
            popup=f"{selected_category}: {row[selected_category]}",
            tooltip=row[selected_category]
        ).add_to(m)

    # Display the map
    st_folium(m, width=725, height=500)  # Adjust size as needed
