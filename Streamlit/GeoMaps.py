import streamlit as st
from streamlit_folium import st_folium
import folium
import json
from DataLoader import load_data

def show_Geomaps():
    st.title('Geomaps')
    st.write("This page allows you to visualize data on a map based on the selected category and continent.")

    df = load_data()  # Load your DataFrame

    # Dropdown for continent selection
    continents = ['Europe', 'North America', 'South America', 'Africa', 'Asia']
    selected_continent = st.selectbox('Select a continent:', continents)

    # Filter DataFrame for the selected continent
    df_continent = df[df['Continent'] == selected_continent]

    # Sample 500 random rows from the continent-specific DataFrame
    df_sampled = df_continent.sample(n=500, replace=False, random_state=42) 

    # Fetching all column names for the dropdown options
    category_names = df_sampled.columns.tolist()

    # Creating a dropdown menu for the user to select a category
    if 'AQI Category' in category_names:
        selected_category = st.selectbox('Select a category:', category_names, index=category_names.index('AQI Category'))
    else:
        selected_category = st.selectbox('Select a category:', category_names)

    # Creating a Folium map centered on the selected continent
    continent_centers = {
        'Europe': [54, 15],
        'North America': [40, -100],
        'South America': [-15, -60],
        'Africa': [10, 20],
        'Asia': [34, 100]
    }
    center = continent_centers.get(selected_continent, [0, 0])  # Default to [0, 0] if not found
    m = folium.Map(location=center, zoom_start=4)


    # Define the path to the GeoJSON file for the selected continent
    geojson_paths = {
        'Europe':  '../GeoMaps/EU Map.json',
        'North America': '../GeoMaps/NA Map.json',
        'South America': '../GeoMaps/SA Map.json',
        'Africa': '../GeoMaps/Africa Map.json',
        'Asia': '../GeoMaps/Asia Map.json'
    }
    geojson_path = geojson_paths.get(selected_continent)

    # Load and add the GeoJSON layer for the selected continent
    try:
        with open(geojson_path, 'r', encoding='utf-8') as f:
            geojson_data = json.load(f)

     # Assuming 'Country' or a similar column in df_sampled matches 'name' or 'admin' in geojson_data features
        # And you have a 'Population' column in df_sampled
        folium.Choropleth(
            geo_data=geojson_data,
            name="choropleth",
            data=df_sampled,
            columns=['Country', 'Population'],  # Adjust 'Country' to your actual matching column
            key_on='feature.properties.admin',  # Adjust to the actual key in your GeoJSON data
            fill_color="YlOrRd",
            fill_opacity=0.7,
            line_opacity=0.2,
            legend_name="Population",
        ).add_to(m)
        folium.GeoJson(geojson_data, name='geojson').add_to(m)
    except Exception as e:
        st.error(f'Error loading GeoJSON file for {selected_continent}: {e}')

    # Add markers for each data point based on the selected category
    for index, row in df_sampled.iterrows():
        folium.Marker(
            [row['City Latitude'], row['City Longitude']],
            popup=f"{selected_category}: {row[selected_category]}",
            tooltip=row[selected_category]
        ).add_to(m)

    # Display the map
    st_folium(m, width=725, height=500)

