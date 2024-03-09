import streamlit as st
from streamlit_folium import st_folium
import folium
import json
import pandas as pd
from DataLoader import load_data


def ensure_country_representation(df, n_samples=500, random_state=42):
    # First, ensure at least one sample from each country
    df_represented = df.groupby('Country').apply(lambda x: x.sample(n=1, random_state=random_state)).reset_index(drop=True)
    
    # Calculate the number of additional samples needed
    additional_samples = max(0, n_samples - df_represented.shape[0])
    
    # Get indices of rows that were not chosen to ensure representation
    remaining_indices = df.index.difference(df_represented.index)
    
    # Sample additional rows if needed
    if additional_samples > 0 and len(remaining_indices) > 0:
        additional_rows = df.loc[remaining_indices].sample(n=min(len(remaining_indices), additional_samples), random_state=random_state)
        df_sampled = pd.concat([df_represented, additional_rows], ignore_index=True)
    else:
        df_sampled = df_represented
    
    return df_sampled


def show_Geomaps():
    st.title('Geomaps')
    st.write("This page allows you to visualize data on a map based on the selected category and continent.")

    df = load_data()

    continents = ['Europe', 'North America', 'South America', 'Africa', 'Asia']
    selected_continent = st.selectbox('Select a continent:', continents)

    df_continent = df[df['Continent'] == selected_continent]

    # Ensure at least one sample from each country is included
    df_sampled = ensure_country_representation(df_continent, n_samples=500, random_state=42)

    # Aggregating total population by country within the selected continent
    aggregated_population = df_continent.groupby('Country')['Population'].sum().reset_index()

    category_names = df.columns.tolist()
    selected_category = st.selectbox('Select a category:', category_names)

    continent_centers = {
        'Europe': [54, 15], 'North America': [40, -100], 'South America': [-15, -60],
        'Africa': [10, 20], 'Asia': [34, 100]
    }
    center = continent_centers.get(selected_continent, [0, 0])
    m = folium.Map(location=center, zoom_start=4)

    geojson_paths = {
        'Europe': '../GeoMaps/EU map test.json', 'North America': '../GeoMaps/NA Map.json',
        'South America': '../GeoMaps/SA Map.json', 'Africa': '../GeoMaps/Africa Map.json',
        'Asia': '../GeoMaps/Asia Map.json'
    }
    geojson_path = geojson_paths.get(selected_continent)

    try:
        with open(geojson_path, 'r', encoding='utf-8') as f:
            geojson_data = json.load(f)

        folium.Choropleth(
            geo_data=geojson_data,
            name="choropleth",
            data=aggregated_population,  # Use the aggregated population DataFrame
            columns=['Country', 'Population'],
            key_on='feature.properties.admin',  # Make sure this matches the property in your GeoJSON
            fill_color="YlOrRd",
            fill_opacity=0.7,
            line_opacity=0.2,
            legend_name="Population",
        ).add_to(m)
        
    except Exception as e:
        st.error(f'Error loading GeoJSON file for {selected_continent}: {e}')

    # Add markers for each data point based on the selected category
    for index, row in df_sampled.iterrows():
        folium.Marker(
            [row['City Latitude'], row['City Longitude']],
            popup=f"{selected_category}: {row[selected_category]}",
            tooltip=row['Country']  # Changed to 'Country' for a more general tooltip
        ).add_to(m)

    st_folium(m, width=725, height=500)