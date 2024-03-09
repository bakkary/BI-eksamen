import streamlit as st
from streamlit_folium import st_folium
import folium
import json
import pandas as pd
from DataLoader import load_data

def ensure_one_sample_per_country(df, random_state=42):
    # Ensure one sample from each country
    return df.groupby('Country').apply(lambda x: x.sample(n=1, random_state=random_state)).reset_index(drop=True)

def show_Geomaps():
    st.title('Geomaps')
    st.write("This page allows you to visualize data on a map based on the selected category and continent.")

    df = load_data()

    continents = ['Europe', 'North America', 'South America', 'Africa', 'Asia']
    selected_continent = st.selectbox('Select a continent:', continents)

    df_continent = df[df['Continent'] == selected_continent]
    df_sampled = ensure_one_sample_per_country(df_continent)

    # Letting user choose a category (metric) to visualize
    category_names = df.columns.tolist()
    selected_category = st.selectbox('Select a metric to visualize:', category_names)

    # Aggregating data based on the selected category (metric)
    if df[selected_category].dtype in [int, float]:
        aggregated_data = df_continent.groupby('Country', as_index=False)[selected_category].sum()
    else:
        aggregated_data = df_sampled[['Country', selected_category]].drop_duplicates()

    continent_centers = {
        'Europe': [54, 15], 
        'North America': [40, -100], 
        'South America': [-15, -60],
        'Africa': [10, 20], 
        'Asia': [34, 100]
    }

    center = continent_centers.get(selected_continent, [0, 0])
    m = folium.Map(location=center, zoom_start=4)

    geojson_paths = {
        'Europe': '../GeoMaps/EU Map.json', 
        'North America': '../GeoMaps/NA Map.json',
        'South America': '../GeoMaps/SA Map.json', 
        'Africa': '../GeoMaps/Africa Map.json',
        'Asia': '../GeoMaps/Asia Map.json'
    }

    geojson_path = geojson_paths.get(selected_continent)

    try:
        with open(geojson_path, 'r', encoding='utf-8') as f:
            geojson_data = json.load(f)

        folium.Choropleth(
            geo_data=geojson_data,
            data=aggregated_data,
            columns=['Country', selected_category],
            key_on='feature.properties.admin',
            fill_color="YlOrRd",
            fill_opacity=0.7,
            line_opacity=0.2,
            legend_name=selected_category  # Dynamically set legend based on selected metric
        ).add_to(m)

    except Exception as e:
        st.error(f'Error loading GeoJSON file for {selected_continent}: {e}')

    # Adding markers for each country based on the sample
    for index, row in df_sampled.iterrows():
        folium.Marker(
            [row['City Latitude'], row['City Longitude']],
            popup=f"{selected_category}: {row[selected_category]}",
            tooltip=row['Country']
        ).add_to(m)

    st_folium(m, width=725, height=500)
x