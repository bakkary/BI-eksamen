import json
import streamlit as st
from streamlit_folium import st_folium
import folium
import pandas as pd
from DataLoader import load_data

def sample_cities(df, max_samples=50, sort_by=None, random_state=42):
    if sort_by and sort_by in df.columns:
        df = df.sort_values(by=sort_by, ascending=False)
    min_one_per_country = df.groupby('Country').apply(lambda x: x.iloc[0]).reset_index(drop=True)
    additional_samples = max_samples - len(min_one_per_country)
    if additional_samples > 0:
        excluded_indices = min_one_per_country.index
        additional_cities = df[~df.index.isin(excluded_indices)].sample(n=min(additional_samples, len(df) - len(excluded_indices)), random_state=random_state)
        result = pd.concat([min_one_per_country, additional_cities], ignore_index=True)
    else:
        result = min_one_per_country
    return result

def load_geojson(geojson_path):
    with open(geojson_path, 'r', encoding='utf-8') as f:
        geojson_data = json.load(f)
    return geojson_data

def generate_map(df_sampled, granularity, geojson_data, selected_category):
    continent_centers = {
        'Europe': [54, 15], 'North America': [40, -100], 'South America': [-15, -60],
        'Africa': [10, 20], 'Asia': [34, 100]
    }
    selected_continent = df_sampled['Continent'].iloc[0]  # Get the selected continent from the sampled data
    center = continent_centers.get(selected_continent, [0, 0])
    m = folium.Map(location=center, zoom_start=4)

    folium.GeoJson(geojson_data).add_to(m)  # Add GeoJSON data to the map

    for index, row in df_sampled.iterrows():
        location = [row['City Latitude'], row['City Longitude']] if granularity == 'City' else [row.get('Country Latitude', 0), row.get('Country Longitude', 0)]
        popup_text = f"{row['City'] if granularity == 'City' else row['Country']}: {row[selected_category]}"
        folium.Marker(location, popup=popup_text).add_to(m)

    st_folium(m, width=725, height=500)

def show_Geomaps():
    st.title('Geomaps')
    st.write("This page allows you to visualize data on a map based on the selected category, continent, and level of granularity (Country or City).")

    df = load_data()

    continents = ['Europe', 'North America', 'South America', 'Africa', 'Asia']
    selected_continent = st.selectbox('Select a continent:', continents)
    df_continent = df[df['Continent'] == selected_continent]
    granularity = st.selectbox('Visualize by:', ['Country', 'City'])
    
    if granularity == 'City':
        category_names = df.columns.tolist()
        sort_by_metric = st.selectbox('Select a metric to sort by:', ['None'] + category_names)
        df_sampled = sample_cities(df_continent, sort_by=sort_by_metric if sort_by_metric != 'None' else None)
    else:  # Country granularity
        df_sampled = df_continent.groupby('Country').apply(lambda x: x.sample(n=1, random_state=42)).reset_index(drop=True)
    
    selected_category = st.selectbox('Select a metric to visualize:', df.columns.tolist())
    total_selected_metric = df_sampled[selected_category].sum()
    st.write(f"Total {selected_category}: {total_selected_metric}")

    # Load GeoJSON data based on selected continent
    geojson_paths = {
        'Europe': '../GeoMaps/EU Map.json', 
        'North America': '../GeoMaps/NA Map.json',
        'South America': '../GeoMaps/SA Map.json', 
        'Africa': '../GeoMaps/Africa Map.json',
        'Asia': '../GeoMaps/Asia Map.json'
    }
    geojson_data = load_geojson(geojson_paths[selected_continent])

    generate_map(df_sampled, granularity, geojson_data, selected_category)

show_Geomaps()
