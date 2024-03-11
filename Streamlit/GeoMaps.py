import json
import streamlit as st
from streamlit_folium import st_folium
import folium
import pandas as pd
import numpy as np
import branca.colormap as cm
from DataLoader import load_data

def sample_cities(df, max_samples=150, sort_by=None, random_state=42):
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

def generate_map(df_sampled, granularity, geojson_data, selected_category, country_totals):
    continent_centers = {
        'Europe': [54, 15], 'North America': [40, -100], 'South America': [-15, -60],
        'Africa': [10, 20], 'Asia': [34, 100]
    }
    selected_continent = df_sampled['Continent'].iloc[0]
    center = continent_centers.get(selected_continent, [0, 0])
    m = folium.Map(location=center, zoom_start=4)

    # Calculate min and max values for the colormap
    min_val, max_val = country_totals[selected_category].min(), country_totals[selected_category].max()
    colormap = cm.linear.YlOrRd_09.scale(min_val, max_val).to_step(n=20)
    colormap.caption = selected_category
    colormap.add_to(m)

    # Apply color gradient based on the country's total metric
    folium.GeoJson(
        geojson_data,
        style_function=lambda feature: {
            'fillColor': colormap(country_totals[country_totals['Country'] == feature['properties']['name']][selected_category].sum()),
            'color': 'black',
            'weight': 2,
            'fillOpacity': 0.5
        },
        tooltip=folium.GeoJsonTooltip(fields=['name'], aliases=['Country']),
    ).add_to(m)

    # Smaller markers for each city
    if granularity == 'City':
        for _, row in df_sampled.iterrows():
            folium.CircleMarker(
                location=[row['City Latitude'], row['City Longitude']],
                radius=5,
                color='blue',
                fill=True,
                fill_color='blue',
                popup=f"{row['City']}: {row[selected_category]}"
            ).add_to(m)

    # Larger marker for the total of each country
    for _, row in country_totals.iterrows():
        folium.CircleMarker(
            location=[row['Country Latitude'], row['Country Longitude']],
            radius=10,
            color='red',
            fill=True,
            fill_color='red',
            popup=f"{row['Country']} Total: {row[selected_category]}",
        ).add_to(m)

    st_folium(m, width=725, height=500)


def show_Geomaps():
    st.title('Geomaps')
    st.write("This page allows you to visualize data on a map based on the selected category, continent, and level of granularity (Country or City).")

    df = load_data()
    numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    continents = ['Europe', 'North America', 'South America', 'Africa', 'Asia']
    selected_continent = st.selectbox('Select a continent:', continents)
    df_continent = df[df['Continent'] == selected_continent]

    granularity = st.selectbox('Visualize by:', ['Country', 'City'])
    selected_category = st.selectbox('Select a metric to visualize:', numerical_columns)

    if granularity == 'City':
        sort_by_metric = st.selectbox('Select a metric to sort by:', ['None'] + numerical_columns)
        df_sampled = sample_cities(df_continent, sort_by=sort_by_metric if sort_by_metric != 'None' else None)
    else:
        df_sampled = df_continent.groupby('Country').apply(lambda x: x.sample(n=1, random_state=42)).reset_index(drop=True)

    # Aggregate the total metric for each country
    country_totals = df_continent.groupby('Country', as_index=False).agg({
        selected_category: 'sum',
        'Country Latitude': 'mean',  # Adjust this based on your data
        'Country Longitude': 'mean'  # Adjust this based on your data
    })

    geojson_paths = {
        'Europe': '../GeoMaps/EU Map.json', 
        'North America': '../GeoMaps/NA Map.json',
        'South America': '../GeoMaps/SA Map.json', 
        'Africa': '../GeoMaps/Africa Map.json',
        'Asia': '../GeoMaps/Asia Map.json'
    }
    geojson_data = load_geojson(geojson_paths[selected_continent])



    generate_map(df_sampled, granularity, geojson_data, selected_category, country_totals)

show_Geomaps()
