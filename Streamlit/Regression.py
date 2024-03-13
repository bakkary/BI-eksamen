import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from DataLoader import load_data

def show_Regression():
    st.title('Data regression')
    st.write("On this page, we will take a look at the dataset and try to visualize it. We will use the data to create some graphs and see if we can find any patterns or trends.")

    df = load_data()  # Load your DataFrame

    st.title('Analysis of air quality in C40 cities using K-means clustering')

    feature_cols = ['City Latitude', 'City Longitude']
    X = df[feature_cols]

    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(X)

    kmeans_score = kmeans.inertia_

    cluster_labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    df['Cluster'] = cluster_labels

    st.write("K-means score:", kmeans_score)

    fig, ax = plt.subplots()
    ax.set_xlabel('City Latitude')
    ax.set_ylabel('City Longitude')
    scatter = ax.scatter(df['City Latitude'], df['City Longitude'], c=df['Cluster'], s=df['AQI Value'], cmap='Greens', label='Cities')
    centroids_plot = ax.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=100, c='red', label='Cluster Centroids')
    plt.colorbar(scatter, label='AQI Value')
    ax.legend()

    st.pyplot(fig)

def main():
    show_Regression()

if __name__ == "__main__":
    main()

    
