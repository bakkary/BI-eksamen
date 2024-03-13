import streamlit as st
import pandas as pd
import numpy as np
from DataLoader import load_data
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

def show_Regression():
    st.title('Linear Regression')
    st.write("On this page, we will take a look at the dataset and try to visualize it. We will use the data to create some graphs and see if we can find any patterns or trends.")

    # Load your DataFrame
    df = load_data() 

    linreg = LinearRegression()

    # Analysis of air quality in C40 cities using K-means clustering
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


    # figure 1
    st.title('Multiple Linear Regression')
    feature_cols = ['City Latitude', 'City Longitude']
    X = df[feature_cols]
    y = df['AQI Value']
    
    # Splitting the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.15)

    linreg = LinearRegression()
    linreg.fit(X_train, y_train)

    y_predicted = linreg.predict(X_test)

    # Plotting the regression results
    fig2, ax2 = plt.subplots()
    ax2.scatter(y_test, y_predicted, color='blue')
    ax2.set_xlabel('City')
    ax2.set_ylabel('AQI Value')
    ax2.set_title('Multiple Linear Regression Results')

    st.pyplot(fig2)

    # Displaying metrics
    mse = mean_squared_error(y_test, y_predicted)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_predicted)

    st.write(f"Mean Squared Error (MSE): {mse}")
    st.write(f"Root Mean Squared Error (RMSE): {rmse}")
    st.write(f"R-squared (R2 ): {r2}")


    # clustering
    # Let's say we want to cluster the data into 3 groups
    num_clusters = 3
    # Initialize the KMeans model
    kmeans = KMeans(n_clusters=num_clusters)
    numeric_df = df.select_dtypes(include=['number'])
    kmeans.fit(numeric_df)
    # Get the cluster labels assigned to each data point
    cluster_labels = kmeans.labels_
    # You can analyze the clusters, e.g., by adding cluster labels to the DataFrame
    df['Cluster'] = cluster_labels
    numeric_columns = df.select_dtypes(include=['number']).columns
    numeric_data = df[numeric_columns]
    numeric_data

    #Plot the clusters based on the first two columns (assuming they represent features)
    plt.scatter(df.iloc[:, 0], df.iloc[:, 1], c=cluster_labels, cmap='viridis')
    plt.title('KMeans Clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar(label='Cluster')
    plt.show()


def main():
    show_Regression()

if __name__ == "__main__":
    main()
