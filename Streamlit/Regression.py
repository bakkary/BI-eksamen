import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from DataLoader import load_data

def show_Regression():
    st.title('Linear Regression')
    st.write("On this page, we will take a look at the dataset and try to visualize it. We will use the data to create some graphs and see if we can find any patterns or trends.")

    df = load_data()  # Load your DataFrame

    aqi_value(df)

    multiple_linear_regression(df)

    

# AQI value
def aqi_value(df):     
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
def multiple_linear_regression(df):
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


# figure 2 X
    

# figure 3 X
    

# figure 4 X 
    

# figure 5
    

# figure 6
    

# figure 7


def main():
    show_Regression()

if __name__ == "__main__":
    main()


