# import pandas for structuring the data
import pandas as pd

# import numpy for numerical analysis
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
import plotly.figure_factory as ff
import plotly.io as pio
import folium
from mpl_toolkits.mplot3d import Axes3D

# Machine learning and modeling
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import r2_score
import sklearn.metrics as sm
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

# Additional utilities
import os
import scipy.cluster.hierarchy as ch
from scipy.spatial.distance import cdist
from sklearn.metrics import mean_squared_error


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

def kmeans_clustering_figure(df):
    st.title('KMeans Clustering of Numeric Data')

    # Selecting numeric data for clustering
    numeric_df = df.select_dtypes(include=['number'])

    # Define the number of clusters
    num_clusters = 3

    # Initialize and fit the KMeans model
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(numeric_df)

    # Get cluster labels assigned to each row/data point
    cluster_labels = kmeans.labels_

    # Add cluster labels to the original DataFrame
    df['Cluster'] = cluster_labels

    # Assuming the first two numeric columns are what you want to plot
    feature_1 = numeric_df.columns[0]  # Adjust as needed
    feature_2 = numeric_df.columns[1]  # Adjust as needed

    # Plotting
    fig, ax = plt.subplots()
    scatter = ax.scatter(df[feature_1], df[feature_2], c=df['Cluster'], cmap='viridis', label=df['Cluster'])
    plt.colorbar(scatter, label='Cluster')
    ax.set_xlabel(feature_1)
    ax.set_ylabel(feature_2)
    ax.set_title('KMeans Clustering of Numeric Data')
    
    st.pyplot(fig)

# figure 6
    

# figure 7


def main():
    show_Regression()

if __name__ == "__main__":
    main()


