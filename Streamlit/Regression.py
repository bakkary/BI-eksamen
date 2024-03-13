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

    linear_regression_predictions(df)

    

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


# figure 2
def linear_regression_predictions(df):
    st.title('Linear Regression Predictions')
    st.write("This section demonstrates the predictions made by the linear regression model on new data points.")

    # Train the model on your training data
    X = df[['City Latitude', 'City Longitude']]
    y = df['AQI Value']
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.15)

    linreg = LinearRegression()  # Initialize the Linear Regression model
    linreg.fit(X_train, y_train)  # Train the model

    # Example values for City Latitude and AQI Value
    latitude_value = 40.7128  # Example latitude value for a city
    aqi_value_value = 50  # Example AQI Value

    # Predict using both features
    regression_predicted = linreg.predict([[latitude_value, aqi_value_value]])
    st.write("Predicted AQI Value:", regression_predicted[0])

    # Visualizing the regression predictions
    fig3, ax3 = plt.subplots()
    ax3.set_xlabel('City Latitude')
    ax3.set_ylabel('AQI Value')
    ax3.scatter(X_test['City Latitude'], y_test, color='blue', label='Actual AQI Value')
    ax3.scatter([latitude_value], regression_predicted, color='red', label='Predicted AQI Value')
    ax3.legend()
    ax3.set_title('Linear Regression Predictions')

    st.pyplot(fig3)


# figure 3 X
    

# figure 4 X 
    

# figure 5
    

# figure 6
    

# figure 7


def main():
    show_Regression()

if __name__ == "__main__":
    main()
