import streamlit as st
import os
import pandas as pd
import numpy as np
from DataLoader import load_data
from sklearn.cluster import KMeans
import sklearn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

def show_Regression():
    st.title('Regression and clustering')
    st.write("On this page, we will take a look at the dataset and try to visualize it. We will use the data to create some graphs and see if we can find any patterns or trends.")

    # Load your DataFrame
    df = load_data() 
    
    df.info()

    linreg = LinearRegression()

    # Analysis of air quality in C40 cities using K-means clustering
    st.title('Analysis of air quality in C40 cities using K-means clustering')
    feature_cols = ['City Latitude', 'City Longitude']
    X = df[feature_cols]

    print(feature_cols)
    print(X)


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

    # Scatterplot
    st.title('Scatterplot')
    current_dir = os.path.dirname(__file__)  # Gets the directory where the script is located
    image_path = 'screenshots\pic 1.png'
    st.image(image_path, caption="Caption for the image", use_column_width=True)
    st.write("Our error value is: 1.6936110240878405e-13")
    st.write("We got a test score of 1,0")

    # Linear Regression
    st.title('Linear Regression ')
    current_dir = os.path.dirname(__file__)  
    image_path = 'screenshots\pic 3.png'
    st.image(image_path, caption="Caption for the image", use_column_width=True)
    st.write("We got a R2 value of 0.0014905377546974297.")

    # 2D clustering
    st.title('2D Clustering')
    current_dir = os.path.dirname(__file__)  
    image_path = 'screenshots\pic 4.png'
    st.image(image_path, caption="Caption for the image", use_column_width=True)
    st.write("We are testing with 3 clusters. ")

    # 3D clustering
    st.title('3D clustering')
    current_dir = os.path.dirname(__file__)  
    image_path = 'screenshots\pic 5.png'
    st.image(image_path, caption="Caption for the image", use_column_width=True)
    st.write("The diagram above makes a lot of sense of 3D.")

    # Cluster on the 5 most efficient parameters
    st.title('Cluster on the 5 most efficient parameters')
    current_dir = os.path.dirname(__file__)  
    image_path = 'screenshots\pic 6.png'
    st.image(image_path, caption="Caption for the image", use_column_width=True)
    st.write("We are testing with 3 clusters. ")
    st.write("We got a K-means score: 1.7939260566779875e+29, which is a very high score and shows that the clusters is far from eachother.")

def main():
    show_Regression()

if __name__ == "__main__":
    main()
