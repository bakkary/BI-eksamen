import os
import numpy as np
import pandas as pd
import streamlit as st
from DataLoader import load_data  # Adjust according to your DataLoader's actual location
import matplotlib.pyplot as plt
import seaborn as sns  # Import Seaborn for heatmap

def show_graphs():
    st.title('Data Visualisation')
    st.write("On this page, we will take a look at the dataset and try to visualize it. We will use the data to create some graphs and see if we can find any patterns or trends.")

    df = load_data()  # Load your DataFrame

    st.write('Here is a preview of the dataset:')
    with st.expander("Dataset Preview"):
        st.dataframe(df.sample(20))

    st.write('We can make a heatmap of the dataset to see if we can find any patterns or trends.')

    # Only consider numeric columns for the correlation matrix
    numeric_df = df.select_dtypes(include=['float64', 'int64'])

    df_cleaned = numeric_df.dropna()  # Drop NA values for clean correlation calculation
    corr_matrix = df_cleaned.corr()  # Calculate the correlation matrix

    # Plot the correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix')
    st.pyplot(plt)  # Display the plot in Streamlit

    st.write('It would make sense to take a look at some of the most correlating columns. We can do this by plotting a scatter plot of the columns with the highest correlation.')
    st.write('One thing to note is that the correlation does not imply that the data is saying enough, but it can give us a hint of where to look for patterns (especially for the AQI values that have a lot of correlation but doesn\'t make sense to compare them).')
    st.write('The total tonnes of CO2 and the population')

    # Plot the scatter plot
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x='Total emissions (metric tonnes CO2e)', y='Population')
    plt.title('Total CO2 Emissions vs. Population')
    plt.xlabel('Total emissions (metric tonnes CO2e)')
    plt.ylabel('Population')
    st.pyplot(plt)  # Display the plot in Streamlit

    st.write('We can also take a look at the distribution of the columns to see if we can find any patterns or trends.')
    st.write('We can start by plotting a histogram of the total emissions.')
    # Plot the histogram
    plt.figure(figsize=(8, 6))
    sns.histplot(df['Total emissions (metric tonnes CO2e)'], kde=True, bins=30)
    plt.title('Total CO2 Emissions Distribution')
    plt.xlabel('Total emissions (metric tonnes CO2e)')
    plt.ylabel('Frequency')
    st.pyplot(plt)  # Display the plot in Streamlit

    st.write('Let\'s take a look at the AQI values for the different countries')
    # Plot the bar chart
    plt.figure(figsize=(10, 8))
    sns.barplot(data=df, x='Country', y='AQI Value')
    plt.title('AQI values for different countries')
    plt.xlabel('Country')
    plt.ylabel('AQI Value')
    plt.xticks(rotation=90)
    st.pyplot(plt)  # Display the plot in Streamlit

    st.write('We can also take a look at the AQI values for different cities and check if they are a part of C40 or not')
    # Combine 'C40_True' and 'C40_False' into a single column
    df['C40_Status'] = df['C40_True'].map({1: 'True', 0: 'False'})

    # Get 50 random entries
    df_sampled = df.sample(n=50)

    # Plot the bar chart with swapped axes
    plt.figure(figsize=(10, 8))
    sns.barplot(data=df_sampled, x='AQI Value', y='City', hue='C40_Status', palette=['red', 'green'], dodge=False)
    plt.title('AQI values for different cities')
    plt.xlabel('AQI Value')
    plt.ylabel('City')
    plt.xticks(rotation=90)
    st.pyplot(plt)  # Display the plot in Jupyter Notebook

    

   
    # Apply log transformation to the column data and plot for 'AQI Value'
    data_for_plot = np.log1p(df['AQI Value'])
    plt.figure(figsize=(10, 6))
    sns.histplot(data_for_plot, kde=True)
    plt.title('Log-Transformed Distribution for AQI Value')
    st.pyplot(plt)  # Display the plot in Streamlit
    plt.clf()  # Clear the figure after displaying it

    # Repeat the process for 'NO2 AQI Value'
    data_for_plot = np.log1p(df['Total emissions (metric tonnes CO2e)'])
    plt.figure(figsize=(10, 6))
    sns.histplot(data_for_plot, kde=True)
    plt.title('Log-Transformed Distribution for Total emissions (metric tonnes CO2e)')
    st.pyplot(plt)
    plt.clf()

    # Repeat the process for 'PM2.5 AQI Value'
    data_for_plot = np.log1p(df['PM2.5 AQI Value'])
    plt.figure(figsize=(10, 6))
    sns.histplot(data_for_plot, kde=True)
    plt.title('Log-Transformed Distribution for PM2.5 AQI Value')
    st.pyplot(plt)
    plt.clf()

    # And finally for 'GDP'
    data_for_plot = np.log1p(df['GDP'])
    plt.figure(figsize=(10, 6))
    sns.histplot(data_for_plot, kde=True)
    plt.title('Log-Transformed Distribution for GDP')
    st.pyplot(plt)
    plt.clf()




    


