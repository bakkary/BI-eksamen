import streamlit as st
from DataLoader import load_data  # Make sure this import matches the filename and casing
import matplotlib.pyplot as plt

def show_Predictions():
    st.title('Generate Graphs')
    st.write("This page allows you to generate graphs from the dataset based on selected attributes.")

    df = load_data()  # Load your DataFrame

    # Fetching all column names for the dropdown options
    column_names = df.columns.tolist()

    # Creating dropdown menu for the user to select the x-axis and y-axis attributes
    x_axis = st.selectbox('Select attribute for the X-axis:', column_names, index=column_names.index('AQI Value'))
    y_axis = st.selectbox('Select attribute for the Y-axis:', column_names, index=column_names.index('CO AQI Value'))

    # Plotting based on the selected attributes
    fig, ax = plt.subplots()
    ax.plot(df[x_axis], df[y_axis], marker='o', linestyle='None')
    ax.set_title(f'Sample Graph: {x_axis} vs. {y_axis}')
    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)
    st.pyplot(fig)
    

    