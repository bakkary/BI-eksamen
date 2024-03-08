import os
import pandas as pd
import streamlit as st

@st.cache  # Standard caching decorator
def load_data():
    """Load and return the DataFrame from the pickle file."""
    file_path = os.path.join(os.getcwd(), 'dataframe.pkl')
    if os.path.exists(file_path):  # Check if the file exists
        return pd.read_pickle(file_path)
    else:
        st.error('Data file not found. Please check the file path.')  # Show an error if the file is not found
        return pd.DataFrame()  # Return an empty DataFrame as a fallback
