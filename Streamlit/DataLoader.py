import os
import pandas as pd
import streamlit as st

@st.cache_data  # Updated from st.cache to st.cache_data
def load_data():
    """Load and return the DataFrame from the pickle file."""
    file_path = os.path.join(os.getcwd(), 'dataframe.pkl')
    return pd.read_pickle(file_path)
