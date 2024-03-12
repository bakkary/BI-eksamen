import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np
from DataLoader import load_data

# Using @st.cache to load your data efficiently by caching the result
# This prevents reloading data from scratch on every user interaction
@st.cache_data
def load_data_cached():
    return load_data()

def show_Predictions():
    st.title('AI Predictions - Focusing on Asian Cities C40 Membership')
    train_model_interactive()

def train_model_interactive():
    df = load_data_cached()
    st.write("Data Loaded Successfully")
    # Provide an option to view the raw dataset
    if st.checkbox('Show raw data'):
        st.write(df.sample(50))

    # Interactive feature selection for model training
    all_features = list(df.columns)
    selected_features = st.multiselect('Select features for training', all_features, default=['AQI Value'])

    if not selected_features:
        st.warning('Please select at least one feature to proceed.')
        return

    # Splitting the dataset into training and testing sets based on user input
    X = df[selected_features]
    y = df[['C40_True', 'C40_False']]  # Combine both 'C40_True' and 'C40_False' for binary classification
    test_size = st.slider('Test set size (%)', min_value=10, max_value=100, value=20, step=5) / 100
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # Preprocessing: handling numeric and categorical features separately
    numeric_features = [f for f in selected_features if df[f].dtype != 'object']
    categorical_features = [f for f in selected_features if df[f].dtype == 'object']
    numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='missing')), ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features), ('cat', categorical_transformer, categorical_features)])

    # Training the RandomForestClassifier with the preprocessed data
    rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))])
    if st.button('Train Model'):
        rf_pipeline.fit(X_train, y_train)
        y_pred = rf_pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        # Displaying model performance metrics
        st.write(f"Model trained successfully with accuracy: {accuracy}")
        st.text("Classification Report:")
        st.text(report)

        # Get the probability of C40 eligibility for a given city
        not_c40_cities = df[df['C40_True'] == 0]['City'].unique()
        selected_city = st.selectbox('Select a city:', not_c40_cities)
        if selected_city:
         city_data = df[df['City'] == selected_city][selected_features]
        if not city_data.empty:
         city_data_processed = preprocessor.transform(city_data)  # Apply preprocessing
        c40_probability = rf_pipeline.predict_proba(city_data_processed)[0][1]  # Probability of C40_True
        st.write(f"The probability of {selected_city} being C40 eligible is: {c40_probability}")


        # Evaluating the hypothesis based on the model's accuracy
        if accuracy > 0.5:  # You might adjust this threshold based on your needs
            st.write("H1: The model can predict C40 membership for cities in Asia.")
        else:
            st.write("H0: The model cannot predict C40 membership for cities in Asia.")

show_Predictions()
