import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import pandas as pd
from DataLoader import load_data

@st.cache_data
def load_data_cached():
    return load_data()

def show_Predictions():
    st.title('AI Predictions - Focusing on C40 Membership')
    train_model_interactive()

def train_model_interactive():
    df = load_data_cached()
    st.write("Data Loaded Successfully")

    if st.checkbox('Show raw data'):
        st.write(df.sample(50))  # Show a random sample of 50 rows for quick inspection

    all_features = list(df.drop(['C40_True', 'C40_False'], axis=1).columns)  # Exclude target variable from features
    selected_features = st.multiselect('Select features for training', all_features, default=['City', 'AQI Value'])
    if not selected_features:
        st.warning('Please select at least one feature to proceed.')
        return

    # Directly use 'C40_True' as the binary target variable
    X = df[selected_features]
    y = df['C40_True']

    test_size = st.slider('Test set size (%)', min_value=10, max_value=50, value=20, step=5) / 100
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    numeric_features = [f for f in selected_features if df[f].dtype in ['int64', 'float64']]
    categorical_features = [f for f in selected_features if df[f].dtype == 'object']

    numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='missing')), ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features), ('cat', categorical_transformer, categorical_features)])

    rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))])

    if st.button('Train Model'):
        with st.spinner('Training model...'):
            rf_pipeline.fit(X_train, y_train)
            y_pred = rf_pipeline.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred)

        st.write(f"Model trained successfully with accuracy: {accuracy}")
        st.text("Classification Report:")
        st.text(report)

        # If you wish to predict the membership of a new city, you would collect the city's data,
        # preprocess it as per your training data, and then use `rf_pipeline.predict_proba(new_city_data)`
        # to get the probabilities of C40 membership.

# This call starts the app
show_Predictions()
