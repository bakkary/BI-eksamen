import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from DataLoader import load_data  # Ensure this is correctly implemented
import joblib

@st.experimental_memo
def load_data_cached():
    return load_data()

@st.experimental_memo
def load_selected_features(selected_features_path='selected_features.joblib'):
    return joblib.load(selected_features_path)

def train_model(df, selected_features):
    X = df[selected_features]
    y = df['C40_True']
    
    test_size = st.session_state.test_size / 100
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    numeric_features = [f for f in selected_features if df[f].dtype in ['int64', 'float64']]
    categorical_features = [f for f in selected_features if df[f].dtype == 'object']

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

    rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('classifier', RandomForestClassifier(random_state=42))])

    rf_pipeline.fit(X_train, y_train)
    y_pred = rf_pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    st.write(f"Model trained successfully with accuracy: {accuracy}")
    st.text(report)

def show_Predictions():
    st.title('AI Predictions - Focusing on C40 Membership')
    df = load_data_cached()

    if 'selected_features' not in st.session_state:
        st.session_state.selected_features = None

    method = st.radio("Choose Feature Selection Method:", ('Use Preprocessed Features', 'Select Your Own Features'))

    if method == 'Use Preprocessed Features':
        st.session_state.selected_features = load_selected_features()
        st.success('Preprocessed features loaded successfully.')
    else:
        st.session_state.selected_features = st.multiselect('Select features for training', list(df.drop(['C40_True', 'C40_False'], axis=1).columns), default=['City', 'AQI Value'])

    if not st.session_state.selected_features:
        st.warning('Please select at least one feature to proceed.')
        return

    # Slider for test size is always shown regardless of selection method
    st.session_state.test_size = st.slider('Test set size (%)', min_value=10, max_value=50, value=20, step=5)

    if st.button('Train Model'):
        train_model(df, st.session_state.selected_features)

if __name__ == '__main__':
    show_Predictions()
