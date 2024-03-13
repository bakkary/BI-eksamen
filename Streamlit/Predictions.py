import streamlit as st
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

def get_feature_names(column_transformer):
    """Get feature names from all transformers."""
    output_features = []
    for name, pipe, features in column_transformer.transformers_:
        if name == "remainder":
            continue
        transformer = pipe.named_steps.get('onehot', pipe)
        if hasattr(transformer, 'get_feature_names_out'):
            feature_names = transformer.get_feature_names_out(features)
            output_features.extend(feature_names)
        else:
            output_features.extend(features)
    return output_features

@st.cache
def load_data():
    """Load and return the DataFrame from the pickle file."""
    file_path = os.path.join(os.getcwd(), 'dataframe.pkl')
    if os.path.exists(file_path):
        return pd.read_pickle(file_path)
    else:
        st.error('Data file not found. Please check the file path.')
        return pd.DataFrame()

@st.cache(allow_output_mutation=True)
def load_feature_importances(feature_importances_path='feature_importances.joblib'):
    return joblib.load(feature_importances_path)

def train_model(df, selected_features):
    X = df[selected_features]
    y = df['C40_True']
    test_size = st.session_state.test_size / 100
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    numeric_features = [f for f in selected_features if pd.api.types.is_numeric_dtype(df[f])]
    categorical_features = [f for f in selected_features if pd.api.types.is_object_dtype(df[f])]

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),  # Fill missing values
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

    rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('classifier', RandomForestClassifier(random_state=42))])

    rf_pipeline.fit(X_train, y_train)
    
    # Adjust this part to correctly map transformed feature names to importances
    feature_names = get_feature_names(rf_pipeline.named_steps['preprocessor'])
    feature_importances = rf_pipeline.named_steps['classifier'].feature_importances_
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
    importance_df.sort_values(by='Importance', ascending=False, inplace=True)

    return rf_pipeline, importance_df

def Show_Predictions():
    st.title('AI Predictions - Focusing on C40 Membership')
    df = load_data()
    feature_importances_df = load_feature_importances()

    selected_features = ["Total emissions (metric tonnes CO2e)", "Population", "GDP", "Land area (in square km)"]

    st.success('Preprocessed features loaded successfully. Selected Features:')
    st.write(selected_features)

    if not selected_features:
        st.warning('Please select at least one feature to proceed.')
        return

    st.session_state.test_size = st.slider('Test set size (%)', min_value=10, max_value=50, value=20, step=5)

    if st.button('Train Model'):
        trained_model, feature_importances_df = train_model(df, selected_features)
        st.session_state['trained_model'] = trained_model
        # Display the feature importances DataFrame
        st.write('Feature Importances:', feature_importances_df)

        # Confusion Matrix
        st.subheader('Confusion Matrix')
        y_pred = trained_model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        st.write(cm)

    # Prediction part
    st.title('Predict City Eligibility for C40 Membership')
    if 'trained_model' in st.session_state:
        input_data = {}
        for feature in selected_features:
            input_data[feature] = st.number_input(f'Enter {feature}:', key=feature)

        if st.button('Predict Eligibility'):
            trained_model = st.session_state['trained_model']
            input_df = pd.DataFrame([input_data])

            try:
                prediction_proba = trained_model.predict_proba(input_df)
                prediction = trained_model.predict(input_df)
                if prediction[0] == 1:
                    st.success('The city is eligible to join C40!')
                else:
                    st.error('The city is not eligible to join C40.')

                st.write(f'Probability of being eligible: {prediction_proba[0][1]:.2f}')

            except ValueError as e:
                st.error(f"Error in prediction: {e}")

if __name__ == '__main__':
    Show_Predictions()
