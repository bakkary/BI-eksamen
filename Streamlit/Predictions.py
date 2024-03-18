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
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, recall_score
import matplotlib.pyplot as plt

# Your DataLoader function
@st.cache_data
def load_data():
    """Load and return the DataFrame from the pickle file."""
    file_path = os.path.join(os.getcwd(), 'dataframe.pkl')
    if os.path.exists(file_path):  # Check if the file exists
        return pd.read_pickle(file_path)
    else:
        st.error('Data file not found. Please check the file path.')  # Show an error if the file is not found
        return pd.DataFrame()  # Return an empty DataFrame as a fallback

@st.experimental_memo
def load_selected_features(selected_features_path='selected_features.joblib'):
    return joblib.load(selected_features_path)

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
    conf_matrix = confusion_matrix(y_test, y_pred)
    

     # Calculate recall
    recall = recall_score(y_test, y_pred)
    
    st.write(f"Model trained successfully with accuracy: {accuracy:.4f}")  # Formatted for precision
    st.write(f"Recall: {recall:.4f}")  # Formatted for precision
    st.text("Classification Report:\n" + report)  # Use st.text for preformatted text



    # Plotting confusion matrix
    fig, ax = plt.subplots()
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    st.pyplot(fig)
    
    # Save the trained model
    joblib.dump(rf_pipeline, 'trained_model.joblib')

    return rf_pipeline  # Return the trained model

def Show_Predictions():
    st.title('AI Predictions - Focusing on C40 Membership')
    st.write('This app uses a Random Forest Classifier to predict whether a city is eligible to join the C40 network based on various features.')
    df = load_data()

    if 'selected_features' not in st.session_state:
        st.session_state['selected_features'] = None

    method = st.radio("Choose Feature Selection Method:", ('Use Preprocessed Features', 'Select Your Own Features'))

    if method == 'Use Preprocessed Features':
        # Directly specify the four desired features instead of loading and filtering
        specific_features = ["Total emissions (metric tonnes CO2e)", "Population", "GDP", "​Land area (in square km)"]
        st.session_state['selected_features'] = specific_features
        
        st.success('Specific features selected for analysis:')
        st.write(st.session_state['selected_features'])
        
    elif method == 'Select Your Own Features':
        all_features = list(df.columns)
        all_features.remove('C40_True')  # Assuming 'C40_True' is the target variable
        selected_features = st.multiselect('Select features for training', all_features, default=['City', 'AQI Value'])
        st.session_state['selected_features'] = selected_features

    if not st.session_state['selected_features']:
        st.warning('Please select at least one feature to proceed.')
        return

    st.session_state.test_size = st.slider('Test set size (%)', min_value=10, max_value=50, value=20, step=5)

    trained_model = None  # Define trained_model variable

    if st.button('Train Model'):
        trained_model = train_model(df, st.session_state['selected_features'])
        # Update session state with trained model
        st.session_state['trained_model'] = trained_model    # Prediction
    st.title('Predict City Eligibility for C40 Membership')

    # Input fields based on selected features
    input_data = {}
    for feature in st.session_state['selected_features']:
        if pd.api.types.is_numeric_dtype(df[feature]):
            input_data[feature] = st.number_input(f'Enter {feature}:')
        elif pd.api.types.is_object_dtype(df[feature]):
            input_data[feature] = st.text_input(f'Enter {feature}:')

    # Make prediction
    if st.button('Predict Eligibility'):
        if 'trained_model' not in st.session_state:
            st.error("Please train the model first.")
            return
        
        trained_model = st.session_state['trained_model']
        input_df = pd.DataFrame([input_data])
        
        try:
            prediction_proba = trained_model.predict_proba(input_df)
            prediction = trained_model.predict(input_df)
            if prediction[0]:
                st.success('The city is eligible to join C40!')
            else:
                st.error('The city is not eligible to join C40.')

            st.write(f'Probability of being eligible: {prediction_proba[0][1]}')
            
            # Display feature importances
            feature_importances = trained_model.named_steps['classifier'].feature_importances_
            importance_df = pd.DataFrame({'Feature': st.session_state['selected_features'], 'Importance': feature_importances})
            importance_df = importance_df.sort_values(by='Importance', ascending=False)
            st.write('Feature Importances:')
            st.write(importance_df)
            
        
        except ValueError:
            st.warning("Some input values are missing, which may affect the accuracy of the prediction.")

if __name__ == '__main__':
    Show_Predictions()
