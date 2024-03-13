import streamlit as st
import pandas as pd
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import joblib
import seaborn as sns
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

    st.write(f"Model trained successfully with accuracy: {accuracy}")
    st.text(report)

    # Plotting confusion matrix
    fig, ax = plt.subplots()
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    st.pyplot(fig)

def Show_Predictions():
    st.title('AI Predictions - Focusing on C40 Membership')
    df = load_data()

    if 'selected_features' not in st.session_state:
        st.session_state['selected_features'] = None

    method = st.radio("Choose Feature Selection Method:", ('Use Preprocessed Features', 'Select Your Own Features'))

    if method == 'Use Preprocessed Features':
        st.session_state['selected_features'] = load_selected_features()
        # Display the selected features loaded from the preprocessed file
        st.success('Preprocessed features loaded successfully. Selected Features:')
        st.write(st.session_state['selected_features'])
    elif method == 'Select Your Own Features':
        all_features = list(df.columns)
        all_features.remove('C40_True')  # Assuming 'C40_True' is the target variable
        st.session_state['selected_features'] = st.multiselect('Select features for training', all_features, default=['City', 'AQI Value'])

    if not st.session_state['selected_features']:
        st.warning('Please select at least one feature to proceed.')
        return

    st.session_state.test_size = st.slider('Test set size (%)', min_value=10, max_value=50, value=20, step=5)

    if st.button('Train Model'):
        train_model(df, st.session_state['selected_features'])

if __name__ == '__main__':
    Show_Predictions()
