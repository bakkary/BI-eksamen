import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import joblib
from DataLoader import load_data

def load_selected_features(selected_features_path=None):
    if selected_features_path is not None:
        return joblib.load(selected_features_path)
    else:
        return None

def show_Predictions(selected_features=None):
    st.title('AI Predictions - Focusing on C40 Membership')
    st.write('For Training this model, we can either use the preprocessed features made with RFECV or select our own features.')
    st.write('After that we can select our test size and train the model.')

    # Add a button to use selected features from the preprocessed model
    use_preprocessed_features = st.button('Use Preprocessed Features')
    if use_preprocessed_features:
        selected_features = load_selected_features('C:/Users/chz/Documents/GitHub/BI-eksamen/Streamlit/selected_features.joblib')
        if selected_features is None:
            st.error('Error loading preprocessed features. Please make sure the file path is correct.')
            return
        else:
            st.success('Preprocessed features loaded successfully.')

    if selected_features is None:
        df = load_data()
        selected_features = st.multiselect('Select features for training', df.columns)
        if not selected_features:
            st.warning('Please select at least one feature to proceed.')
            return

    train_model_interactive(selected_features)

def train_model_interactive(selected_features):
    df = load_data()
    st.write("Data Loaded Successfully")

    if st.checkbox('Show raw data'):
        st.write(df.sample(50))

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

# This call starts the app
show_Predictions()
