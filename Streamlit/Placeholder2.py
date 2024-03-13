import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

# Load your data here
# df = ...

# Drop 'C40_True' from the features since it's the target variable
X = df.drop(['C40_True', 'C40_False', 'Country', 'City', 'Continent'], axis=1)
y = df['C40_True']  # Use 'C40_False' as the target variable

# Identifying numeric and categorical features
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(exclude=['int64', 'float64']).columns

# Preprocessing pipelines for both numeric and categorical data
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

# Apply preprocessing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)

# Apply RFECV for feature selection
selector = RFECV(estimator=LogisticRegression(), step=1, cv=5, scoring='f1')
selector.fit(X_train_preprocessed, y_train)

# Increase the minimum number of features to select
selector.min_features_to_select = 5

# Save the selected feature names
selected_features = [X.columns[i] for i in range(len(selector.support_)) if selector.support_[i]]
joblib.dump(selected_features, 'selected_features.joblib')

# Print the selected features
print("Selected Features:", selected_features)

# Train the RandomForestClassifier on the selected features
rf_final = RandomForestClassifier(random_state=42)
rf_final.fit(selector.transform(X_train_preprocessed), y_train)

# Evaluate the model
y_pred = rf_final.predict(selector.transform(X_test_preprocessed))
print(f"Model F1-score: {f1_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))

# Save the preprocessor, selector, selected features, and the final model for later use
joblib.dump(preprocessor, 'preprocessor.joblib')
joblib.dump(selector, 'selector.joblib')
joblib.dump(selected_features, 'selected_features.joblib')
joblib.dump(rf_final, 'rf_final.joblib')
