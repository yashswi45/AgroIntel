# crop_recommendation_system.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, mean_squared_error, r2_score
import joblib
import warnings

warnings.filterwarnings('ignore')


# ======================
# DATA PROCESSING
# ======================
def load_and_preprocess_data(filepath):
    """Load and preprocess the agricultural dataset"""
    df = pd.read_csv(filepath)

    # Clean column names
    df.columns = [col.strip().replace(' ', '_') for col in df.columns]

    # Handle missing values
    df = df.dropna()

    # Convert scientific notation to float
    if 'Production' in df.columns:
        df['Production'] = df['Production'].apply(lambda x: float(x) if 'E' in str(x) else x)

    # Feature engineering
    df['Yield'] = df['Production'] / df['Area']  # Yield in tons/hectare

    # Encode categorical variables
    categorical_cols = ['Crop', 'Season', 'State', 'Soil_Type']
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    # Select relevant features
    features = ['Crop_Year', 'Season', 'State', 'Annual_Rainfall',
                'Fertilizer', 'Pesticide', 'Soil_Type']

    # Scale numerical features
    scaler = StandardScaler()
    numerical_cols = ['Crop_Year', 'Annual_Rainfall', 'Fertilizer', 'Pesticide']
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    return df, features, encoders, scaler


# ======================
# CROP RECOMMENDATION MODEL
# ======================
def train_crop_recommendation_model(df, features):
    """Train and save crop recommendation classifier"""
    X = df[features]
    y = df['Crop']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=10,
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    print("Crop Recommendation Model Performance:")
    print(classification_report(y_test, y_pred))

    return model


# ======================
# YIELD PREDICTION MODEL
# ======================
def train_yield_prediction_model(df, features):
    """Train and save yield prediction regressor"""
    X = df[features + ['Crop']]  # Include crop as feature
    y = df['Yield']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    print("\nYield Prediction Model Performance:")
    print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")
    print(f"R2 Score: {r2_score(y_test, y_pred):.4f}")

    return model


# ======================
# PREDICTION FUNCTIONS
# ======================
def recommend_crops(user_input, crop_model, encoders, scaler):
    """Recommend top 3 crops based on conditions"""
    # Prepare input DataFrame
    input_df = pd.DataFrame([user_input])

    # Encode categorical features
    for col in ['Season', 'State', 'Soil_Type']:
        input_df[col] = encoders[col].transform([user_input[col]])[0]

    # Scale numerical features
    numerical_cols = ['Crop_Year', 'Annual_Rainfall', 'Fertilizer', 'Pesticide']
    input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

    # Get features in correct order
    features = ['Crop_Year', 'Season', 'State', 'Annual_Rainfall',
                'Fertilizer', 'Pesticide', 'Soil_Type']
    X = input_df[features]

    # Get probabilities for all crops
    probas = crop_model.predict_proba(X)[0]
    crop_probs = list(zip(encoders['Crop'].classes_, probas))

    # Return top 3 crops with probabilities
    return sorted(crop_probs, key=lambda x: x[1], reverse=True)[:3]


def predict_yield(crop, conditions, yield_model, encoders, scaler):
    """Predict yield for a specific crop under given conditions"""
    # Prepare input DataFrame
    input_df = pd.DataFrame([conditions])

    # Encode categorical features
    for col in ['Season', 'State', 'Soil_Type']:
        input_df[col] = encoders[col].transform([conditions[col]])[0]

    # Encode crop
    input_df['Crop'] = encoders['Crop'].transform([crop])[0]

    # Scale numerical features
    numerical_cols = ['Crop_Year', 'Annual_Rainfall', 'Fertilizer', 'Pesticide']
    input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

    # Get features in correct order
    features = ['Crop_Year', 'Season', 'State', 'Annual_Rainfall',
                'Fertilizer', 'Pesticide', 'Soil_Type', 'Crop']
    X = input_df[features]

    return yield_model.predict(X)[0]


# ======================
# MAIN EXECUTION
# ======================
if __name__ == "__main__":
    # Load and preprocess data
    print("Loading and preprocessing data...")
    df, features, encoders, scaler = load_and_preprocess_data("crop_data.csv")

    # Train models
    print("\nTraining crop recommendation model...")
    crop_model = train_crop_recommendation_model(df, features)

    print("\nTraining yield prediction model...")
    yield_model = train_yield_prediction_model(df, features)

    # Save models and artifacts
    joblib.dump(crop_model, 'crop_recommendation_model.pkl')
    joblib.dump(yield_model, 'yield_prediction_model.pkl')
    joblib.dump(encoders, 'label_encoders.pkl')
    joblib.dump(scaler, 'feature_scaler.pkl')

    # Test predictions
    test_conditions = {
        'Crop_Year': 2023,
        'Season': 'Kharif',
        'State': 'Assam',
        'Annual_Rainfall': 2051.4,
        'Fertilizer': 7000,
        'Pesticide': 2000,
        'Soil_Type': 'Clay'
    }

    print("\nTesting crop recommendation...")
    recommended = recommend_crops(test_conditions, crop_model, encoders, scaler)
    print("Top 3 Recommended Crops:")
    for crop, prob in recommended:
        print(f"- {crop}: {prob:.1%} confidence")

        # Predict yield for each recommended crop
        test_conditions['Crop'] = crop
        yield_pred = predict_yield(crop, test_conditions, yield_model, encoders, scaler)
        print(f"  Predicted Yield: {yield_pred:.2f} tons/hectare")