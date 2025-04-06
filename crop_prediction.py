from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import pandas as pd


def train_model(df, target_col='Crop'):
    """Train crop recommendation model"""
    features = [
        'Crop_Year',
        'Season',
        'State',
        'Annual_Rainfall',
        'Fertilizer',
        'Pesticide',
        'Soil_Type'
    ]

    # Verify all features exist
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"Missing features: {missing}")

    X = df[features]
    y = df[target_col]


    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    print("Model Performance:")
    print(classification_report(y_test, y_pred))

    joblib.dump(model, 'crop_model.pkl')
    return model


def load_artifacts():
    """Load all required artifacts for prediction"""
    artifacts = {
        'model': joblib.load('crop_model.pkl'),
        'encoders': {
            'Season': joblib.load('Season_encoder.pkl'),
            'State': joblib.load('State_encoder.pkl'),
            'Soil_Type': joblib.load('Soil_Type_encoder.pkl'),
            'Crop': joblib.load('Crop_encoder.pkl')
        },
        'scaler': joblib.load('feature_scaler.pkl')
    }
    return artifacts


def recommend_crops(input_data, artifacts):
    """Make crop recommendations"""
    try:
        expected_features = [
            'Crop_Year',
            'Season',
            'State',
            'Annual_Rainfall',
            'Fertilizer',
            'Pesticide',
            'Soil_Type'
        ]

        # Prepare input with only expected features
        input_df = pd.DataFrame({k: [input_data[k]] for k in expected_features})

        # Encode categorical features
        for col in ['Season', 'State', 'Soil_Type']:
            if input_data[col] not in artifacts['encoders'][col].classes_:
                print(f"Warning: {col} value '{input_data[col]}' not in training data")
                return []
            input_df[col] = artifacts['encoders'][col].transform([input_data[col]])[0]

        # Scale numerical features
        numerical_cols = ['Crop_Year', 'Annual_Rainfall', 'Fertilizer', 'Pesticide']
        input_df[numerical_cols] = artifacts['scaler'].transform(input_df[numerical_cols])

        # Predict
        probas = artifacts['model'].predict_proba(input_df[expected_features])[0]
        crop_probs = list(zip(artifacts['encoders']['Crop'].classes_, probas))

        return sorted(crop_probs, key=lambda x: x[1], reverse=True)[:3]

    except Exception as e:
        print(f"Error in recommendation: {str(e)}")
        return []