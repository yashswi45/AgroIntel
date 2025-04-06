from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import pandas as pd


def train_model(df):
    """Train yield prediction model"""
    features = [
        'Crop',
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
    y = df['Yield']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    print("Yield Model Performance:")
    print(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")
    print(f"R2: {r2_score(y_test, y_pred):.2f}")

    joblib.dump(model, 'yield_model.pkl')
    return model


def predict_yield(crop, conditions, artifacts):
    """Predict yield for given crop and conditions"""
    try:
        # Define expected features (must match training features)
        expected_features = [
            'Crop',
            'Crop_Year',
            'Season',
            'State',
            'Annual_Rainfall',
            'Fertilizer',
            'Pesticide',
            'Soil_Type'
        ]

        # Prepare input with only expected features
        input_df = pd.DataFrame({k: [conditions[k]] if k != 'Crop' else [crop]
                                 for k in expected_features})

        # Encode categorical features
        for col in ['Season', 'State', 'Soil_Type', 'Crop']:
            if col == 'Crop':
                value = crop
            else:
                value = conditions[col]

            if value not in artifacts['encoders'][col].classes_:
                print(f"Warning: {col} value '{value}' not in training data")
                return None
            input_df[col] = artifacts['encoders'][col].transform([value])[0]

        # Scale numerical features
        numerical_cols = ['Crop_Year', 'Annual_Rainfall', 'Fertilizer', 'Pesticide']
        input_df[numerical_cols] = artifacts['scaler'].transform(input_df[numerical_cols])

        # Predict
        return artifacts['yield_model'].predict(input_df[expected_features])[0]

    except Exception as e:
        print(f"Error in yield prediction: {str(e)}")
        return None