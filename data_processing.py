import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import warnings

warnings.filterwarnings('ignore')


def clean_column_names(df):
    """Clean and standardize column names"""
    df.columns = [col.strip().replace(' ', '_') for col in df.columns]
    return df


def handle_scientific_notation(df):
    """Convert scientific notation to float"""
    if 'Production' in df.columns:
        df['Production'] = df['Production'].apply(
            lambda x: float(x) if 'E' in str(x) else x
        )
    return df


def preprocess_data(df):
    """Main preprocessing pipeline"""
    df = clean_column_names(df)
    df = handle_scientific_notation(df)
    df = df.dropna()

    # Fix season names using .loc to avoid SettingWithCopyWarning
    df.loc[:, 'Season'] = df['Season'].str.replace('Whole Ye', 'Whole Year')

    # Calculate yield
    df.loc[:, 'Yield'] = df['Production'] / df['Area']

    return df


def create_encoders(df):
    """Create and save label encoders"""
    categorical_cols = ['Crop', 'Season', 'State', 'Soil_Type']
    encoders = {}

    # Define expected categories based on your data
    expected_categories = {
        'Season': ['Kharif', 'Rabi', 'Summer', 'Winter', 'Whole Year', 'Autumn'],
        'State': ['Assam', 'Karnataka'],
        'Soil_Type': ['Clay', 'Black']
    }

    for col in categorical_cols:
        le = LabelEncoder()
        if col in expected_categories:
            le.fit(expected_categories[col] + list(df[col].unique()))
        else:
            le.fit(df[col])
        encoders[col] = le
        joblib.dump(le, f'{col}_encoder.pkl')

    return encoders


def scale_features(df):
    """Scale numerical features"""
    numerical_cols = [
        'Crop_Year',
        'Annual_Rainfall',
        'Fertilizer',
        'Pesticide'
    ]

    # Verify columns exist
    missing_cols = [col for col in numerical_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")

    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    joblib.dump(scaler, 'feature_scaler.pkl')
    return df, scaler


def process_data(input_path, output_path):
    """Complete data processing pipeline"""
    df = pd.read_csv(input_path)
    df = preprocess_data(df)

    # Create and save encoders
    encoders = create_encoders(df)

    # Encode categorical variables
    for col in encoders:
        df[col] = encoders[col].transform(df[col])

    # Scale numerical features
    df, scaler = scale_features(df)

    # Save processed data
    df.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}")

    return df, encoders, scaler
