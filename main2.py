from data_processing import process_data
from crop_prediction import train_model as train_crop_model, load_artifacts, recommend_crops
from yield_prediction import train_model as train_yield_model, predict_yield
import joblib


def main():
    # Data processing
    print("Processing data...")
    input_path = 'C:/Users/user/PycharmProjectshelloworld/SOIL MODEL 1/crop_yield_with_soil.csv'
    output_path = 'C:/Users/user/PycharmProjectshelloworld/SOIL MODEL 1/preprocessed_data_with_soil_type.csv'
    df, encoders, scaler = process_data(input_path, output_path)

    # Model training
    print("\nTraining crop recommendation model...")
    crop_model = train_crop_model(df)

    print("\nTraining yield prediction model...")
    yield_model = train_yield_model(df)

    # Load all artifacts
    artifacts = load_artifacts()
    artifacts['yield_model'] = joblib.load('yield_model.pkl')

    # Test prediction
    print("\nTesting system with sample data...")
    test_conditions = {
        'Crop_Year': 1997,
        'Season': 'Kharif',
        'State': 'Assam',
        'Annual_Rainfall': 2051.4,
        'Fertilizer': 7000,
        'Pesticide': 2000,
        'Soil_Type': 'Clay'
    }

    # Get recommendations
    recommendations = recommend_crops(test_conditions, artifacts)
    print("\nTop 3 Recommended Crops:")
    if recommendations:
        for crop, prob in recommendations:
            yield_pred = predict_yield(crop, test_conditions, artifacts)
            if yield_pred is not None:
                print(f"- {crop} (Confidence: {prob:.1%}, Predicted Yield: {yield_pred:.2f} tons/ha)")
            else:
                print(f"- {crop} (Confidence: {prob:.1%}, Yield prediction unavailable)")
    else:
        print("No recommendations could be generated")


if __name__ == "__main__":
    main()
