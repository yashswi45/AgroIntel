# #
# #
# #
# # import streamlit as st
# # import tensorflow as tf
# # from tensorflow.keras.models import load_model
# # from tensorflow.keras.preprocessing.image import load_img, img_to_array
# # import numpy as np
# # import pandas as pd
# # import joblib
# # from PIL import Image
# # import os
# #
# # # Custom CSS for styling
# # st.markdown("""
# # <style>
# #     .sidebar .sidebar-content {
# #         background-image: linear-gradient(#3c9d40,#327e36);
# #         color: white;
# #     }
# #     .stButton>button {
# #         background-color: #4CAF50;
# #         color: white;
# #         border-radius: 5px;
# #         padding: 0.5rem 1rem;
# #     }
# #     .stButton>button:hover {
# #         background-color: #45a049;
# #     }
# #     .stExpander {
# #         border: 1px solid #e1e4e8;
# #         border-radius: 8px;
# #         padding: 1rem;
# #     }
# #     .stAlert {
# #         border-radius: 8px;
# #     }
# #     .title-text {
# #         color: #2e7d32;
# #         font-weight: 700;
# #     }
# #     .feature-card {
# #         border-radius: 10px;
# #         padding: 1.5rem;
# #         margin-bottom: 1rem;
# #         background-color: #3D8D7A;
# #         box-shadow: 0 4px 6px rgba(0,0,0,0.1);
# #     }
# # </style>
# # """, unsafe_allow_html=True)
# #
# #
# # # Load all models and encoders at startup
# # @st.cache_resource
# # @st.cache_resource
# # def load_models():
# #     try:
# #         st.write("Loading soil model...")
# #         soil_model = load_model('soil_type_model.h5')
# #
# #         st.write("Loading crop model...")
# #         crop_model = joblib.load('crop_model.pkl')
# #
# #         st.write("Loading yield model...")
# #         yield_model = joblib.load('yield_model.pkl')
# #
# #         st.write("Loading encoders...")
# #         encoders = {
# #             'Season': joblib.load('Season_encoder.pkl'),
# #             'State': joblib.load('State_encoder.pkl'),
# #             'Soil_Type': joblib.load('Soil_Type_encoder.pkl'),
# #             'Crop': joblib.load('Crop_encoder.pkl')
# #         }
# #
# #         st.write("Loading scaler...")
# #         scaler = joblib.load('feature_scaler.pkl')
# #
# #         return soil_model, crop_model, yield_model, encoders, scaler
# #     except Exception as e:
# #         st.error(f"Error loading models: {str(e)}")
# #         return None, None, None, None, None
# #
# #
# #
# # # Soil type classification function
# # def classify_soil(image, model):
# #     img = image.resize((150, 150))
# #     img_array = img_to_array(img)
# #     img_array = img_array / 255.0  # Normalize
# #     img_array = np.expand_dims(img_array, axis=0)
# #
# #     predictions = model.predict(img_array)
# #     class_idx = np.argmax(predictions[0])
# #
# #     soil_types = ['alluvial', 'black', 'cinder', 'clay', 'laterite', 'peat', 'red', 'yellow']
# #     return soil_types[class_idx], predictions[0][class_idx]
# #
# #
# # # Crop recommendation function
# # def recommend_crops(conditions, model, encoders, scaler):
# #     try:
# #         input_df = pd.DataFrame([conditions])
# #
# #         for col in ['Season', 'State', 'Soil_Type']:
# #             input_df[col] = encoders[col].transform([conditions[col]])[0]
# #
# #         numerical_cols = ['Crop_Year', 'Annual_Rainfall', 'Fertilizer', 'Pesticide']
# #         input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])
# #
# #         features = ['Crop_Year', 'Season', 'State', 'Annual_Rainfall',
# #                     'Fertilizer', 'Pesticide', 'Soil_Type']
# #         probas = model.predict_proba(input_df[features])[0]
# #         crop_probs = list(zip(encoders['Crop'].classes_, probas))
# #
# #         return sorted(crop_probs, key=lambda x: x[1], reverse=True)[:3]
# #     except Exception as e:
# #         st.error(f"Recommendation error: {str(e)}")
# #         return []
# #
# #
# # # Yield prediction function
# # def predict_yield(crop, conditions, model, encoders, scaler):
# #     try:
# #         input_df = pd.DataFrame([conditions])
# #         input_df['Crop'] = crop
# #
# #         for col in ['Season', 'State', 'Soil_Type', 'Crop']:
# #             value = crop if col == 'Crop' else conditions[col]
# #             input_df[col] = encoders[col].transform([value])[0]
# #
# #         numerical_cols = ['Crop_Year', 'Annual_Rainfall', 'Fertilizer', 'Pesticide']
# #         input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])
# #
# #         features = ['Crop', 'Crop_Year', 'Season', 'State',
# #                     'Annual_Rainfall', 'Fertilizer', 'Pesticide', 'Soil_Type']
# #         return model.predict(input_df[features])[0]
# #     except Exception as e:
# #         st.error(f"Yield prediction error: {str(e)}")
# #         return None
# #
# #
# # # Home Page
# # def home_page():
# #     st.markdown("<h1 class='title-text'>🌾 Crop Advisory Model</h1>", unsafe_allow_html=True)
# #     st.markdown("""
# #     <div class='feature-card'>
# #         <h3>Your Intelligent Agricultural Assistant</h3>
# #         <p>Our AI-powered platform helps farmers make data-driven decisions for better crop management
# #         and improved yields. Combining soil analysis with environmental factors, we provide personalized
# #         recommendations for your farming needs.</p>
# #     </div>
# #     """, unsafe_allow_html=True)
# #
# #     st.image(
# #         "https://images.unsplash.com/photo-1500382017468-9049fed747ef?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=2070&q=80",
# #         use_column_width=True, caption="Modern Agriculture with AI Assistance")
# #
# #     st.markdown("""
# #     ## Key Features
# #
# #     <div class='feature-card'>
# #         <h4>🌱 Soil Type Classification</h4>
# #         <p>Upload images of your soil to automatically determine soil type with our advanced CNN model.</p>
# #     </div>
# #
# #     <div class='feature-card'>
# #         <h4>📊 Crop Recommendation Engine</h4>
# #         <p>Get personalized crop suggestions based on your soil, location, and farming conditions.</p>
# #     </div>
# #
# #     <div class='feature-card'>
# #         <h4>📈 Yield Prediction</h4>
# #         <p>Accurate yield forecasts to help with planning and resource allocation.</p>
# #     </div>
# #     """, unsafe_allow_html=True)
# #
# #     st.markdown("""
# #     ## What Else We Can Do
# #
# #     <div class='feature-card'>
# #         <h4>Future Enhancements</h4>
# #         <ul>
# #             <li>🌦️ Real-time weather integration for dynamic recommendations</li>
# #             <li>💰 Market price trends for crop selection optimization</li>
# #             <li>🧪 Fertilizer and pesticide optimization calculator</li>
# #             <li>📅 Seasonal planting calendar generator</li>
# #             <li>🌍 Regional disease/pest outbreak alerts</li>
# #             <li>🤖 Chatbot for agricultural Q&A</li>
# #         </ul>
# #     </div>
# #     """, unsafe_allow_html=True)
# #
# #
# # # Soil Classifier Page
# # def soil_classifier_page(soil_model):
# #     st.markdown("<h1 class='title-text'>🌍 Soil Type Classifier</h1>", unsafe_allow_html=True)
# #     st.markdown("""
# #     <div class='feature-card'>
# #         <p>Upload a clear photo of your soil to automatically determine its type.
# #         Our AI model can identify 8 different soil classifications with high accuracy.</p>
# #     </div>
# #     """, unsafe_allow_html=True)
# #
# #     col1, col2 = st.columns([2, 1])
# #     with col1:
# #         uploaded_file = st.file_uploader("Choose a soil image", type=["jpg", "jpeg", "png"])
# #
# #         if uploaded_file is not None:
# #             image = Image.open(uploaded_file)
# #             st.image(image, caption="Uploaded Soil Image", use_column_width=True)
# #
# #             if st.button("Analyze Soil", key="analyze_soil"):
# #                 with st.spinner("Processing soil image..."):
# #                     soil_type, confidence = classify_soil(image, soil_model)
# #                     st.success(f"""
# #                     **Analysis Results:**
# #                     - Predicted Soil Type: **{soil_type.capitalize()}**
# #                     - Confidence Level: **{confidence:.1%}**
# #                     """)
# #
# #                     # Soil type information
# #                     soil_info = {
# #                         'alluvial': "Rich in minerals, excellent for rice and wheat",
# #                         'black': "High clay content, good for cotton and sugarcane",
# #                         'clay': "Good water retention, ideal for rice and lentils",
# #                         'laterite': "Rich in iron, suitable for tea and coffee",
# #                         'red': "Good drainage, works well for millets and oilseeds",
# #                         'yellow': "Moderate fertility, good for maize and pulses"
# #                     }
# #
# #                     if soil_type in soil_info:
# #                         st.info(f"**{soil_type.capitalize()} Soil Characteristics:** {soil_info[soil_type]}")
# #
# #     with col2:
# #         st.markdown("""
# #         <br><br><br> <br>
# #         ### Tips for Best Results:
# #         - Take photo in natural daylight
# #         - Capture a close-up of undisturbed soil
# #         - Include a reference object for scale
# #         - Avoid shadows or glare
# #
# #         ### Supported Soil Types:
# #         - Alluvial
# #         - Black
# #         - Cinder
# #         - Clay
# #         - Laterite
# #         - Peat
# #         - Red
# #         - Yellow
# #         <br><br><br><br>
# #         """,unsafe_allow_html=True)
# #
# #
# # # Crop Advisor Page
# # def crop_advisor_page(crop_model, yield_model, encoders, scaler):
# #     st.markdown("<h1 class='title-text'>🌱 Crop Recommendation System</h1>", unsafe_allow_html=True)
# #     st.markdown("""
# #     <div class='feature-card'>
# #         <p>Get personalized crop recommendations based on your soil type, location,
# #         and agricultural practices. Our system analyzes multiple factors to suggest
# #         the most suitable crops for your farm.</p>
# #     </div>
# #     """, unsafe_allow_html=True)
# #
# #     with st.form("crop_recommendation_form"):
# #         st.subheader("Farm Conditions")
# #
# #         col1, col2 = st.columns(2)
# #         with col1:
# #             soil_type = st.selectbox("Soil Type", encoders['Soil_Type'].classes_)
# #             season = st.selectbox("Growing Season", encoders['Season'].classes_)
# #         with col2:
# #             state = st.selectbox("State/Region", encoders['State'].classes_)
# #             year = st.number_input("Year", min_value=2000, max_value=2030, value=2023)
# #
# #         st.subheader("Environmental Factors")
# #         col1, col2, col3 = st.columns(3)
# #         with col1:
# #             rainfall = st.number_input("Annual Rainfall (mm)", min_value=0, value=1500)
# #         with col2:
# #             fertilizer = st.number_input("Fertilizer (kg/ha)", min_value=0, value=5000)
# #         with col3:
# #             pesticide = st.slider("Pesticide Usage (kg/ha)", 0, 5000, 2000)
# #
# #         submitted = st.form_submit_button("Get Recommendations")
# #
# #         if submitted:
# #             conditions = {
# #                 'Crop_Year': year,
# #                 'Season': season,
# #                 'State': state,
# #                 'Annual_Rainfall': rainfall,
# #                 'Fertilizer': fertilizer,
# #                 'Pesticide': pesticide,
# #                 'Soil_Type': soil_type
# #             }
# #
# #             with st.spinner("Analyzing farm conditions..."):
# #                 recommendations = recommend_crops(conditions, crop_model, encoders, scaler)
# #
# #                 if recommendations:
# #                     st.success("## Recommended Crops")
# #
# #                     for crop, confidence in recommendations:
# #                         with st.expander(f"**{crop}** (Confidence: {confidence:.1%})", expanded=True):
# #                             yield_pred = predict_yield(crop, conditions, yield_model, encoders, scaler)
# #
# #                             col1, col2 = st.columns([1, 2])
# #                             with col1:
# #                                 if yield_pred:
# #                                     st.metric("Predicted Yield", f"{yield_pred:.2f} tons/ha")
# #                                 else:
# #                                     st.warning("Yield data unavailable")
# #
# #                             with col2:
# #                                 # Crop-specific advice
# #                                 advice = {
# #                                     "Rice": "Requires standing water, ideal for clay soils",
# #                                     "Wheat": "Best in well-drained loamy soils",
# #                                     "Maize": "Adaptable to various soils but needs good nitrogen",
# #                                     "Cotton": "Thrives in black soils with warm climate",
# #                                     "Sugarcane": "Needs fertile soil with good water retention"
# #                                 }
# #
# #                                 if crop in advice:
# #                                     st.info(f"**Growing Tips:** {advice[crop]}")
# #                                 else:
# #                                     st.info("General advice: Ensure proper irrigation and soil preparation")
# #                 else:
# #                     st.warning("No strong recommendations available for these conditions")
# #
# #
# # # Main App
# # def main():
# #     st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3079/3079155.png", width=100)
# #     st.sidebar.title("Crop Advisory Model")
# #
# #     # Navigation
# #     app_page = st.sidebar.radio("Navigation",
# #                                 ["Home", "Soil Classifier", "Crop Advisor"],
# #                                 index=0)
# #
# #     # Load models only when needed
# #     if app_page != "Home":
# #         soil_model, crop_model, yield_model, encoders, scaler = load_models()
# #         if None in [soil_model, crop_model, yield_model, encoders, scaler]:
# #             st.error("Failed to load required models. Please check your model files.")
# #             return
# #
# #     if app_page == "Home":
# #         home_page()
# #     elif app_page == "Soil Classifier":
# #         soil_classifier_page(soil_model)
# #     elif app_page == "Crop Advisor":
# #         crop_advisor_page(crop_model, yield_model, encoders, scaler)
# #
# #     # Footer
# #     st.sidebar.markdown("---")
# #     st.sidebar.markdown("""
# #     ### About This App
# #     Developed to help farmers make data-driven decisions using AI and machine learning.
# #
# #     **Key Technologies:**
# #     - TensorFlow/Keras for soil classification
# #     - Scikit-learn for crop recommendations
# #     - Streamlit for interactive interface
# #
# #     [GitHub Repository](#) | [Contact Us](#)
# #     """)
# #
# #
# # if __name__ == "__main__":
# #     main()
#
#
# import streamlit as st
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.image import load_img, img_to_array
# import numpy as np
# import pandas as pd
# import joblib
# from PIL import Image
# import os
#
# # Custom CSS for styling
# st.markdown("""
# <style>
#     .sidebar .sidebar-content {
#         background-image: linear-gradient(#3c9d40,#327e36);
#         color: white;
#     }
#     .stButton>button {
#         background-color: #4CAF50;
#         color: white;
#         border-radius: 5px;
#         padding: 0.5rem 1rem;
#     }
#     .stButton>button:hover {
#         background-color: #45a049;
#     }
#     .stExpander {
#         border: 1px solid #e1e4e8;
#         border-radius: 8px;
#         padding: 1rem;
#     }
#     .stAlert {
#         border-radius: 8px;
#     }
#     .title-text {
#         color: #2e7d32;
#         font-weight: 700;
#     }
#     .feature-card {
#         border-radius: 10px;
#         padding: 1.5rem;
#         margin-bottom: 1rem;
#         background-color: #3D8D7A;
#         box-shadow: 0 4px 6px rgba(0,0,0,0.1);
#     }
# </style>
# """, unsafe_allow_html=True)
#
#
# # Load all models and encoders at startup
# @st.cache_resource
# def load_models():
#     try:
#         soil_model = load_model('soil_type_model.h5')
#         crop_model = joblib.load('crop_model.pkl')
#         yield_model = joblib.load('yield_model.pkl')
#         encoders = {
#             'Season': joblib.load('Season_encoder.pkl'),
#             'State': joblib.load('State_encoder.pkl'),
#             'Soil_Type': joblib.load('Soil_Type_encoder.pkl'),
#             'Crop': joblib.load('Crop_encoder.pkl')
#         }
#         scaler = joblib.load('feature_scaler.pkl')
#         return soil_model, crop_model, yield_model, encoders, scaler
#     except Exception as e:
#         st.error(f"Error loading models: {str(e)}")
#         return None, None, None, None, None
#
#
# # Soil type classification function
# def classify_soil(image, model):
#     img = image.resize((150, 150))
#     img_array = img_to_array(img)
#     img_array = img_array / 255.0  # Normalize
#     img_array = np.expand_dims(img_array, axis=0)
#
#     predictions = model.predict(img_array)
#     class_idx = np.argmax(predictions[0])
#
#     soil_types = ['alluvial', 'black', 'cinder', 'clay', 'laterite', 'peat', 'red', 'yellow']
#     return soil_types[class_idx]
#
#
# # Crop recommendation function
# def recommend_crops(conditions, model, encoders, scaler):
#     try:
#         input_df = pd.DataFrame([conditions])
#
#         for col in ['Season', 'State', 'Soil_Type']:
#             input_df[col] = encoders[col].transform([conditions[col]])[0]
#
#         numerical_cols = ['Crop_Year', 'Annual_Rainfall', 'Fertilizer', 'Pesticide']
#         input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])
#
#         features = ['Crop_Year', 'Season', 'State', 'Annual_Rainfall',
#                     'Fertilizer', 'Pesticide', 'Soil_Type']
#         probas = model.predict_proba(input_df[features])[0]
#         crop_probs = list(zip(encoders['Crop'].classes_, probas))
#
#         return sorted(crop_probs, key=lambda x: x[1], reverse=True)[:3]
#     except Exception as e:
#         st.error(f"Recommendation error: {str(e)}")
#         return []
#
#
# # Yield prediction function
# def predict_yield(crop, conditions, model, encoders, scaler):
#     try:
#         input_df = pd.DataFrame([conditions])
#         input_df['Crop'] = crop
#
#         for col in ['Season', 'State', 'Soil_Type', 'Crop']:
#             value = crop if col == 'Crop' else conditions[col]
#             input_df[col] = encoders[col].transform([value])[0]
#
#         numerical_cols = ['Crop_Year', 'Annual_Rainfall', 'Fertilizer', 'Pesticide']
#         input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])
#
#         features = ['Crop', 'Crop_Year', 'Season', 'State',
#                     'Annual_Rainfall', 'Fertilizer', 'Pesticide', 'Soil_Type']
#         return model.predict(input_df[features])[0]
#     except Exception as e:
#         st.error(f"Yield prediction error: {str(e)}")
#         return None
#
#
# # Home Page
# def home_page():
#     st.markdown("<h1 class='title-text'>🌾 Crop Advisory Model</h1>", unsafe_allow_html=True)
#     st.markdown("""
#     <div class='feature-card'>
#         <h3>Your Intelligent Agricultural Assistant</h3>
#         <p>Our AI-powered platform helps farmers make data-driven decisions for better crop management
#         and improved yields. Combining soil analysis with environmental factors, we provide personalized
#         recommendations for your farming needs.</p>
#     </div>
#     """, unsafe_allow_html=True)
#
#     st.image(
#         "https://images.unsplash.com/photo-1500382017468-9049fed747ef?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=2070&q=80",
#         use_column_width=True, caption="Modern Agriculture with AI Assistance")
#
#     st.markdown("""
#     ## Key Features
#
#     <div class='feature-card'>
#         <h4>🌱 Soil Type Classification</h4>
#         <p>Upload images of your soil to automatically determine soil type with our advanced CNN model.</p>
#     </div>
#
#     <div class='feature-card'>
#         <h4>📊 Crop Recommendation Engine</h4>
#         <p>Get personalized crop suggestions based on your soil, location, and farming conditions.</p>
#     </div>
#
#     <div class='feature-card'>
#         <h4>📈 Yield Prediction</h4>
#         <p>Accurate yield forecasts to help with planning and resource allocation.</p>
#     </div>
#     """, unsafe_allow_html=True)
#
#     st.markdown("""
#     ## What Else We Can Do
#
#     <div class='feature-card'>
#         <h4>Future Enhancements</h4>
#         <ul>
#             <li>🌦️ Real-time weather integration for dynamic recommendations</li>
#             <li>💰 Market price trends for crop selection optimization</li>
#             <li>🧪 Fertilizer and pesticide optimization calculator</li>
#             <li>📅 Seasonal planting calendar generator</li>
#             <li>🌍 Regional disease/pest outbreak alerts</li>
#             <li>🤖 Chatbot for agricultural Q&A</li>
#         </ul>
#     </div>
#
#     ## About This Project
#
#     <div class='feature-card'>
#         <p>This agricultural advisory application was developed to help farmers make better decisions
#         using machine learning and AI technologies. The system combines soil analysis with environmental
#         factors to provide personalized recommendations.</p>
#         <p><strong>Technologies used:</strong> TensorFlow/Keras, Scikit-learn, Streamlit</p>
#     </div>
#     """, unsafe_allow_html=True)
#
#
# # Soil Classifier Page
# def soil_classifier_page(soil_model):
#     st.markdown("<h1 class='title-text'>🌍 Soil Type Classifier</h1>", unsafe_allow_html=True)
#     st.markdown("""
#     <div class='feature-card'>
#         <p>Upload a clear photo of your soil to automatically determine its type.
#         Our AI model can identify 8 different soil classifications with high accuracy.</p>
#     </div>
#     """, unsafe_allow_html=True)
#
#     uploaded_file = st.file_uploader("Choose a soil image", type=["jpg", "jpeg", "png"])
#
#     if uploaded_file is not None:
#         image = Image.open(uploaded_file)
#         st.image(image, caption="Uploaded Soil Image", use_column_width=True)
#
#         if st.button("Analyze Soil", key="analyze_soil"):
#             with st.spinner("Processing soil image..."):
#                 soil_type = classify_soil(image, soil_model)
#                 st.success(f"""
#                 **Analysis Results:**
#                 - Predicted Soil Type: **{soil_type.capitalize()}**
#                 """)
#
#                 # Soil type information
#                 soil_info = {
#                     'alluvial': "Rich in minerals, excellent for rice and wheat",
#                     'black': "High clay content, good for cotton and sugarcane",
#                     'clay': "Good water retention, ideal for rice and lentils",
#                     'laterite': "Rich in iron, suitable for tea and coffee",
#                     'red': "Good drainage, works well for millets and oilseeds",
#                     'yellow': "Moderate fertility, good for maize and pulses"
#                 }
#
#                 if soil_type in soil_info:
#                     st.info(f"**{soil_type.capitalize()} Soil Characteristics:** {soil_info[soil_type]}")
#
#     st.markdown("""
#     ### Tips for Best Results:
#     - Take photo in natural daylight
#     - Capture a close-up of undisturbed soil
#     - Include a reference object for scale
#     - Avoid shadows or glare
#
#     ### Supported Soil Types:
#     - Alluvial
#     - Black
#     - Cinder
#     - Clay
#     - Laterite
#     - Peat
#     - Red
#     - Yellow
#     """, unsafe_allow_html=True)
#
#
# # Crop Advisor Page
# def crop_advisor_page(crop_model, yield_model, encoders, scaler):
#     st.markdown("<h1 class='title-text'>🌱 Crop Recommendation System</h1>", unsafe_allow_html=True)
#     st.markdown("""
#     <div class='feature-card'>
#         <p>Get personalized crop recommendations based on your soil type, location,
#         and agricultural practices. Our system analyzes multiple factors to suggest
#         the most suitable crops for your farm.</p>
#     </div>
#     """, unsafe_allow_html=True)
#
#     with st.form("crop_recommendation_form"):
#         st.subheader("Farm Conditions")
#
#         col1, col2 = st.columns(2)
#         with col1:
#             soil_type = st.selectbox("Soil Type", encoders['Soil_Type'].classes_)
#             season = st.selectbox("Growing Season", encoders['Season'].classes_)
#         with col2:
#             state = st.selectbox("State/Region", encoders['State'].classes_)
#             year = st.number_input("Year", min_value=2000, max_value=2030, value=2023)
#
#         st.subheader("Environmental Factors")
#         col1, col2, col3 = st.columns(3)
#         with col1:
#             rainfall = st.number_input("Annual Rainfall (mm)", min_value=0, value=1500)
#         with col2:
#             fertilizer = st.number_input("Fertilizer (kg/ha)", min_value=0, value=5000)
#         with col3:
#             pesticide = st.slider("Pesticide Usage (kg/ha)", 0, 5000, 2000)
#
#         submitted = st.form_submit_button("Get Recommendations")
#
#         if submitted:
#             conditions = {
#                 'Crop_Year': year,
#                 'Season': season,
#                 'State': state,
#                 'Annual_Rainfall': rainfall,
#                 'Fertilizer': fertilizer,
#                 'Pesticide': pesticide,
#                 'Soil_Type': soil_type
#             }
#
#             with st.spinner("Analyzing farm conditions..."):
#                 recommendations = recommend_crops(conditions, crop_model, encoders, scaler)
#
#                 if recommendations:
#                     st.success("## Recommended Crops")
#
#                     for crop, _ in recommendations:
#                         with st.expander(f"**{crop}**", expanded=True):
#                             yield_pred = predict_yield(crop, conditions, yield_model, encoders, scaler)
#
#                             col1, col2 = st.columns([1, 2])
#                             with col1:
#                                 if yield_pred:
#                                     st.metric("Predicted Yield", f"{yield_pred:.2f} tons/ha")
#                                 else:
#                                     st.warning("Yield data unavailable")
#
#                             with col2:
#                                 # Crop-specific advice
#                                 advice = {
#                                     "Rice": "Requires standing water, ideal for clay soils",
#                                     "Wheat": "Best in well-drained loamy soils",
#                                     "Maize": "Adaptable to various soils but needs good nitrogen",
#                                     "Cotton": "Thrives in black soils with warm climate",
#                                     "Sugarcane": "Needs fertile soil with good water retention"
#                                 }
#
#                                 if crop in advice:
#                                     st.info(f"**Growing Tips:** {advice[crop]}")
#                                 else:
#                                     st.info("General advice: Ensure proper irrigation and soil preparation")
#                 else:
#                     st.warning("No strong recommendations available for these conditions")
#
#
# # Main App
# def main():
#     st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3079/3079155.png", width=100)
#     st.sidebar.title("Crop Advisory Model")
#
#     # Navigation
#     app_page = st.sidebar.radio("Navigation",
#                                 ["Home", "Soil Classifier", "Crop Advisor"],
#                                 index=0)
#
#     # Load models only when needed
#     if app_page != "Home":
#         soil_model, crop_model, yield_model, encoders, scaler = load_models()
#         if None in [soil_model, crop_model, yield_model, encoders, scaler]:
#             st.error("Failed to load required models. Please check your model files.")
#             return
#
#     if app_page == "Home":
#         home_page()
#     elif app_page == "Soil Classifier":
#         soil_classifier_page(soil_model)
#     elif app_page == "Crop Advisor":
#         crop_advisor_page(crop_model, yield_model, encoders, scaler)
#
#     # Footer
#     st.sidebar.markdown("---")
#     st.sidebar.markdown("""
#     ### About This App
#     Developed to help farmers make data-driven decisions using AI and machine learning.
#
#     **Key Technologies:**
#     - TensorFlow/Keras for soil classification
#     - Scikit-learn for crop recommendations
#     - Streamlit for interactive interface
#
#     [GitHub Repository](#) | [Contact Us](#)
#     """)
#
#
# if __name__ == "__main__":
#     main()
#


import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import pandas as pd
import joblib
from PIL import Image
import os

# Custom CSS for styling with the new color scheme
st.markdown(f"""
<style>
    body {{
        color: black;
    }}
    .sidebar .sidebar-content {{
        background-color: #A38765;
        color: black;
    }}
    .stButton>button {{
        background-color: #C5B57D;
        color: black;
        border-radius: 5px;
        padding: 0.5rem 1rem;
    }}
    .stButton>button:hover {{
        background-color: #A38765;
        color: white;
    }}
    .stExpander {{
        border: 1px solid #A38765;
        border-radius: 8px;
        padding: 1rem;
        background-color: #BDB2CA;
    }}
    .stAlert {{
        border-radius: 8px;
    }}
    .title-text {{
        color: #A38765;
        font-weight: 700;
    }}
    .feature-card {{
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        background-color: #D0E3CC;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        color: black;
    }}
    .main {{
        background-color: #F7FFDD;
    }}
    .stNumberInput, .stSelectbox, .stSlider {{
        background-color: #DEB8C9;
    }}
</style>
""", unsafe_allow_html=True)


# Load all models and encoders at startup
@st.cache_resource
def load_models():
    try:
        soil_model = load_model('soil_type_model.h5')
        crop_model = joblib.load('crop_model.pkl')
        yield_model = joblib.load('yield_model.pkl')
        encoders = {
            'Season': joblib.load('Season_encoder.pkl'),
            'State': joblib.load('State_encoder.pkl'),
            'Soil_Type': joblib.load('Soil_Type_encoder.pkl'),
            'Crop': joblib.load('Crop_encoder.pkl')
        }
        scaler = joblib.load('feature_scaler.pkl')
        return soil_model, crop_model, yield_model, encoders, scaler
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None, None, None


# Soil type classification function
def classify_soil(image, model):
    img = image.resize((150, 150))
    img_array = img_to_array(img)
    img_array = img_array / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    class_idx = np.argmax(predictions[0])

    soil_types = ['alluvial', 'black', 'cinder', 'clay', 'laterite', 'peat', 'red', 'yellow']
    return soil_types[class_idx]


# Crop recommendation function
def recommend_crops(conditions, model, encoders, scaler):
    try:
        input_df = pd.DataFrame([conditions])

        for col in ['Season', 'State', 'Soil_Type']:
            input_df[col] = encoders[col].transform([conditions[col]])[0]

        numerical_cols = ['Crop_Year', 'Annual_Rainfall', 'Fertilizer', 'Pesticide']
        input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

        features = ['Crop_Year', 'Season', 'State', 'Annual_Rainfall',
                    'Fertilizer', 'Pesticide', 'Soil_Type']
        probas = model.predict_proba(input_df[features])[0]
        crop_probs = list(zip(encoders['Crop'].classes_, probas))

        return sorted(crop_probs, key=lambda x: x[1], reverse=True)[:3]
    except Exception as e:
        st.error(f"Recommendation error: {str(e)}")
        return []


# Yield prediction function
def predict_yield(crop, conditions, model, encoders, scaler):
    try:
        input_df = pd.DataFrame([conditions])
        input_df['Crop'] = crop

        for col in ['Season', 'State', 'Soil_Type', 'Crop']:
            value = crop if col == 'Crop' else conditions[col]
            input_df[col] = encoders[col].transform([value])[0]

        numerical_cols = ['Crop_Year', 'Annual_Rainfall', 'Fertilizer', 'Pesticide']
        input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

        features = ['Crop', 'Crop_Year', 'Season', 'State',
                    'Annual_Rainfall', 'Fertilizer', 'Pesticide', 'Soil_Type']
        return model.predict(input_df[features])[0]
    except Exception as e:
        st.error(f"Yield prediction error: {str(e)}")
        return None


# Home Page
# def home_page():
#     st.markdown("<h1 class='title-text'>🌾 Crop Advisory Model</h1>", unsafe_allow_html=True)
#     st.markdown("""
#     <div class='feature-card'>
#         <h3>Your Intelligent Agricultural Assistant</h3>
#         <p>Our AI-powered platform helps farmers make data-driven decisions for better crop management
#         and improved yields. Combining soil analysis with environmental factors, we provide personalized
#         recommendations for your farming needs.</p>
#     </div>
#     """, unsafe_allow_html=True)
#
#     st.image(
#         "https://www.grabco.co.uk/soil-guide/img/soil-triangle.png",
#         use_column_width=True, caption="Modern Agriculture with AI Assistance")
#
#     st.markdown("""
#     ## Key Features
#
#     <div class='feature-card'>
#         <h4>🌱 Soil Type Classification</h4>
#         <p>Upload images of your soil to automatically determine soil type with our advanced CNN model.</p>
#     </div>
#
#     <div class='feature-card'>
#         <h4>📊 Crop Recommendation Engine</h4>
#         <p>Get personalized crop suggestions based on your soil, location, and farming conditions.</p>
#     </div>
#
#     <div class='feature-card'>
#         <h4>📈 Yield Prediction</h4>
#         <p>Accurate yield forecasts to help with planning and resource allocation.</p>
#     </div>
#     """, unsafe_allow_html=True)
#
#     st.markdown("""
#     ## What Else We Can Do
#
#     <div class='feature-card'>
#         <h4>Future Enhancements</h4>
#         <ul>
#             <li>🌦️ Real-time weather integration for dynamic recommendations</li>
#             <li>💰 Market price trends for crop selection optimization</li>
#             <li>🧪 Fertilizer and pesticide optimization calculator</li>
#             <li>📅 Seasonal planting calendar generator</li>
#             <li>🌍 Regional disease/pest outbreak alerts</li>
#             <li>🤖 Chatbot for agricultural Q&A</li>
#         </ul>
#     </div>
#
#     ## About This Project
#
#     <div class='feature-card'>
#         <p>This agricultural advisory application was developed to help farmers make better decisions
#         using machine learning and AI technologies. The system combines soil analysis with environmental
#         factors to provide personalized recommendations.</p>
#         <p><strong>Technologies used:</strong> TensorFlow/Keras, Scikit-learn, Streamlit</p>
#     </div>
#     """, unsafe_allow_html=True)


def home_page():
    # Hero Section
    # Hero Section
    st.markdown("<h1 class='title-text'>🌾 Smart Crop Advisor</h1>", unsafe_allow_html=True)

    # Intelligent Agricultural Assistant section
    st.markdown("""
    </br></br></br>
       <div class='feature-card'>
           <h3>Your Intelligent Agricultural Assistant</h3>
           <p>AI-powered platform helping farmers make data-driven decisions for better crop 
           management and improved yields through soil analysis and environmental insights.</p>
       </div>
       """, unsafe_allow_html=True)

    # Image placed below the text
    st.image(
        "https://www.grabco.co.uk/soil-guide/img/soil-triangle.png",
        use_container_width=True,
        caption="Soil Classification Guide"
    )

    st.markdown("---")

    # Key Features Section
    st.markdown("<h2 class='title-text'>✨ Core Features</h2>", unsafe_allow_html=True)

    features = [
        {
            "icon": "",
            "title": "Classify Soil Type",
            "desc": "Upload soil images for automatic classification using our advanced CNN model.",
            "color": "#D0E3CC"
        },
        {
            "icon": "",
            "title": "Recommend Crops",
            "desc": "Get personalized suggestions based on soil, location, and farming conditions.",
            "color": "#BDB2CA"
        },
        {
            "icon": "",
            "title": "Yield Prediction",
            "desc": " Accurate yield forecasts to optimize your suggestions based on soil ,location ,and farming conditions..",
            "color": "#DEB8C9"
        }
    ]

    cols = st.columns(3)
    for i, feature in enumerate(features):
        with cols[i]:
            st.markdown(f"""
            <div style='background-color:{feature['color']}; border-radius:10px; padding:1.5rem; margin-bottom:1rem;'>
                <h3>{feature['icon']} {feature['title']}</h3>
                <p>{feature['desc']}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # How It Works Section
    st.markdown("<h2 class='title-text'>🛠️ How It Works</h2>", unsafe_allow_html=True)

    steps = [
        {"icon": "1️⃣", "text": "Upload soil image or enter farm details"},
        {"icon": "2️⃣", "text": "Our AI analyzes the input data"},
        {"icon": "3️⃣", "text": "Get personalized recommendations"},
        {"icon": "4️⃣", "text": "Implement suggestions for better yields"}
    ]

    for step in steps:
        st.markdown(f"""
        <div style='background-color:#46655B; border-radius:8px; padding:1rem; margin:0.5rem 0;'>
            <b>{step['icon']} {step['text']}</b>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Future Enhancements Section
    st.markdown("<h2 class='title-text'>🚀 Future Scope</h2>", unsafe_allow_html=True)

    enhancements = [
        "🌦️ Real-time weather integration",
        "💰 Market price trends analysis",
        "🧪 Fertilizer optimization calculator",
        "📅 Smart planting calendar",
        "🌍 Pest/disease alerts",
        "🤖 Agricultural Q&A chatbot"
    ]

    # st.markdown("""
    # <div style='background-color:#D0E3CC; border-radius:10px; padding:1.5rem;'>
    #     <div style='column-count: 2;'>
    # """, unsafe_allow_html=True)

    for item in enhancements:
        st.markdown(f"<div style='margin-bottom:0.5rem;'>{item}</div>", unsafe_allow_html=True)

    st.markdown("</div></div>", unsafe_allow_html=True)

    st.markdown("---")

    # About Section
    st.markdown("<h2 class='title-text'>📝 About This Project</h2>", unsafe_allow_html=True)

    tech_cols = st.columns(3)
    tech_stack = [
        {"name": "TensorFlow", "use": "Soil Classification"},
        {"name": "Scikit-learn", "use": "Crop Recommendations"},
        {"name": "Streamlit", "use": "Interactive Interface"}
    ]

    for i, tech in enumerate(tech_stack):
        with tech_cols[i]:
            st.markdown(f"""
            <div style='background-color:#C5BDB2; border-radius:8px; padding:1rem; text-align:center;'>
                <h4>{tech['name']}</h4>
                <p>{tech['use']}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("""
    <div style='background-color:#BAA898; border-radius:8px; padding:1.5rem; margin-top:1rem;'>
        <p>This agricultural advisory application was developed to help farmers make better decisions 
        using machine learning and AI technologies. The system combines soil analysis with environmental 
        factors to provide personalized recommendations.</p>
    </div>
    """, unsafe_allow_html=True)


# Soil Classifier Page
def soil_classifier_page(soil_model):
    st.markdown("<h1 class='title-text'>🌍 Soil Type Classifier</h1>", unsafe_allow_html=True)
    st.markdown("""
    <div class='feature-card'>
        <p>Upload a clear photo of your soil to automatically determine its type. 
        Our AI model can identify 8 different soil classifications with high accuracy.</p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Choose a soil image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Soil Image", use_column_width=True)

        if st.button("Analyze Soil", key="analyze_soil"):
            with st.spinner("Processing soil image..."):
                soil_type = classify_soil(image, soil_model)
                st.success(f"""
                **Analysis Results:**
                - Predicted Soil Type: **{soil_type.capitalize()}**
                """)

                # Soil type information
                soil_info = {
                    'alluvial': "Rich in minerals, excellent for rice and wheat",
                    'black': "High clay content, good for cotton and sugarcane",
                    'clay': "Good water retention, ideal for rice and lentils",
                    'laterite': "Rich in iron, suitable for tea and coffee",
                    'red': "Good drainage, works well for millets and oilseeds",
                    'yellow': "Moderate fertility, good for maize and pulses"
                }

                if soil_type in soil_info:
                    st.info(f"**{soil_type.capitalize()} Soil Characteristics:** {soil_info[soil_type]}")

    st.markdown("""
    ### Tips for Best Results:
    - Take photo in natural daylight
    - Capture a close-up of undisturbed soil
    - Include a reference object for scale
    - Avoid shadows or glare

    ### Supported Soil Types:
    - Alluvial
    - Black
    - Cinder
    - Clay
    - Laterite
    - Peat
    - Red
    - Yellow
    """, unsafe_allow_html=True)


# Crop Advisor Page
def crop_advisor_page(crop_model, yield_model, encoders, scaler):
    st.markdown("<h1 class='title-text'>🌱 Crop Recommendation System</h1>", unsafe_allow_html=True)
    st.markdown("""
    <div class='feature-card'>
        <p>Get personalized crop recommendations based on your soil type, location, 
        and agricultural practices. Our system analyzes multiple factors to suggest 
        the most suitable crops for your farm.</p>
    </div>
    """, unsafe_allow_html=True)

    with st.form("crop_recommendation_form"):
        st.subheader("Farm Conditions")

        col1, col2 = st.columns(2)
        with col1:
            soil_type = st.selectbox("Soil Type", encoders['Soil_Type'].classes_)
            season = st.selectbox("Growing Season", encoders['Season'].classes_)
        with col2:
            state = st.selectbox("State/Region", encoders['State'].classes_)
            year = st.number_input("Year", min_value=2000, max_value=2030, value=2023)

        st.subheader("Environmental Factors")
        col1, col2, col3 = st.columns(3)
        with col1:
            rainfall = st.number_input("Annual Rainfall (mm)", min_value=0, value=1500)
        with col2:
            fertilizer = st.number_input("Fertilizer (kg/ha)", min_value=0, value=5000)
        with col3:
            pesticide = st.slider("Pesticide Usage (kg/ha)", 0, 5000, 2000)

        submitted = st.form_submit_button("Get Recommendations")

        if submitted:
            conditions = {
                'Crop_Year': year,
                'Season': season,
                'State': state,
                'Annual_Rainfall': rainfall,
                'Fertilizer': fertilizer,
                'Pesticide': pesticide,
                'Soil_Type': soil_type
            }

            with st.spinner("Analyzing farm conditions..."):
                recommendations = recommend_crops(conditions, crop_model, encoders, scaler)

                if recommendations:
                    st.success("## Recommended Crops")

                    for crop, _ in recommendations:
                        with st.expander(f"**{crop}**", expanded=True):
                            yield_pred = predict_yield(crop, conditions, yield_model, encoders, scaler)

                            col1, col2 = st.columns([1, 2])
                            with col1:
                                if yield_pred:
                                    st.metric("Predicted Yield", f"{yield_pred:.2f} tons/ha")
                                else:
                                    st.warning("Yield data unavailable")

                            with col2:
                                # Crop-specific advice
                                advice = {
                                    "Rice": "Requires standing water, ideal for clay soils",
                                    "Wheat": "Best in well-drained loamy soils",
                                    "Maize": "Adaptable to various soils but needs good nitrogen",
                                    "Cotton": "Thrives in black soils with warm climate",
                                    "Sugarcane": "Needs fertile soil with good water retention"
                                }

                                if crop in advice:
                                    st.info(f"**Growing Tips:** {advice[crop]}")
                                else:
                                    st.info("General advice: Ensure proper irrigation and soil preparation")
                else:
                    st.warning("No strong recommendations available for these conditions")


# Main App
def main():
    st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3079/3079155.png", width=100)
    st.sidebar.title("Crop Advisory Model")

    # Navigation - removed "About This Page" from the sidebar
    app_page = st.sidebar.radio("Navigation",
                                ["Home", "Soil Classifier", "Crop Advisor"],
                                index=0)

    # Load models only when needed
    if app_page != "Home":
        soil_model, crop_model, yield_model, encoders, scaler = load_models()
        if None in [soil_model, crop_model, yield_model, encoders, scaler]:
            st.error("Failed to load required models. Please check your model files.")
            return

    if app_page == "Home":
        home_page()
    elif app_page == "Soil Classifier":
        soil_classifier_page(soil_model)
    elif app_page == "Crop Advisor":
        crop_advisor_page(crop_model, yield_model, encoders, scaler)


if __name__ == "__main__":
    main()
    