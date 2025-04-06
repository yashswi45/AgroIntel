import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras.preprocessing import image

# Model training (assuming model is trained and saved as 'soil_type_model.h5')

# Load the trained model
model = tf.keras.models.load_model('soil_type_model.h5')

# Load and preprocess the image
img_path = "C:/Users/user/OneDrive/Pictures/Others/alluvial-soil-img3.jpg"
# img_path = "C:/Users/user/OneDrive/Desktop/ysoil.jpg"

img = image.load_img(img_path, target_size=(150, 150))
img_array = image.img_to_array(img)
img_array = img_array / 255.0

# Predict the soil type
predicted_class = model.predict(np.expand_dims(img_array, axis=0))
predicted_label = np.argmax(predicted_class)

# Mapping class indices to soil types (adjust this list as per your dataset)
soil_types = ['alluvial', 'black', 'cinder', 'clay', 'laterite', 'peat', 'red', 'yellow']

# Output the predicted soil type
print(f"Predicted Soil Type: {soil_types[predicted_label]}")
