import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

dataset_path = "C:/Users/user/PycharmProjectshelloworld/SOIL MODEL 1/soil_updated"    # Path to the "Soil_types" folder

# path = "C:/Users/user/PycharmProjectshelloworld/SOIL MODEL 1/soil_updated"

train_dir = f"{dataset_path}/train"

train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=(150, 150),  # Resize all images to 150x150
                                                    batch_size=32,
                                                    class_mode='categorical')

# Build the CNN Model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(8, activation='softmax')  # Adjust the number of output neurons to match your soil types (8 classes)
])


# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#  Train the model
model.fit(train_generator, epochs=10)

# Save the model
model.save('soil_type_model.h5')
