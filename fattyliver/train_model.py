# train_model.py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Data generator: rescale pixel values
train_datagen = ImageDataGenerator(rescale=1./255)

# Load dataset from folders
train_generator = train_datagen.flow_from_directory(
    'dataset/train',          # path to dataset
    target_size=(128,128),    # resize images
    batch_size=2,             # small batch since only 4 images
    class_mode='binary'       # two classes: normal/fatty
)

# Build CNN model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')  # binary classification
])

# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model
model.fit(train_generator, epochs=10)  # for small dataset, 10 epochs is enough

# Save trained model
model.save('liver_model.h5')
print("Model saved as liver_model.h5")
