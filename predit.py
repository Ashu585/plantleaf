import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np

# Set the path to the dataset directory
dataset_dir = r"C:\Users\lap2\Desktop\Apple___Black_rot"

# Set the image and batch size
img_height, img_width = 224, 224
batch_size = 25

# Define the labels
labels = [
    "Apple___Infected",
    "Apple___Healthy",
]

# Data augmentation and preprocessing
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    #20% of data used for validation durning traing
    validation_split=0.2
)
# Load the dataset using the data generator
train_generator = datagen.flow_from_directory(
    dataset_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    dataset_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)
model = tf.keras.Sequential([
    tf.keras.applications.ResNet50( include_top=False,weights='imagenet',input_shape=(img_height, img_width, 3)),
    tf.keras.layers.GlobalAveragePooling2D(),
    #capture the complex features
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(len(labels), activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
num_epochs = 10
history = model.fit(
    train_generator,
    epochs=num_epochs,
    validation_data=validation_generator
)
model.save('model.h5')
test_image_path = r"C:\Users\lap2\Desktop\internship\infec1\infec\acb4845c-bee1-478c-b11c-ade0ae397c51___JR_FrgE.S 8818.JPG"
test_image = tf.keras.preprocessing.image.load_img(test_image_path, target_size=(img_height, img_width))
test_image = tf.keras.preprocessing.image.img_to_array(test_image)
test_image = test_image / 255.0
test_image = tf.expand_dims(test_image, axis=0)

predicted_probabilities = model.predict(test_image)[0]
predicted_label_index = tf.argmax(predicted_probabilities)
predicted_label = labels[predicted_label_index]

print("Predicted label:", predicted_label)