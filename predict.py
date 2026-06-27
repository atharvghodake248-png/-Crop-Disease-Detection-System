import tensorflow as tf
import numpy as np
import json
from tensorflow.keras.utils import load_img, img_to_array

# Load model
model = tf.keras.models.load_model("crop_model.keras")

# Load class mapping
with open("class_names.json", "r") as f:
    class_indices = json.load(f)

# Reverse mapping
class_names = {v: k for k, v in class_indices.items()}

IMG_SIZE = (224, 224)

def predict_image(img_path):
    img = load_img(img_path, target_size=IMG_SIZE)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)

    print("Prediction:", class_names[class_index])

img_path = input("Enter image path: ")
predict_image(img_path)