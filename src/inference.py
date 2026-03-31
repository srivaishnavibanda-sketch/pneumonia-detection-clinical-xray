from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import pandas as pd
import os

# Load saved model
model = load_model('model/pneumonia_model.h5')

# Load new data (replace with your actual new data)
IMG_DIR = "data/images_jpg"
CSV_PATH = "data/clinical_data.csv"  # or your test dataset

df = pd.read_csv(CSV_PATH)

IMG_SIZE = (224, 224)

for _, row in df.iterrows():
    img_path = os.path.join(IMG_DIR, row['image_id'])
    if os.path.exists(img_path):
        # Prepare image
        img = load_img(img_path, target_size=IMG_SIZE)
        img = img_to_array(img) / 255.0
        img = np.expand_dims(img, axis=0)  # add batch dimension

        # Prepare clinical data vector (same normalization as training)
        clinical_vector = np.array([[
            row["age"] / 100,
            1 if row["gender"] == "M" else 0,
            row["fever"] / 110,
            row["spo2"] / 100,
            row["wbc"] / 15,
            row["urinalysis"]
        ]])

        # Predict
        pred = model.predict([img, clinical_vector])
        label = np.argmax(pred, axis=1)[0]
        print(f"Image: {row['image_id']} - Predicted label: {label}")
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input
import os

# Load the trained model
model = load_model('pneumonia_model.h5')

# Define the image and clinical data directories
IMG_DIR = "data/images_jpg"
CSV_PATH = "data/clinical_data.csv"

# Function to preprocess the image
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = preprocess_input(img)  # Preprocess for EfficientNet
    return img

# Function to preprocess clinical data
def preprocess_clinical_data(row):
    clinical_vector = [
        row["age"] / 100,
        1 if row["gender"] == "M" else 0,
        row["fever"] / 110,
        row["spo2"] / 100,
        row["wbc"] / 15,
        row["urinalysis"]
    ]
    return np.array(clinical_vector).reshape(1, -1)  # Reshape to match input shape

# Example usage
image_id = "0004cfab-14fd-4e49-80ba-63a80b6bddd6.jpg"
clinical_data_row = {
    "age": 71,
    "gender": "M",
    "fever": 98.0,
    "spo2": 99,
    "wbc": 11.16,
    "urinalysis": 0.74
}

# Preprocess inputs
image_path = os.path.join(IMG_DIR, image_id)
img = preprocess_image(image_path)
clinical_data = preprocess_clinical_data(clinical_data_row)

# Make prediction
prediction = model.predict([img, clinical_data])
predicted_label = np.argmax(prediction, axis=1)

# Output the result
print(f"Predicted label: {predicted_label}")
