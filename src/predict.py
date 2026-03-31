import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input

def preprocess_sample(image_path, clinical_data):
    IMG_SIZE = (224, 224)
    img = load_img(image_path, target_size=IMG_SIZE)
    img = img_to_array(img)
    img = preprocess_input(img)  # Important: EfficientNet preprocessing
    img = np.expand_dims(img, axis=0)

    clinical_norm = np.array([[
        clinical_data['age'] / 100,
        1 if clinical_data['gender'] == 'M' else 0,
        clinical_data['fever'] / 110,
        clinical_data['spo2'] / 100,
        clinical_data['wbc'] / 15,
        clinical_data['urinalysis']
    ]])

    return img, clinical_norm

def predict(image_path, clinical_data):
    model = load_model("model/pneumonia_model.h5")
    img_input, clin_input = preprocess_sample(image_path, clinical_data)
    prediction = model.predict([img_input, clin_input])
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction) * 100
    return predicted_class, confidence, prediction

if __name__ == "__main__":
    sample_image = "data/images_jpg/0004cfab-14fd-4e49-80ba-63a80b6bddd6.jpg"
    sample_clinical = {
        'age': 71,
        'gender': 'M',
        'fever': 98.0,
        'spo2': 99,
        'wbc': 11.16,
        'urinalysis': 0.74
    }
    pred_class, confidence, pred_probs = predict(sample_image, sample_clinical)
    labels = ["Normal", "Pneumonia"]
    print(f"Prediction: {labels[pred_class]} with confidence {confidence:.2f}%")
