from flask import Flask, render_template, request
import os 
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# ---------------- PATHS ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "pneumonia_model"
)

# ---------------- LOAD SAVEDMODEL ----------------
model = tf.saved_model.load(MODEL_PATH)

# inference function
infer = model.signatures["serving_default"]

print("✅ SavedModel Loaded Successfully")
print(infer.structured_input_signature)


# ---------------- HOME ----------------
@app.route("/")
def home():
    return render_template("index.html")


# ---------------- CLINICAL PAGE ----------------
@app.route("/clinical")
def clinical():
    return render_template("clinical.html")


# ---------------- PREDICTION FUNCTION ----------------
def predict_pneumonia(image_path, clinical_features):

    # ---- Image preprocessing ----
    img = image.load_img(image_path, target_size=(224, 224))
    img = image.img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0).astype(np.float32)

    # ---- Clinical preprocessing ----
    clinical_features = np.array(clinical_features)\
        .reshape(1, -1).astype(np.float32)

    # ---- MODEL INFERENCE ----
    prediction_dict = infer(
        input_1=tf.constant(img),
        input_2=tf.constant(clinical_features)
    )

    prediction = list(prediction_dict.values())[0].numpy()[0][0]

    if prediction > 0.5:
        result = "Pneumonia Detected"
        confidence = prediction
    else:
        result = "Normal"
        confidence = 1 - prediction

    return result, float(confidence)


# ---------------- PREDICT ROUTE ----------------
@app.route("/predict", methods=["POST"])
def predict():

    file = request.files["image"]
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    # ---- Clinical inputs (default values allowed) ----
    age = float(request.form.get("age", 30))
    gender = request.form.get("gender", "M")
    fever = float(request.form.get("fever", 98))
    spo2 = float(request.form.get("spo2", 98))
    wbc = float(request.form.get("wbc", 7))
    urinalysis = float(request.form.get("urinalysis", 1))

    # encode gender
    gender = 1 if gender == "M" else 0

    clinical_features = [
        age,
        gender,
        fever,
        spo2,
        wbc,
        urinalysis
    ]

    result, confidence = predict_pneumonia(filepath, clinical_features)

    return render_template(
        "result.html",
        result=result,
        confidence=confidence,
        image_path=file.filename
    )


# ---------------- RUN APP ----------------
if __name__ == "__main__":
    app.run(debug=True)
