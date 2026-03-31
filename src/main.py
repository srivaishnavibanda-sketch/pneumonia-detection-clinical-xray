import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate, GlobalAveragePooling2D
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical

# =========================
# PATHS
# =========================
IMG_DIR = "data/images_jpg"
CSV_PATH = "data/clinical_data.csv"

IMG_SIZE = (224,224)

print("🔄 Loading CSV...")
df = pd.read_csv(CSV_PATH)

images, clinical, labels = [], [], []

print("⚡ Fast loading images (Hackathon Mode)...")

# 🚀 LOAD ONLY 500 IMAGES
for i, row in df.iterrows():

    if i == 500:
        break

    if i % 100 == 0:
        print(f"Loaded {i} images")

    img_path = os.path.join(IMG_DIR, row["image_id"])

    if os.path.exists(img_path):

        img = load_img(img_path, target_size=IMG_SIZE)
        img = img_to_array(img)/255.0
        images.append(img)

        clinical_vector = [
            row["age"]/100,
            1 if row["gender"]=="M" else 0,
            row["fever"]/110,
            row["spo2"]/100,
            row["wbc"]/15,
            row["urinalysis"]
        ]

        clinical.append(clinical_vector)
        labels.append(row["label"])

images = np.array(images)
clinical = np.array(clinical)
labels = to_categorical(labels)

print("✅ Data Ready")

# =========================
# SPLIT
# =========================
x_img_train, x_img_test, x_clin_train, x_clin_test, y_train, y_test = train_test_split(
    images, clinical, labels, test_size=0.2, random_state=42
)

# =========================
# MODEL
# =========================
print("🧠 Building Model...")

img_input = Input(shape=(224,224,3))

base_model = EfficientNetB0(
    weights="imagenet",
    include_top=False,
    input_tensor=img_input
)

# 🚀 FREEZE BACKBONE (SUPER FAST)
for layer in base_model.layers:
    layer.trainable = False

x = GlobalAveragePooling2D()(base_model.output)
x = Dense(64, activation="relu")(x)
img_out = Dense(32, activation="relu")(x)

clin_input = Input(shape=(6,))
clin_out = Dense(32, activation="relu")(clin_input)

combined = Concatenate()([img_out, clin_out])
x = Dense(32, activation="relu")(combined)
output = Dense(2, activation="softmax")(x)

model = Model(inputs=[img_input, clin_input], outputs=output)

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# =========================
# TRAIN
# =========================
print("🚀 FAST TRAINING STARTED...")

model.fit(
    [x_img_train, x_clin_train],
    y_train,
    validation_split=0.1,
    epochs=3,        # ⚡ FAST
    batch_size=16
)

# =========================
# EVALUATE
# =========================
print("📊 Evaluating...")

pred = model.predict([x_img_test, x_clin_test])

print(classification_report(
    np.argmax(y_test,axis=1),
    np.argmax(pred,axis=1)
))

# =========================
# SAVE MODEL (NEW FORMAT)
# =========================
os.makedirs("model", exist_ok=True)

model.save("model/pneumonia_model")

print("✅ MODEL SAVED SUCCESSFULLY")
