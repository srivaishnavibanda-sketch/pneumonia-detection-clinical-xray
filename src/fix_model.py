import tensorflow as tf
from tensorflow.keras.models import load_model

print("Loading old model...")

# Load WITHOUT compilation
model = load_model("model/pneumonia_model.h5", compile=False)

print("Saving new compatible model...")

# Save again in new format
model.save("model/pneumonia_model_fixed.h5")

print("✅ Model fixed successfully!")
