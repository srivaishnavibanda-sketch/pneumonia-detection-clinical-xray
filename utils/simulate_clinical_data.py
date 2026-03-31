import pandas as pd
import random
import numpy as np

# Load image labels
df = pd.read_csv('data/image_labels.csv')

# Simulate clinical features
def simulate_features():
    return {
        "age": random.randint(1, 90),
        "gender": random.choice(["M", "F"]),
        "fever": round(random.uniform(97.0, 104.0), 1),
        "spo2": random.randint(80, 100),
        "wbc": round(random.uniform(4.0, 12.0), 2),  # in 10^9/L
        "urinalysis": round(random.uniform(0.0, 1.0), 2)
    }

# Add simulated features to each image
clinical_data = []
for _, row in df.iterrows():
    features = simulate_features()
    features["image_id"] = row["image_id"]
    features["label"] = row["label"]
    clinical_data.append(features)

# Save to CSV
clinical_df = pd.DataFrame(clinical_data)
clinical_df.to_csv("data/clinical_data.csv", index=False)

print("✅ Simulated clinical data saved to data/clinical_data.csv")
