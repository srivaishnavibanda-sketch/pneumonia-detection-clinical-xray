import os
import pydicom
import numpy as np
from PIL import Image
from tqdm import tqdm

# Input DICOM folder
input_dir = 'data/images/stage_2_train_images'
# Output JPG folder
output_dir = 'data/images_jpg'

os.makedirs(output_dir, exist_ok=True)

def dcm_to_jpg(dcm_path, jpg_path):
    try:
        dcm = pydicom.dcmread(dcm_path)
        img = dcm.pixel_array.astype(float)

        # Normalize
        img = (np.maximum(img, 0) / img.max()) * 255.0
        img = np.uint8(img)

        img_pil = Image.fromarray(img).convert("RGB")
        img_pil.save(jpg_path)
    except Exception as e:
        print(f"Failed to convert {dcm_path}: {e}")

# Loop through all .dcm files
for filename in tqdm(os.listdir(input_dir)):
    if filename.endswith(".dcm"):
        dcm_path = os.path.join(input_dir, filename)
        jpg_path = os.path.join(output_dir, filename.replace(".dcm", ".jpg"))
        dcm_to_jpg(dcm_path, jpg_path)

print("✅ All DICOM files converted to JPG.")
