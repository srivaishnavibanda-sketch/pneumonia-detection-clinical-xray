import pandas as pd

# Load RSNA label file
labels_df = pd.read_csv('data/images/stage_2_train_labels.csv')

# We will assign pneumonia = 1, no pneumonia = 0
# Each image might appear more than once, so group by Image and look for any Target = 1
label_summary = labels_df.groupby('patientId')['Target'].max().reset_index()

# Rename columns for consistency
label_summary.columns = ['image_id', 'label']

# Add .jpg to match converted image names
label_summary['image_id'] = label_summary['image_id'] + '.jpg'

# Save to CSV
label_summary.to_csv('data/image_labels.csv', index=False)

print("✅ Saved labels to data/image_labels.csv")
