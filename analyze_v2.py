import os
import numpy as np
from sklearn.model_selection import train_test_split
from config import PROCESSED_DIR

patient_ids = np.load(os.path.join(PROCESSED_DIR, "patient_ids.npy"), allow_pickle=True)
unique_patients = np.unique(patient_ids)
train_p, temp_p = train_test_split(unique_patients, test_size=0.2, random_state=42)
val_p, test_p = train_test_split(temp_p, test_size=0.5, random_state=42)

print(f"Total test patients: {len(test_p)}")
print(test_p)   # <- no slicing, shows all of them