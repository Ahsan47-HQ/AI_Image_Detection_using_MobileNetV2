import os
import shutil
from sklearn.model_selection import train_test_split

# Paths
TEST_DIR = "data/test_old"
VAL_DIR = "data/val"
NEW_TEST_DIR = "data/test"

classes = ["REAL", "FAKE"]

# Create folders
for cls in classes:
    os.makedirs(os.path.join(VAL_DIR, cls), exist_ok=True)
    os.makedirs(os.path.join(NEW_TEST_DIR, cls), exist_ok=True)

# Process each class separately
for cls in classes:
    class_path = os.path.join(TEST_DIR, cls)
    files = os.listdir(class_path)

    # Split 50-50 (we want val and test both)
    val_files, test_files = train_test_split(
        files,
        test_size=0.5,
        random_state=42
    )

    # Move files
    for f in val_files:
        shutil.move(
            os.path.join(class_path, f),
            os.path.join(VAL_DIR, cls, f)
        )

    for f in test_files:
        shutil.move(
            os.path.join(class_path, f),
            os.path.join(NEW_TEST_DIR, cls, f)
        )

print("Split complete: test_old -> val + test")