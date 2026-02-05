import os
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm


print("Current working directory:", os.getcwd())
print("Dataset exists?", os.path.exists("Dataset/training_set"))

# Paths
DATASET_PATH = "Dataset/Training_set/training_set"
IMG_SIZE = 64

data = []
labels = []

print("Loading images...")

for label, folder in enumerate(["cats", "dogs"]):
    folder_path = os.path.join(DATASET_PATH, folder)

    for img_name in tqdm(os.listdir(folder_path)[:2000]):
        try:
            img_path = os.path.join(folder_path, img_name)
            img = cv2.imread(img_path)

            # Convert to grayscale
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Resize
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

            # Flatten
            img = img.flatten()

            data.append(img)
            labels.append(label)

        except:
            pass

data = np.array(data)
labels = np.array(labels)

print("Total images:", len(data))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=42
)

print("Training SVM model...")

# SVM model
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)

print("Model training completed!")

# Predictions
y_pred = svm.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
