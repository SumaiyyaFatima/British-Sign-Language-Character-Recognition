import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Load raw CSV
data = []
labels = []

with open("BSL_Dataset.csv", "r") as file:
    for line in file:
        if line.startswith("#") or line.strip() == "":
            continue
        parts = line.strip().split(",")
        if len(parts) < 64:
            continue  # Skip if no label
        *landmarks, label = parts

        try:
            landmarks = [float(x) for x in landmarks]
            if len(landmarks) == 63:  # One hand
                landmarks += [0.0] * 63  # Pad to match two-hand format
            elif len(landmarks) != 126:
                continue  # Skip malformed rows
            data.append(landmarks)
            labels.append(label.strip())
        except ValueError:
            continue  # Skip rows with invalid numbers

print(f"✅ Loaded {len(data)} samples")

# Convert to arrays
X = np.array(data, dtype=np.float32)
y = np.array(labels)

# Normalize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Train classifier
clf = RandomForestClassifier(n_estimators=150, random_state=42)
clf.fit(X_train, y_train)

# Accuracy
acc = clf.score(X_test, y_test)
print(f"✅ Model Accuracy: {acc:.2f}")

# === Model Performance Analysis ===
# Predict on test set
y_pred = clf.predict(X_test)

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=np.unique(y))
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=np.unique(y), yticklabels=np.unique(y))  # type: ignore
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Class balance
print("\nClass Balance:")
print(Counter(y))

# Save model and scaler
joblib.dump(clf, "bsl_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("✅ Model and scaler saved!")
