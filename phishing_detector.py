# Import Libraries
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Sample Dataset
# -----------------------------

data = {
    'email': [
        'Congratulations! You won a free iPhone. Click now!',
        'Your bank account is suspended. Verify immediately.',
        'Meeting scheduled tomorrow at 10 AM.',
        'Project submission deadline is Friday.',
        'Win cash prizes instantly by clicking here.',
        'Team lunch planned for next week.',
        'Update your password urgently.',
        'Invoice attached for your recent purchase.'
    ],

    'label': [
        'Phishing',
        'Phishing',
        'Safe',
        'Safe',
        'Phishing',
        'Safe',
        'Phishing',
        'Safe'
    ]
}

# -----------------------------
# Convert to DataFrame
# -----------------------------

df = pd.DataFrame(data)

print("\nDataset:\n")
print(df)

# -----------------------------
# Features and Labels
# -----------------------------

X = df['email']
y = df['label']

# -----------------------------
# Convert Text into Numerical Features
# -----------------------------

vectorizer = TfidfVectorizer()

X_vectorized = vectorizer.fit_transform(X)

# -----------------------------
# Split Dataset
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized,
    y,
    test_size=0.25,
    random_state=42
)

# -----------------------------
# Train Model
# -----------------------------

model = MultinomialNB()

model.fit(X_train, y_train)

# -----------------------------
# Predictions
# -----------------------------

y_pred = model.predict(X_test)

# -----------------------------
# Accuracy
# -----------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", round(accuracy * 100, 2), "%")

# -----------------------------
# Classification Report
# -----------------------------

print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)

# -----------------------------
# Confusion Matrix
# -----------------------------

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=['Phishing', 'Safe']
)

print("\nConfusion Matrix:\n")
print(cm)

# -----------------------------
# Plot Confusion Matrix
# -----------------------------

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['Phishing', 'Safe'],
    yticklabels=['Phishing', 'Safe']
)

plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
plt.title('Phishing Email Detection')

plt.tight_layout()

plt.show()

# -----------------------------
# Test Custom Email
# -----------------------------

sample_email = [
    "Your PayPal account has been locked. Click here to verify."
]

# Convert email into vector
sample_vector = vectorizer.transform(sample_email)

# Predict
prediction = model.predict(sample_vector)

# -----------------------------
# Display Result
# -----------------------------

print("\nCustom Email:")
print(sample_email[0])

print("\nPrediction Result:")

if prediction[0] == "Phishing":
    print("Phishing Email Detected")
else:
    print("Safe Email")