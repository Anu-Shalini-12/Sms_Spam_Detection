# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import chardet

# Load the dataset
file_path = r'C:\Users\0042H8744\spam.csv'

# Detect the encoding
with open(file_path, 'rb') as f:
    result = chardet.detect(f.read())

# Print the detected encoding
print(f"Detected encoding: {result['encoding']}")

# Read the CSV file with the detected encoding
df = pd.read_csv(file_path, encoding=result['encoding'])

# Drop unnamed columns
df = df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)

# Print the column names
print("Column names:", df.columns)

# Assuming your dataset has columns 'v2' and 'v1'
X = df['v2']
y = df['v1']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert text data to numerical features using CountVectorizer
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Use RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train_vec, y_train)

# Make predictions on the test set
y_pred = classifier.predict(X_test_vec)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Classification Report:\n", report)

# Save the trained model for future use
model_filename = 'sms_spam_detection_model.joblib'
joblib.dump(classifier, model_filename)
print(f"Model saved as {model_filename}")
