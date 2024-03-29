# Sms_Spam_Detection

## Overview

This script is designed to perform SMS spam detection using a machine learning model based on the RandomForestClassifier. It utilizes a dataset containing SMS messages labeled as spam or ham (non-spam) to train the model.

## Requirements

- **Python:** Make sure you have Python installed on your system. You can download Python from [python.org](https://www.python.org/).
  
- **Libraries:** Install required Python libraries using the following command:
  ```bash
  pip install pandas scikit-learn joblib chardet

## Steps to Run

1. **Clone Repository:**
   - Clone or download the repository containing the "sms_spam_detection.py" script.

2. **Navigate to the Directory:**
   - Open a terminal or command prompt and navigate to the directory where "sms_spam_detection.py" is located.

3. **Run the Script:**
   - Execute the script using the following command:
     ```bash
     python sms_spam_detection.py
     ```

## Explanation

1. **Import Libraries:**
   - The script starts by importing necessary libraries, including Pandas for data manipulation, scikit-learn for machine learning, and joblib for model saving/loading.

2. **Load Dataset:**
   - The dataset is loaded from a CSV file containing SMS messages labeled as spam or ham. The file path should be specified in the `file_path` variable.

3. **Detect Encoding:**
   - The script uses the `chardet` library to automatically detect the encoding of the CSV file.

4. **Clean Data:**
   - Unnamed columns are dropped, and the remaining columns are displayed.

5. **Prepare Data:**
   - The script separates the dataset into features (X) and labels (y). It then splits the data into training and testing sets.

6. **Convert Text to Features:**
   - The CountVectorizer is used to convert text data into numerical features, creating a bag-of-words representation.

7. **Train Model:**
   - A RandomForestClassifier is trained on the training set.

8. **Make Predictions:**
   - The trained model is used to make predictions on the test set.

9. **Evaluate Model:**
   - The accuracy and classification report are printed to evaluate the model's performance.

10. **Save Model:**
    - The trained model is saved using joblib for future use.

## Additional Notes

- **CSV File Path:**
  - Ensure that the CSV file path in `file_path` is correct.

- **Dataset Columns:**
  - The script assumes the dataset has columns 'v1' for labels and 'v2' for text messages.

- **Hyperparameter Adjustment:**
  - You can adjust hyperparameters like the number of estimators in the RandomForestClassifier to improve the model's performance.
## Author

-  Anu-shalini-12





