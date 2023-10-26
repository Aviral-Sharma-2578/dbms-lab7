import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier  # Import the Random Forest classifier
from sklearn.metrics import confusion_matrix, classification_report

# Initialize custom stopwords
custom_stopwords = set()

# Load custom stopwords from a text file
custom_stopwords_file = "custom_stopwords.txt"
if os.path.exists(custom_stopwords_file):
    with open(custom_stopwords_file, 'r', encoding='utf-8') as f:
        custom_stopwords = set(f.read().splitlines())

# Load email data and preprocess text (including custom stopwords)
def load_data(directory):
    emails = []
    labels = []
    for file in os.listdir(directory):
        with open(os.path.join(directory, file), 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            emails.append(content)
            if "ham" in directory:
                labels.append(0)  # 0 for ham
            else:
                labels.append(1)  # 1 for spam
    return emails, labels

# Load email data from train and test directories
train_ham_emails, train_ham_labels = load_data("train-mails-ham")
train_spam_emails, train_spam_labels = load_data("train-mails-spam")
test_ham_emails, test_ham_labels = load_data("test-mails-ham")
test_spam_emails, test_spam_labels = load_data("test-mails-spam")

# Combine train and test emails and labels
train_emails = train_ham_emails + train_spam_emails
train_labels = train_ham_labels + train_spam_labels
test_emails = test_ham_emails + test_spam_emails
test_labels = test_ham_labels + test_spam_labels

# Create TF-IDF features
tfidf_vectorizer = TfidfVectorizer(stop_words=custom_stopwords)

train_X = tfidf_vectorizer.fit_transform(train_emails)
test_X = tfidf_vectorizer.transform(test_emails)
train_y = np.array(train_labels)
test_y = np.array(test_labels)

# Train a Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)  # You can adjust the number of trees and other parameters

clf.fit(train_X, train_y)

# Predict the labels on the test data
predicted_y = clf.predict(test_X)

# Calculate the confusion matrix
confusion = confusion_matrix(test_y, predicted_y)

# Extract values from the confusion matrix
tn, fp, fn, tp = confusion.ravel()

# Print confusion matrix
print("Confusion Matrix:")
print(confusion)

# Print detailed report
print("\nClassification Report:")
print(classification_report(test_y, predicted_y, target_names=["ham", "spam"]))

# Print true positives, true negatives, false positives, and false negatives
print("\nTrue Negatives:", tn)
print("False Positives:", fp)
print("False Negatives:", fn)
print("True Positives:", tp)
