import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# 📂 Load dataset
data = pd.read_csv("spam.csv")

# Convert labels to numbers
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Features and target
X = data['message']
y = data['label']

# Convert text to numbers
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Test with custom input
msg = ["Congratulations! You won a free ticket"]
msg_vec = vectorizer.transform(msg)
prediction = model.predict(msg_vec)

if prediction[0] == 1:
    print("Spam Message 🚫")
else:
    print("Not Spam ✅")