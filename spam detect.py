import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# 1. Sample Dataset (Replace this with your actual CSV file loading)
# Example data: 'text' contains the email content, 'label' is 0 for ham (good) and 1 for spam
data={
    'text': [
        'Hey, are we still meeting for lunch today at 1 PM?',
        'URGENT! Your credit card account has been compromised. Click here now!',
        'Can you send me the final report by end of day?',
        'WINNER!!! You have won a free vacation to paradise! Call now to claim.',
        'Dear friend, please find attached the invoice for last month.',
        'Get rich quick! Make $5000 a day working from home. No experience needed!'
    ],
    'label':[0, 1, 0, 1, 0, 1]
}
df=pd.DataFrame(data)

x=df['text']
y=df['label']

# 2. Split data into training (80%) and testing (20%) sets
x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.2, random_state=42)

# 3. Convert text data into numerical 
vectorizer=CountVectorizer()
x_train_counts=vectorizer.fit_transform(x_train)
x_test_counts=vectorizer.transform(x_test)

# 4. Initialize and train the Naive Bayes Classifier
model=MultinomialNB()
model.fit(x_train_counts, y_train)

# 5. Evaluate the model on test data
predictions=model.predict(x_test_counts)
print(f"Model Accuracy: {accuracy_score(y_test, predictions):.2f}\n")

# 6. Test the model with brand new emails
new_emails=[
    "Congratulations! You won a cash prize of one million dollars.",
    "Hi team, just a reminder that the project deadline is tomorrow morning."
]

# Convert new emails using the exact same vectorizer instance
new_emails_counts=vectorizer.transform(new_emails)
new_predictions=model.predict(new_emails_counts)

# Display the classification results
for email, prediction in zip(new_emails, new_predictions):
    category="SPAM" if prediction==1 else "HAM (Legitimate)"
    print(f"Email: '{email}'-> Result: {category}")