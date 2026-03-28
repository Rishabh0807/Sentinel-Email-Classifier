import pandas as pd
# 1. Load data (use the path where you uploaded the csv)
df = pd.read_csv('spam.csv', encoding='latin-1')

# 2. Drop useless columns often found in this dataset
df = df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)

# 3. Rename columns for clarity
df.columns = ['label', 'message']

# 4. Convert labels to numbers (Spam = 1, Ham = 0)
df['label'] = df['label'].map({'ham': 0, 'spam': 1})
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

# Initialize the vectorizer (this removes common words like 'the', 'is', 'at')
cv = CountVectorizer(stop_words='english')

# Convert the text into a matrix of token counts
X = cv.fit_transform(df['message']) 
y = df['label']

# Split into Training (80%) and Testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.naive_bayes import MultinomialNB

# Initialize and train
model = MultinomialNB()
model.fit(X_train, y_train)

# Check accuracy
print(f"Model Accuracy: {model.score(X_test, y_test) * 100:.2f}%")
def predict_spam(sample_text):
    data = cv.transform([sample_text]).toarray()
    prediction = model.predict(data)
    return "SPAM 🚨" if prediction[0] == 1 else "NOT SPAM ✅"

# Test it!
print(predict_spam("Congratulations! You won a 1000 rupee voucher. Click here to claim."))
print(predict_spam("Hey, are we still meeting for the AdVITya event at 5 PM?"))
import joblib

# Save the model
joblib.dump(model, 'spam_model.pkl')

# Save the vectorizer (CRUCIAL: you need this to process new text)
joblib.dump(cv, 'vectorizer.pkl')

print("✅ Files saved: spam_model.pkl and vectorizer.pkl")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Generate the matrix
cm = confusion_matrix(y_test, model.predict(X_test))

# Plot it
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - 98.03% Accuracy')
plt.show()
from sklearn.metrics import classification_report

print(classification_report(y_test, model.predict(X_test)))
def secure_predict(text):
    if not text.strip():
        return "Empty message"
    
    # Pre-process just like the training data
    vectorized_text = cv.transform([text])
    prediction = model.predict(vectorized_text)
    
    # Get probability (optional flex)
    probability = model.predict_proba(vectorized_text)[0][prediction[0]]
    
    result = "SPAM" if prediction[0] == 1 else "HAM"
    return f"{result} (Confidence: {probability*100:.2f}%)"

# Test it
print(secure_predict("Hurry! Free lunch at the VIT Canteen for first 10 students!"))
import re

def is_valid_email(email):
    # Standard email pattern: letters/numbers + @ + domain + .com/in/org
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email)

def run_final_demo():
    print("--- 🛡️ VIT AI Spam Shield 🛡️ ---")
    
    # 1. Ask for the Email ID
    user_email = input("Enter your Email ID: ").strip()
    
    # 2. Validate Format
    if not is_valid_email(user_email):
        print("❌ Invalid Email format! Please use 'example@domain.com'.")
        return

    # 3. Ask for the Email Content
    print(f"\nLogged in as: {user_email}")
    email_content = input("Paste the Email Body/Content here: ").strip()

    if not email_content:
        print("❌ Email content cannot be empty.")
        return

    # 4. AI Prediction Logic
    # (Using the 'cv' and 'model' variables from your previous cells)
    vectorized_input = cv.transform([email_content])
    prediction = model.predict(vectorized_input)[0]
    
    # Get Confidence
    prob = model.predict_proba(vectorized_input)[0]
    confidence = prob[prediction] * 100

    # 5. Output Result
    print("\n--- 🧠 AI Analysis Report ---")
    if prediction == 1:
        print(f"RESULT: 🚨 SPAM / PHISHING DETECTED")
        print(f"VERDICT: This email looks like a scam. Do not click any links.")
    else:
        print(f"RESULT: ✅ SAFE / HAM")
        print(f"VERDICT: This email appears to be legitimate.")
    
    print(f"CONFIDENCE SCORE: {confidence:.2f}%")
    print("----------------------------")

# Run the demo
run_final_demo()