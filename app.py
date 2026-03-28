import streamlit as st
import joblib

# 1. Load the Trained AI (The .pkl files you uploaded to GitHub)
@st.cache_resource
def load_models():
    model = joblib.load('spam_model.pkl')
    cv = joblib.load('vectorizer.pkl')
    return model, cv

try:
    model, cv = load_models()
except Exception as e:
    st.error("Error loading model files. Make sure .pkl files are in the repository.")

# 2. Website Interface Setup
st.set_page_config(page_title="Sentinel Email Classifier", page_icon="🛡️")

st.title("🛡️ Sentinel: AI Email Spam Shield")
st.subheader("BTech First Year Project | VIT 2026")

st.markdown("""
Welcome to the Sentinel Classifier. This tool uses **Natural Language Processing (NLP)** to identify if an email is legitimate or a potential security threat.
""")

# 3. THE INPUT BOX (This replaces the input() function)
st.write("---")
email_text = st.text_area(
    "Paste the email content you want to analyze below:", 
    height=250, 
    placeholder="Example: Congratulations! You've won a $1,000 gift card..."
)

# 4. PREDICTION LOGIC
if st.button("Run Security Scan"):
    if email_text.strip():
        # Step 1: Convert text to numbers using the loaded Vectorizer
        data = cv.transform([email_text])
        
        # Step 2: Predict using the loaded Model
        prediction = model.predict(data)
        
        # Step 3: Show the Result
        st.write("### Analysis Result:")
        if prediction[0] == 1:
            st.error("🚨 **WARNING: This email is classified as SPAM.**")
            st.warning("It contains patterns typical of phishing, scams, or junk mail.")
        else:
            st.success("✅ **SAFE: This email is classified as HAM (Legitimate).**")
            st.info("No malicious patterns were detected in this message.")
    else:
        st.info("Please paste an email message into the box above to begin the scan.")

# 5. Technical Footer
st.divider()
st.caption("Algorithm: Multinomial Naive Bayes | Framework: Streamlit & Scikit-Learn")
