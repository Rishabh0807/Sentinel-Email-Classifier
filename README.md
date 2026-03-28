# 🛡️ Sentinel: AI-Powered Email Spam Shield
### **BTech First Year Project | VIT 2026**

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)

---

## 📖 Project Overview
Sentinel is an end-to-end Machine Learning application designed to combat digital communication clutter. Using **Natural Language Processing (NLP)** and supervised learning, the system accurately classifies incoming messages into **Spam** or **Ham** (Legitimate). 

This project was developed to demonstrate the practical application of the **Multinomial Naive Bayes** algorithm in solving real-world security challenges.

---

## 🚀 dataset
Download dataset from here:
👉 **https://www.kaggle.com/datasets/jackksoncsie/spam-email-dataset**

---

## 🛠️ Tech Stack & Libraries
The project utilizes a modern AI/ML pipeline:

* **Language:** `Python 3.10+`
* **Machine Learning:** `Scikit-Learn` (Naive Bayes & CountVectorizer)
* **Data Handling:** `Pandas` & `NumPy`
* **Model Deployment:** `Streamlit Cloud`
* **Development Environment:** `Spyder IDE`
* **Model Serialization:** `Joblib`

---

## 📂 Project Structure
The repository is organized to follow standard deployment protocols:

```text
├── app.py                # Main Streamlit web application code
├── spam_model.pkl        # The pre-trained AI "Brain"
├── vectorizer.pkl        # The NLP vocabulary processor
├── requirements.txt      # List of libraries for cloud installation
├── README.md             # Project documentation (You are here!)
└── .gitignore            # Files to be ignored by GitHub
