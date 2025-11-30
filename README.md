# 🤖 Fake News Detector AI — ML + Gemini API (Smart Verification)

This project is an **AI-powered Fake News Detection System** with a modern **Tkinter GUI**, combining:

* **Machine Learning (ML) Models**
* **TF-IDF Vectorization**
* **Google Gemini API (Enhanced Fact-Checking)**
* **Smart Verification Mode** — ML predicts first, Gemini verifies *only if fake*

It is optimized for **speed, accuracy, and real-time news analysis**.

---

## 🚀 Features

### 🔥 **1. Smart Verification Mode**

* ML model predicts first (fast)
* If prediction = **FAKE**, Gemini API performs deep verification
* Saves time + API usage

### 🧠 **2. Multiple ML Models Trained**

* Multinomial Naive Bayes
* Logistic Regression
* Random Forest
* Linear SVM
* Automatically selects **best-performing model**

### 🎨 **3. Modern Tkinter GUI**

* Dark theme
* Tabs: *Detect News*, *Model Info*, *Settings*
* Loading animation during Gemini verification

### 💬 **4. Detailed Analysis Output**

* Prediction (REAL or FAKE)
* Confidence score
* Real/Fake probability
* Gemini reasons & findings

### 📦 **5. Fully Optimized Preprocessing**

* Fast text cleaning
* NLTK tokenizer / lemmatizer
* Handles large CSV datasets

---

## 📸 Screenshots (Add after running)

> You can add screenshots like this:

```
![App Screenshot](screenshots/main_gui.png)
![Fake Verification](screenshots/fake_check.png)
```

---

## 📁 Project Structure

```
📦 FakeNewsDetector-AI
│
├── GUIFakeNewsDect2.py        # Main application (GUI + ML + Gemini)
├── final_news.csv             # Dataset (user provided)
├── gemini_config.json         # Stores API key (auto created)
├── README.md                  # Project documentation
└── screenshots/               # Add your images here
```

---

## 🛠️ Installation

### **1️⃣ Install Python 3.10+**

Download from: [https://www.python.org/downloads/](https://www.python.org/downloads/)

---

### **2️⃣ Install required libraries**

```bash
pip install pandas numpy nltk scikit-learn matplotlib seaborn tqdm google-generativeai
```

---

### **3️⃣ Download NLTK resources**

```python
import nltk
nltk.download('punkt')
nltk.download('wordnet')
```

---

## 🔑 Google Gemini API Setup

### **Get your API key:**

[https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)

### **Insert into the App:**

* Open the GUI → Settings tab
* Paste your key → Save

Or manually create:

```json
{
  "api_key": "YOUR_API_KEY_HERE"
}
```

---

## ▶️ Running the Application

### **Windows**

```bash
python GUIFakeNewsDect2.py
```

### **Mac/Linux**

```bash
python3 GUIFakeNewsDect2.py
```

The GUI will open automatically.

---

## 🧪 How It Works (Pipeline)

### **Step 1 — Text Preprocessing**

* Lowercase
* Remove symbols
* Remove extra spaces
* Combine title + content

### **Step 2 — ML Prediction**

* TF-IDF vectorizer creates 3000 features
* Best model predicts REAL/FAKE
* ML confidence is calculated

### **Step 3 — Smart Verification (Fake Only)**

Gemini returns:

* verdict
* confidence
* reasons
* key findings
* fact-check notes

### **Step 4 — GUI Output**

* Final verified result
* Real/Fake probability
* Explanation
* Confidence

---

## 📊 Model Performance (Sample)

```
Multinomial Naive Bayes   : 91.3%
Logistic Regression       : 92.1%
Random Forest             : 88.4%
Linear SVM                : 93.6%  ← BEST MODEL
```

---

## 🧱 Tech Stack

| Component       | Technology          |
| --------------- | ------------------- |
| GUI             | Tkinter             |
| ML Models       | Scikit-Learn        |
| Text Processing | NLTK                |
| Vectorization   | TF-IDF              |
| API             | Google Gemini Flash |
| Dataset         | CSV (news dataset)  |

---

## 🙌 Credits

Developed by **MUHAMMAD ISMAIL**
*Fake News Detector AI Project — 2025*

Cover Image / Banner can be added here.

---

## 📄 License

This project is **open-source** and free to use.

