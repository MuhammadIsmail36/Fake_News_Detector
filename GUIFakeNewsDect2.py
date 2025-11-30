import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import os
import time
from tqdm import tqdm
import warnings
import json
import google.generativeai as genai
from datetime import datetime, timedelta
import requests
import threading

warnings.filterwarnings('ignore')

# =============================================================================
# GEMINI API INTEGRATION
# =============================================================================

class GeminiNewsAnalyzer:
    def __init__(self, api_key=None):
        self.api_key = api_key or self.get_api_key()
        self.is_available = False
        self.setup_gemini()
        
    def get_api_key(self):
        """Get Gemini API key from environment or file"""
        # Try environment variable first
        api_key = os.getenv('GEMINI_API_KEY')
        if api_key:
            return api_key
            
        # Try config file
        try:
            with open('gemini_config.json', 'r') as f:
                config = json.load(f)
                return config.get('api_key')
        except:
            return None
    
    def setup_gemini(self):
        """Initialize Gemini API"""
        if not self.api_key:
            print("⚠️  Gemini API key not found. Gemini features disabled.")
            self.is_available = False
            return
            
        try:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel("models/gemini-flash-latest")
            # Test the API with a simple call
            response = self.model.generate_content("Test")
            self.is_available = True
            print("✅ Gemini API initialized successfully")
        except Exception as e:
            print(f"⚠️  Gemini API setup failed: {e}")
            self.is_available = False
    
    def analyze_with_gemini(self, title, content):
        """Analyze news using Gemini API with comprehensive fact-checking"""
        if not self.is_available:
            return None
            
        try:
            prompt = f"""
            FAKE NEWS DETECTION ANALYSIS - BE FACTUAL AND OBJECTIVE
            
            NEWS TITLE: {title}
            NEWS CONTENT: {content}
            
            Analyze this news article and determine if it's likely REAL or FAKE news.
            
            CONSIDER THESE FACTORS:
            1. **Factual Consistency**: Does it align with known facts and events?
            2. **Source Indicators**: Are credible sources cited? Any suspicious sources?
            3. **Sensationalism**: Is the language exaggerated, emotional, or clickbait-style?
            4. **Evidence**: Are claims supported by evidence, data, or verifiable information?
            5. **Logical Consistency**: Does the story make logical sense?
            6. **Writing Quality**: Is it professionally written with proper grammar?
            7. **Urgency Tactics**: Does it use excessive urgency or fear-mongering?
            
            RESPOND WITH THIS EXACT JSON FORMAT ONLY:
            {{
                "verdict": "real" or "fake",
                "confidence": 0-100,
                "reasons": ["reason1", "reason2", "reason3"],
                "key_findings": ["finding1", "finding2"],
                "fact_check_notes": ["note1", "note2"],
                "analysis_method": "gemini_api"
            }}
            
            Be objective and evidence-based in your analysis.
            """
            
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Extract JSON from response (in case there's additional text)
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return result
            else:
                # Try direct parsing
                return json.loads(response_text)
                
        except Exception as e:
            print(f"❌ Gemini API error: {e}")
            return None

# Download necessary NLTK resources for text processing
def download_nltk_resources():
    """Download required NLTK resources with error handling"""
    try:
        print("Loading NLTK resources...")
        nltk.download('punkt', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        print("NLTK resources downloaded.")
    except Exception as e:
        print(f"Error downloading NLTK resources: {e}")
        messagebox.showwarning("NLTK Warning", "Some NLTK resources failed to download. Text processing may be limited.")

# =============================================================================
# OPTIMIZED DATA PREPROCESSING AND MODEL TRAINING
# =============================================================================

def load_and_preprocess_data_optimized():
    """Optimized loading and preprocessing for large datasets"""
    print("🔄 Loading and preprocessing dataset...")
    start_time = time.time()
    
    try:
        # Try multiple possible file locations
        file_paths = [
            'final_news.csv',
            './final_news.csv',
            'C:/Users/DELL/Desktop/python/Ds_project/final_news.csv'
        ]
        
        df = None
        for file_path in file_paths:
            if os.path.exists(file_path):
                print(f"📁 Loading dataset from: {file_path}")
                df = pd.read_csv(file_path, on_bad_lines='skip', engine='python', encoding='latin1')
                break
        
        if df is None:
            # If file not found, show file dialog
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            file_path = filedialog.askopenfilename(
                title="Select news.csv file",
                filetypes=[("CSV files", "*.csv")]
            )
            if file_path:
                df = pd.read_csv(file_path, on_bad_lines='skip', engine='python', encoding='latin1')
            else:
                raise FileNotFoundError("No news.csv file found and no file selected")
        
        print(f"✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Data cleaning with column existence checks
        columns_to_drop = ['Unnamed: 0', 'date']
        existing_columns = [col for col in columns_to_drop if col in df.columns]
        if existing_columns:
            df = df.drop(columns=existing_columns)
        
        # Check for required columns
        required_columns = ['title', 'text', 'label']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise KeyError(f"Missing required columns: {missing_columns}")
        
        # Remove rows with missing values
        initial_size = len(df)
        df.dropna(subset=['title', 'text', 'subject'], inplace=True)
        print(f"🧹 Removed {initial_size - len(df)} rows with missing values")
        
        # Check label values
        print("🏷️ Unique labels in dataset:", df['label'].unique())
        
        # OPTIMIZED TEXT PREPROCESSING
        print("🔧 Optimized text preprocessing...")
        
        # Convert to string and lowercase (vectorized operations)
        df['title'] = df['title'].astype(str).str.lower()
        df['text'] = df['text'].astype(str).str.lower()
        
        # Simplified cleaning function
        def clean_text_fast(text):
            text = re.sub(r'[^a-zA-Z\s]', ' ', text)
            text = re.sub(r'\s+', ' ', text)
            return text.strip()
        
        # Apply cleaning with progress indication
        df['title'] = df['title'].apply(clean_text_fast)
        df['text'] = df['text'].apply(clean_text_fast)
        
        # Create combined text directly without full tokenization
        df['combined_text'] = df['title'] + ' ' + df['text']
        
        # Remove very short texts
        df = df[df['combined_text'].str.len() > 50]
        
        print(f"✅ Preprocessing completed. Final dataset: {df.shape[0]} rows")
        print(f"⏱️ Preprocessing time: {time.time() - start_time:.2f} seconds")
        
        return df
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        raise

def train_models_optimized(df):
    """Optimized model training for speed"""
    print("🚀 Starting optimized model training...")
    training_start = time.time()
    
    # OPTIMIZED FEATURE ENGINEERING
    print("🔧 Creating optimized TF-IDF features...")
    vectorizer = TfidfVectorizer(
        max_features=3000,           # Reduced from 5000 for speed
        stop_words='english',
        ngram_range=(1, 1),          # Only unigrams (faster)
        min_df=5,                    # Higher min frequency
        max_df=0.9,                  # Slightly higher max frequency
        sublinear_tf=True,           # Use sublinear TF scaling
        use_idf=True,
        smooth_idf=True
    )
    
    # Fit and transform
    X = vectorizer.fit_transform(df['combined_text'])
    y = df['label']
    
    print(f"📊 Feature matrix shape: {X.shape}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"📈 Data split: {X_train.shape[0]} train, {X_test.shape[0]} test samples")
    
    # OPTIMIZED MODEL SELECTION - Focus on faster models
    models = {
        'Multinomial Naive Bayes': MultinomialNB(alpha=0.1),
        'Logistic Regression': LogisticRegression(
            random_state=42, 
            max_iter=500,           # Reduced iterations
            C=1.0,
            solver='liblinear'      # Faster solver
        ),
        'Random Forest (Fast)': RandomForestClassifier(
            n_estimators=50,        # Reduced from 100
            max_depth=20,           # Limit depth
            random_state=42,
            n_jobs=-1               # Use all cores
        ),
        'Linear SVM': SVC(
            kernel='linear',        # Linear kernel is much faster
            probability=True,
            random_state=42,
            C=1.0
        )
    }
    
    # Train models and find best one
    model_performances = {}
    trained_models = {}
    training_times = {}
    
    for model_name, model in models.items():
        print(f"⏳ Training {model_name}...")
        model_start = time.time()
        
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            model_time = time.time() - model_start
            model_performances[model_name] = accuracy
            trained_models[model_name] = model
            training_times[model_name] = model_time
            
            print(f"✅ {model_name}: {accuracy:.4f} accuracy in {model_time:.1f}s")
            
        except Exception as e:
            print(f"❌ Error training {model_name}: {e}")
            continue
    
    # Find best model
    if model_performances:
        best_model_name = max(model_performances, key=model_performances.get)
        best_model = trained_models[best_model_name]
        
        total_training_time = time.time() - training_start
        
        print("\n🎯 TRAINING SUMMARY:")
        print("=" * 50)
        for model_name, accuracy in model_performances.items():
            print(f"• {model_name}: {accuracy:.4f} ({training_times[model_name]:.1f}s)")
        print(f"⭐ Best Model: {best_model_name} ({model_performances[best_model_name]:.4f})")
        print(f"⏱️ Total training time: {total_training_time:.1f} seconds ({total_training_time/60:.1f} minutes)")
        print("=" * 50)
        
        return vectorizer, best_model, best_model_name, model_performances, X_test, y_test
    else:
        raise Exception("No models were successfully trained")

# =============================================================================
# ENHANCED GUI APPLICATION WITH SMART GEMINI INTEGRATION
# =============================================================================

class EnhancedFakeNewsDetectorGUI:
    def __init__(self, root, vectorizer, model, model_name, model_performances, X_test, y_test):
        self.root = root
        self.vectorizer = vectorizer
        self.model = model
        self.model_name = model_name
        self.model_performances = model_performances
        self.X_test = X_test
        self.y_test = y_test
        
        # Initialize Gemini analyzer
        self.gemini_analyzer = GeminiNewsAnalyzer()
        
        # Loading animation variables
        self.loading_active = False
        self.loading_dots = 0
        
        self.setup_gui()
        
    def setup_gui(self):
        """Setup the GUI interface"""
        self.root.title("🤖 Fake News Detector AI - Smart Verification")
        self.root.geometry("900x800")
        self.root.configure(bg='#2C3E50')
        
        # Create style for modern look
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TFrame', background='#2C3E50')
        style.configure('TLabel', background='#2C3E50', foreground='white', font=('Arial', 10))
        style.configure('TButton', font=('Arial', 10), padding=10)
        style.configure('Header.TLabel', font=('Arial', 16, 'bold'))
        style.configure('Result.TLabel', font=('Arial', 12, 'bold'))
        style.configure('TCheckbutton', background='#2C3E50', foreground='white')
        
        # Main container
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Header
        header_label = ttk.Label(main_frame, text="🔍 FAKE NEWS DETECTOR AI - SMART VERIFICATION", style='Header.TLabel')
        header_label.pack(pady=10)
        
        # Model info
        info_label = ttk.Label(main_frame, text=f"Model: {self.model_name} | Accuracy: {self.model_performances[self.model_name]:.2%}")
        info_label.pack(pady=5)
        
        # Gemini status
        gemini_status = "✅ Available" if self.gemini_analyzer.is_available else "❌ Not Available"
        gemini_label = ttk.Label(main_frame, text=f"Gemini Verification: {gemini_status}")
        gemini_label.pack(pady=2)
        
        # Analysis mode info
        mode_info = ttk.Label(main_frame, text="🤖 Smart Mode: Instant results for real news, enhanced verification for fake news", 
                             font=('Arial', 9, 'italic'))
        mode_info.pack(pady=2)
        
        # Create notebook for tabs
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Tab 1: News Detection
        self.create_detection_tab(notebook)
        
        # Tab 2: Model Info
        self.create_info_tab(notebook)
        
        # Tab 3: Settings
        self.create_settings_tab(notebook)
        
    def create_detection_tab(self, notebook):
        """Create the news detection tab"""
        detection_frame = ttk.Frame(notebook)
        notebook.add(detection_frame, text="📰 Detect Fake News")
        
        # Title input
        ttk.Label(detection_frame, text="News Title:").pack(anchor='w', pady=(10, 5))
        self.title_entry = scrolledtext.ScrolledText(detection_frame, height=3, width=80, font=('Arial', 10))
        self.title_entry.pack(fill=tk.X, pady=5)
        
        # Content input
        ttk.Label(detection_frame, text="News Content:").pack(anchor='w', pady=(10, 5))
        self.content_entry = scrolledtext.ScrolledText(detection_frame, height=10, width=80, font=('Arial', 10))
        self.content_entry.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Buttons frame
        button_frame = ttk.Frame(detection_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        self.analyze_button = ttk.Button(button_frame, text="🔍 Analyze News", 
                                       command=self.analyze_news)
        self.analyze_button.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(button_frame, text="🧹 Clear", 
                  command=self.clear_inputs).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="💡 Example", 
                  command=self.load_example).pack(side=tk.LEFT, padx=5)
        
        # Results frame
        results_frame = ttk.LabelFrame(detection_frame, text="Analysis Results", padding=10)
        results_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Method indicator
        self.method_label = ttk.Label(results_frame, text="Method: Not analyzed yet", font=('Arial', 9))
        self.method_label.pack(anchor='w', pady=5)
        
        self.result_label = ttk.Label(results_frame, text="Enter news above and click 'Analyze News'", 
                                     style='Result.TLabel')
        self.result_label.pack(pady=5)
        
        self.confidence_label = ttk.Label(results_frame, text="")
        self.confidence_label.pack(pady=2)
        
        self.details_label = ttk.Label(results_frame, text="", wraplength=800, justify=tk.LEFT)
        self.details_label.pack(pady=5, fill=tk.X)
        
        # Loading indicator (initially hidden)
        self.loading_label = ttk.Label(results_frame, text="", font=('Arial', 10, 'italic'))
        self.loading_label.pack(pady=5)
        
    def create_info_tab(self, notebook):
        """Create the model information tab"""
        info_frame = ttk.Frame(notebook)
        notebook.add(info_frame, text="📊 Model Information")
        
        # Model performance
        perf_frame = ttk.LabelFrame(info_frame, text="Model Performance", padding=10)
        perf_frame.pack(fill=tk.X, pady=10, padx=10)
        
        perf_text = "⚡ Optimized Model Accuracy Scores:\n\n"
        for model_name, accuracy in self.model_performances.items():
            perf_text += f"• {model_name}: {accuracy:.2%}\n"
        
        perf_label = ttk.Label(perf_frame, text=perf_text, justify=tk.LEFT)
        perf_label.pack(anchor='w')
        
        # Smart mode info
        smart_frame = ttk.LabelFrame(info_frame, text="Smart Verification Mode", padding=10)
        smart_frame.pack(fill=tk.X, pady=10, padx=10)
        
        smart_text = """
🤖 Smart Analysis Workflow:

1. 🚀 INSTANT ML ANALYSIS
   • Uses trained model for initial prediction
   • Instant results for REAL news

2. 🔍 ENHANCED VERIFICATION (Only for FAKE predictions)
   • If ML predicts FAKE → Enhanced verification starts
   • Shows "Verifying..." while analyzing
   • Provides final verified result with detailed analysis

3. 💡 BENEFITS
   • Faster results for real news
   • Higher accuracy for fake news
   • Cost-efficient API usage
   • Best of both approaches
        """
        smart_label = ttk.Label(smart_frame, text=smart_text, justify=tk.LEFT)
        smart_label.pack(anchor='w')
        
        # Test model button
        ttk.Button(info_frame, text="🧪 Test Model Performance", 
                  command=self.show_model_performance).pack(pady=10)
        
    def create_settings_tab(self, notebook):
        """Create the settings tab"""
        settings_frame = ttk.Frame(notebook)
        notebook.add(settings_frame, text="⚙️ Settings")
        
        # Gemini API settings
        api_frame = ttk.LabelFrame(settings_frame, text="Gemini API Configuration", padding=10)
        api_frame.pack(fill=tk.X, pady=10, padx=10)
        
        ttk.Label(api_frame, text="API Key:").pack(anchor='w')
        self.api_key_entry = ttk.Entry(api_frame, width=60, show="*")
        self.api_key_entry.pack(fill=tk.X, pady=5)
        
        ttk.Button(api_frame, text="Save API Key", 
                  command=self.save_api_key).pack(pady=5)
        
        ttk.Label(api_frame, text="How to get API key:", font=('Arial', 9, 'italic')).pack(anchor='w', pady=(10,0))
        ttk.Label(api_frame, text="1. Visit: https://aistudio.google.com/app/apikey", 
                 font=('Arial', 8)).pack(anchor='w')
        ttk.Label(api_frame, text="2. Create new API key", 
                 font=('Arial', 8)).pack(anchor='w')
        ttk.Label(api_frame, text="3. Paste above and save", 
                 font=('Arial', 8)).pack(anchor='w')
        
    def save_api_key(self):
        """Save Gemini API key"""
        api_key = self.api_key_entry.get().strip()
        if api_key:
            try:
                # Save to config file
                config = {'api_key': api_key}
                with open('gemini_config.json', 'w') as f:
                    json.dump(config, f)
                
                # Reinitialize Gemini
                self.gemini_analyzer = GeminiNewsAnalyzer(api_key)
                messagebox.showinfo("Success", "API key saved and Gemini initialized!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save API key: {e}")
        else:
            messagebox.showwarning("Warning", "Please enter an API key")
    
    def start_loading_animation(self):
        """Start loading animation"""
        self.loading_active = True
        self.loading_dots = 0
        self.update_loading_animation()
    
    def stop_loading_animation(self):
        """Stop loading animation"""
        self.loading_active = False
        self.loading_label.config(text="")
    
    def update_loading_animation(self):
        """Update loading animation dots"""
        if self.loading_active:
            dots = "." * (self.loading_dots % 4)
            self.loading_label.config(text=f"Verifying analysis{dots}", foreground="orange")
            self.loading_dots += 1
            self.root.after(500, self.update_loading_animation)
    
    def clean_text_fast(self, text):
        """Fast text cleaning for real-time analysis"""
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip().lower()
    
    def analyze_news(self):
        """Smart analysis: ML first, Gemini verification only if ML predicts FAKE"""
        title = self.title_entry.get("1.0", tk.END).strip()
        content = self.content_entry.get("1.0", tk.END).strip()
        
        if not title or not content:
            messagebox.showwarning("Input Error", "Please enter both title and content!")
            return
        
        # Disable analyze button during processing
        self.analyze_button.config(state='disabled')
        
        try:
            # Step 1: Always run ML analysis first
            ml_prediction, ml_confidence, real_prob, fake_prob = self.ml_analysis_only(title, content)
            
            # Step 2: Check if ML predicts FAKE and Gemini is available
            if ml_prediction.lower() == 'fake' and self.gemini_analyzer.is_available:
                # Show loading state immediately (no ML result shown)
                self.show_loading_state()
                
                # Run Gemini verification in a separate thread
                threading.Thread(target=self.verify_with_gemini, 
                               args=(title, content, ml_prediction, ml_confidence), 
                               daemon=True).start()
            else:
                # If ML predicts REAL or Gemini not available, show ML result directly
                if ml_prediction.lower() == 'fake' and not self.gemini_analyzer.is_available:
                    method_used = "ML Model"
                else:
                    method_used = "ML Model"
                self.display_final_results(ml_prediction, ml_confidence, real_prob, fake_prob, method_used, [])
                self.analyze_button.config(state='normal')
                
        except Exception as e:
            messagebox.showerror("Analysis Error", f"Error during analysis: {str(e)}")
            self.analyze_button.config(state='normal')
            self.stop_loading_animation()
    
    def show_loading_state(self):
        """Show loading state for Gemini verification"""
        self.method_label.config(text="Method: Enhanced Verification")
        self.result_label.config(text="🔍 Analyzing news content...", foreground="orange")
        self.confidence_label.config(text="")
        self.details_label.config(text="Performing enhanced fact-checking and verification")
        self.start_loading_animation()
    
    def verify_with_gemini(self, title, content, ml_prediction, ml_confidence):
        """Verify ML fake prediction with Gemini (runs in separate thread)"""
        try:
            # Verify with Gemini
            gemini_result = self.gemini_analyzer.analyze_with_gemini(title, content)
            
            # Schedule UI update in main thread
            self.root.after(0, self.finish_gemini_verification, 
                          gemini_result, ml_prediction, ml_confidence)
        except Exception as e:
            # Schedule error handling in main thread
            self.root.after(0, self.handle_gemini_error, e, ml_prediction, ml_confidence)
    
    def finish_gemini_verification(self, gemini_result, ml_prediction, ml_confidence):
        """Finish Gemini verification (called in main thread)"""
        self.stop_loading_animation()
        self.analyze_button.config(state='normal')
        
        if gemini_result:
            gemini_verdict = gemini_result.get('verdict', 'unknown')
            gemini_confidence = gemini_result.get('confidence', 50)
            reasons = gemini_result.get('reasons', [])
            findings = gemini_result.get('key_findings', [])
            notes = gemini_result.get('fact_check_notes', [])
            
            # Use Gemini's verdict and confidence
            real_prob = gemini_confidence/100 if gemini_verdict == 'real' else (100-gemini_confidence)/100
            fake_prob = 1 - real_prob
            
            method_used = "Enhanced Verification"
            self.display_final_results(gemini_verdict, gemini_confidence/100, real_prob, fake_prob, method_used, reasons)
        else:
            # If Gemini fails, fall back to ML result
            real_prob = ml_confidence if ml_prediction.lower() == 'real' else 1-ml_confidence
            fake_prob = 1 - real_prob
            self.display_final_results(ml_prediction, ml_confidence, real_prob, fake_prob, "ML Model", [])
    
    def handle_gemini_error(self, error, ml_prediction, ml_confidence):
        """Handle Gemini verification errors (called in main thread)"""
        self.stop_loading_animation()
        self.analyze_button.config(state='normal')
        
        # Fall back to ML result
        real_prob = ml_confidence if ml_prediction.lower() == 'real' else 1-ml_confidence
        fake_prob = 1 - real_prob
        self.display_final_results(ml_prediction, ml_confidence, real_prob, fake_prob, "ML Model", [])
        
        # Show error message
        messagebox.showerror("Verification Error", f"Enhanced verification failed. Showing ML result instead.\nError: {str(error)}")
    
    def ml_analysis_only(self, title, content):
        """Traditional ML analysis only - returns prediction data"""
        combined = title + ' ' + content
        processed_text = self.clean_text_fast(combined)
        
        # Transform and predict
        vectorized_text = self.vectorizer.transform([processed_text])
        prediction = self.model.predict(vectorized_text)[0]
        
        # Get probabilities
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(vectorized_text)[0]
            confidence = max(probabilities)
            real_prob = probabilities[0] if self.model.classes_[0] == 'real' else probabilities[1]
            fake_prob = 1 - real_prob
        else:
            confidence = 0.85
            real_prob = 0.5
            fake_prob = 0.5
        
        return prediction, confidence, real_prob, fake_prob
    
    def display_final_results(self, prediction, confidence, real_prob, fake_prob, method_used, reasons):
        """Display final analysis results (only one prediction shown to user)"""
        self.method_label.config(text=f"Method: {method_used}")
        
        if prediction.lower() == 'real':
            color = "green"
            icon = "✅"
            message = "This appears to be REAL news"
            details = f"Real news probability: {real_prob:.2%}\nFake news probability: {fake_prob:.2%}"
        else:
            color = "red"
            icon = "🚩"
            message = "This appears to be FAKE news"
            details = f"Fake news probability: {fake_prob:.2%}\nReal news probability: {real_prob:.2%}"
        
        # Add reasons if available (from Gemini)
        if reasons:
            details += "\n\nKey Findings:\n• " + "\n• ".join(reasons[:3])
        
        self.result_label.config(text=f"{icon} {message}", foreground=color)
        self.confidence_label.config(text=f"Confidence: {confidence:.2%}")
        self.details_label.config(text=details)
    
    def clear_inputs(self):
        """Clear all input fields"""
        self.title_entry.delete("1.0", tk.END)
        self.content_entry.delete("1.0", tk.END)
        self.result_label.config(text="Enter news above and click 'Analyze News'", foreground="black")
        self.confidence_label.config(text="")
        self.details_label.config(text="")
        self.method_label.config(text="Method: Not analyzed yet")
        self.loading_label.config(text="")
        self.analyze_button.config(state='normal')
        self.stop_loading_animation()
    
    def load_example(self):
        """Load example news articles"""
        examples = [
            {
                "title": "NASA Announces Discovery of Water on Mars in 2024 Mission",
                "content": "In a groundbreaking discovery, NASA's Perseverance rover has confirmed the presence of liquid water on Mars during its 2024 mission. Scientists announced today that new spectroscopic analysis reveals seasonal water flows in the Martian valleys. This recent finding could have major implications for future colonization efforts and the search for extraterrestrial life. The data was peer-reviewed and published in Nature Journal last week.",
                "type": "real"
            },
            {
                "title": "BREAKING: New AI Technology Can Read Minds - Governments Hiding Truth!",
                "content": "SHOCKING discovery! Secret documents reveal that tech giants have developed AI that can literally read your thoughts! They've been testing this on unsuspecting citizens since 2023. The government is covering it up to maintain control! This exclusive information can't be found anywhere else! Share this before they delete it! Your privacy is at risk - ACT NOW!",
                "type": "fake"
            }
        ]
        
        # Clear current inputs
        self.clear_inputs()
        
        # Load fake news example (to demonstrate Gemini verification)
        example = examples[1]  # Fake news example
        self.title_entry.insert("1.0", example["title"])
        self.content_entry.insert("1.0", example["content"])
        
        messagebox.showinfo("Example Loaded", 
                          "A fake news example has been loaded.\n\n" +
                          "In Smart Mode:\n" +
                          "• Enhanced verification will run automatically\n" +
                          "• You'll see 'Verifying...' during analysis\n" +
                          "• Final result with detailed analysis will be shown\n" +
                          "Click 'Analyze News' to see the enhanced verification in action!")
    
    def show_model_performance(self):
        """Show detailed model performance"""
        y_pred = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        
        # Handle different label types
        classes = np.unique(self.y_test)
        if len(classes) == 2:
            pos_label = classes[1] if 'real' not in classes else 'real'
            precision = precision_score(self.y_test, y_pred, pos_label=pos_label, zero_division=0)
            recall = recall_score(self.y_test, y_pred, pos_label=pos_label, zero_division=0)
            f1 = f1_score(self.y_test, y_pred, pos_label=pos_label, zero_division=0)
        else:
            precision = precision_score(self.y_test, y_pred, average='macro', zero_division=0)
            recall = recall_score(self.y_test, y_pred, average='macro', zero_division=0)
            f1 = f1_score(self.y_test, y_pred, average='macro', zero_division=0)
        
        performance_text = f"""
📊 Detailed Model Performance:

Accuracy: {accuracy:.2%}
Precision: {precision:.2%}
Recall: {recall:.2%}
F1-Score: {f1:.2%}

Classification Report:
{classification_report(self.y_test, y_pred, zero_division=0)}
        """
        
        # Create performance window
        perf_window = tk.Toplevel(self.root)
        perf_window.title("Model Performance Details")
        perf_window.geometry("600x500")
        
        text_area = scrolledtext.ScrolledText(perf_window, wrap=tk.WORD, width=70, height=25)
        text_area.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        text_area.insert(tk.INSERT, performance_text)
        text_area.config(state=tk.DISABLED)

# =============================================================================
# MAIN EXECUTION - SMART GEMINI INTEGRATION
# =============================================================================

def main_enhanced():
    """Enhanced main function with smart Gemini integration"""
    print("🚀 Starting SMART Fake News Detector Application...")
    print("⚡ Smart Mode: ML first → Gemini verifies only FAKE predictions")
    print("🎯 Benefits: Faster real news, more accurate fake news detection")
    print("💰 Cost-efficient: Gemini only used when needed")
    print("=" * 60)
    
    try:
        # Download NLTK resources
        download_nltk_resources()
        
        # Load and preprocess data with optimized method
        df = load_and_preprocess_data_optimized()
        
        # Train models with optimized method
        vectorizer, best_model, best_model_name, model_performances, X_test, y_test = train_models_optimized(df)
        
        # Create enhanced GUI
        root = tk.Tk()
        app = EnhancedFakeNewsDetectorGUI(root, vectorizer, best_model, best_model_name, model_performances, X_test, y_test)
        
        print("\n✅ Smart application initialized successfully!")
        print("🔍 Smart Features:")
        print("   • ML analysis first (fast)")
        print("   • Gemini verification only for FAKE predictions")
        print("   • Cost-efficient API usage")
        print("   • Higher accuracy for fake news")
        print("🎮 GUI is ready to use!")
        
        # Start the GUI
        root.mainloop()
        
    except Exception as e:
        print(f"❌ Error during application startup: {e}")
        messagebox.showerror("Startup Error", f"Failed to start application: {str(e)}")

if __name__ == "__main__":
    main_enhanced()