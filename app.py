import streamlit as st


st.set_page_config(page_title="EmotionSense", layout="wide", initial_sidebar_state="expanded")


import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import os
import pandas as pd


def setup_nltk_resources():
    nltk_packages_to_configure = {
        'stopwords': 'corpora/stopwords',
        'punkt': 'tokenizers/punkt',    
        'wordnet': 'corpora/wordnet'
    }
    all_successful = True
    startup_logs = ["--- Initializing NLTK Resource Check ---"]

    local_nltk_data_dir = os.path.join(os.getcwd(), "nltk_data_app_local")
    
    if not os.path.exists(local_nltk_data_dir):
        try:
            os.makedirs(local_nltk_data_dir)
            startup_logs.append(f"NLTK: Created local NLTK data directory: {local_nltk_data_dir}")
        except Exception as e:
            startup_logs.append(f"NLTK: ERROR creating local NLTK data directory '{local_nltk_data_dir}': {e}")

    if os.path.exists(local_nltk_data_dir) and local_nltk_data_dir not in nltk.data.path:
        nltk.data.path.insert(0, local_nltk_data_dir) 
        startup_logs.append(f"NLTK: Added '{local_nltk_data_dir}' to NLTK data path.")
    
    for package_id, path_to_check in nltk_packages_to_configure.items():
        try:
            nltk.data.find(path_to_check)
            startup_logs.append(f"NLTK: Resource '{package_id}' found.")
        except LookupError:
            startup_logs.append(f"NLTK: Resource '{package_id}' not found. Attempting download...")
            try:
                
                be_quiet_on_download = False if package_id in ['punkt', 'wordnet'] else True
                nltk.download(package_id, download_dir=local_nltk_data_dir, quiet=be_quiet_on_download)
                
                
                nltk.data.find(path_to_check) 
                startup_logs.append(f"NLTK: Resource '{package_id}' successfully downloaded/verified.")
            except Exception as e_download:
                startup_logs.append(f"NLTK: CRITICAL ERROR for '{package_id}'. Download or verification failed: {str(e_download)}")
                all_successful = False
                if package_id in ['punkt', 'wordnet']: 
                  
                    for msg in startup_logs: print(msg)
                    return False 
        except Exception as e_initial_find:
            startup_logs.append(f"NLTK: Unexpected error checking for '{package_id}': {str(e_initial_find)}")
            all_successful = False
            if package_id in ['punkt', 'wordnet']:
                 for msg in startup_logs: print(msg)
                 return False

    startup_logs.append("--- NLTK Resource Check Complete ---")
    for msg in startup_logs: 
        print(msg)
    return all_successful


nltk_ready = setup_nltk_resources()

if not nltk_ready:
    
    st.error("Failed to initialize critical NLTK resources (punkt or wordnet). The app cannot continue. Please check the deployment logs for detailed NLTK download messages.")
    st.stop() 


lemmatizer = WordNetLemmatizer()
stop_words_english = set(stopwords.words('english'))

def preprocess_text(text_input):
    text_input = str(text_input).lower()
    text_input = re.sub(r'http\S+|www\S+|https\S+', '', text_input, flags=re.MULTILINE)
    text_input = re.sub(r'\@\w+', '', text_input)
    text_input = re.sub(r'#', '', text_input)
    text_input = re.sub(r'[^\w\s]', '', text_input)
    tokens = word_tokenize(text_input)
    processed_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words_english]
    return " ".join(processed_tokens)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
VECTORIZER_PATH = os.path.join(MODELS_DIR, 'tfidf_vectorizer.joblib')
MODEL_FILENAME = 'logistic_regression_model.joblib' # Defaulting to Logistic Regression
MODEL_PATH = os.path.join(MODELS_DIR, MODEL_FILENAME)

try:
    tfidf_vectorizer = joblib.load(VECTORIZER_PATH)
    model = joblib.load(MODEL_PATH)
except Exception as e:
    st.error(f"Critical Error: Could not load model or vectorizer. Ensure files are in 'models/' directory. Details: {e}")
    st.stop()


emotion_mapping = {
    0: 'sadness', 1: 'joy', 2: 'love',
    3: 'anger', 4: 'fear', 5: 'surprise'
}


st.title("🎭 EmotionSense: Tweet Emotion Classifier")
st.markdown("Enter text below to predict the expressed emotion.")

user_input = st.text_area("Enter your text here:", height=100, placeholder="E.g., 'What a wonderful surprise!'")

if st.button("Classify Emotion", type="primary", use_container_width=True):
    if user_input and user_input.strip():
        cleaned_input = preprocess_text(user_input)
        input_vectorized = tfidf_vectorizer.transform([cleaned_input])
        prediction_label = model.predict(input_vectorized)[0]
        predicted_emotion_name = emotion_mapping.get(prediction_label, "Unknown")
        
        st.subheader("Predicted Emotion:")
        st.markdown(f"<h2 style='text-align: center; color: steelblue; border: 1px solid #e1e1e1; border-radius: 5px; padding: 10px;'>{predicted_emotion_name.capitalize()}</h2>", unsafe_allow_html=True)

        if hasattr(model, "predict_proba"):
            prediction_proba = model.predict_proba(input_vectorized)[0]
            st.markdown("---")
            st.write("Prediction Confidence:")
            prob_df = pd.DataFrame({
                "Emotion": [emotion_mapping.get(i, f"Label {i}") for i in model.classes_],
                "Probability": prediction_proba
            }).sort_values(by="Probability", ascending=False).reset_index(drop=True)
            st.dataframe(prob_df, use_container_width=True, height=(len(model.classes_)+1)*35+3)
    else:
        st.warning("👈 Please enter some text to classify.")


st.sidebar.header("About This App")
model_type_display = MODEL_FILENAME.replace('_model','').replace('.joblib','').replace('_',' ').capitalize()
st.sidebar.info(f"""
    This app uses a **{model_type_display}** model to classify text emotion.
    Trained on a subset of the 'emotions.csv' dataset.
    Labels: Sadness, Joy, Love, Anger, Fear, Surprise.
""")
