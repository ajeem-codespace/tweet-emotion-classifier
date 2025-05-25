import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import os
import pandas as pd


nltk_packages = ['stopwords', 'punkt', 'wordnet']
for package_id in nltk_packages:
    try:
        if package_id == 'punkt': 
            nltk.data.find(f'tokenizers/{package_id}')
        elif package_id in ['stopwords', 'wordnet']:
            nltk.data.find(f'corpora/{package_id}')
    except nltk.downloader.DownloadError: 
        nltk.download(package_id, quiet=False if package_id == 'punkt' else True)
    except LookupError: 
        nltk.download(package_id, quiet=False if package_id == 'punkt' else True)



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



#MODEL_FILENAME = 'logistic_regression_model.joblib'
# MODEL_FILENAME = 'svm_model_rbf.joblib'
MODEL_FILENAME = 'svm_model_linear.joblib'
# --- END CHOOSE YOUR MODEL ---
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

#  App Interface 
st.set_page_config(page_title="EmotionSense", layout="wide", initial_sidebar_state="expanded")

st.title("Tweet Emotion Classifier")
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

#Sidebar 
st.sidebar.header("About This App")
model_type_display = MODEL_FILENAME.replace('_model','').replace('.joblib','').replace('_',' ').capitalize()
st.sidebar.info(f"""
    This app uses a **{model_type_display}** model to classify text emotion.
    Trained on a subset of the 'emotions.csv' dataset.
    Labels: Sadness, Joy, Love, Anger, Fear, Surprise.
""")


