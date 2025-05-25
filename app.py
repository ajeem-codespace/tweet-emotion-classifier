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
print("Attempting to ensure NLTK packages are available...") # For logs

for package_id in nltk_packages:
    try:
        # Determine the correct path pattern for nltk.data.find()
        if package_id == 'punkt':
            path_to_check = f'tokenizers/{package_id}'
        elif package_id in ['stopwords', 'wordnet']: # Corrected 'in'
            path_to_check = f'corpora/{package_id}'
        else: # Should not happen with the current nltk_packages list
            path_to_check = f'misc/{package_id}' # Fallback, though unlikely needed
        
        nltk.data.find(path_to_check)
        print(f"NLTK: Resource '{package_id}' found.")
    except LookupError: # This is the correct exception to catch for nltk.data.find()
        print(f"NLTK: Resource '{package_id}' not found. Attempting download...")
        # For 'punkt', download with verbose output to see details in Streamlit logs
        # For other packages, quiet=True is generally fine.
        is_punkt_and_verbose = (package_id == 'punkt')
        try:
            nltk.download(package_id, quiet=(not is_punkt_and_verbose))
            # Verify after download attempt
            nltk.data.find(path_to_check)
            print(f"NLTK: Resource '{package_id}' successfully downloaded and available.")
        except Exception as e_download: # Catch any error during download or re-check
            error_msg = f"NLTK: CRITICAL ERROR for '{package_id}'. Download or verification failed: {str(e_download)}"
            print(error_msg)
            st.error(error_msg)
            if package_id == 'punkt': # If 'punkt' fails, the app is likely unusable
                st.warning("The 'punkt' tokenizer (essential for text processing) failed to load. The app might not function correctly.")
                st.stop() # Stop the app if 'punkt' is critical and fails
    except Exception as e_outer: # Catch any other unexpected error
        error_msg_outer = f"NLTK: Unexpected error checking for '{package_id}': {str(e_outer)}"
        print(error_msg_outer)
        st.error(error_msg_outer)




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


