import streamlit as st
import joblib
import time 
from preprocess import clean_text,fast_tokenize
@st.cache_resource
def load_assets():
    model = joblib.load("Model2.joblib")
    vectorizer = joblib.load("Tfidf.joblib")
    return model, vectorizer

Model, tfidf = load_assets()
st.set_page_config(page_title="TruthScope – Spot the Lies",page_icon='⚖️',layout='centered',initial_sidebar_state='expanded')

st.title('FactCheckr: Smart Fake News Detector')
st.write("Identify fake news instantly with a trained NLP model based on real-world datasets.")

user_input=st.text_area(" Enter your Text Here ",height=200)


if st.button('Predict'):
    if user_input.strip() == "":
        st.warning("Please Enter Valid Text for prediction")
    else:
        cleantext=clean_text(user_input)
        tokenized_text=fast_tokenize(cleantext)
        tfidf_vec=tfidf.transform([tokenized_text])
        prediction=Model.predict(tfidf_vec)[0]
        prob = Model.predict_proba(tfidf_vec)[0][prediction]


        with st.spinner("Analyzing the news..."):
         time.sleep(1.5)


        if prediction == 0:
            st.warning(f"⚠️ This news appears to be **FAKE**. Be cautious before sharing it.Confidence: {prob*100:.2f}%")
            st.toast("🚨 Fake news detected!", icon="❗")
        else:
            st.success(f"✅ This news seems to be **REAL** and trustworthy.{prob*100:.2f}%")
            st.toast("🎉 News verified successfully!", icon="✅")

st.markdown("---")     
st.caption("Built with Python, Streamlit, and a trained NLP model for fake news detection.")


