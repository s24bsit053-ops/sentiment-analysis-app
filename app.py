import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="Movie Review Sentiment", page_icon="🎬")
st.title("🎬 Movie Review Sentiment Analyzer")
st.write("Type a movie review and the model will predict whether it is **positive** or **negative**.")


@st.cache_resource
def load_model():
    return pipeline("text-classification", model="aqsaahsan/sentiment-distilbert")


clf = load_model()

examples = {
    "Choose an example...": "",
    "Positive example": "A brilliant, moving film with outstanding performances.",
    "Negative example": "A dull, predictable mess that wastes a great cast.",
}
choice = st.selectbox("Try an example", list(examples.keys()))
text = st.text_area("Enter a movie review:", value=examples[choice], height=150)

if st.button("Analyze"):
    if text.strip():
        result = clf(text)[0]
        label, score = result["label"], result["score"]
        if label == "POSITIVE":
            st.success(f"😊 POSITIVE  (confidence: {score:.2%})")
        else:
            st.error(f"😞 NEGATIVE  (confidence: {score:.2%})")
    else:
        st.warning("Please enter some text first.")

st.caption("Model: DistilBERT fine-tuned on Rotten Tomatoes reviews. Binary model: it has no neutral class.")
