# Movie Review Sentiment Analyzer

## Problem statement
Classify a movie review as **positive** or **negative** using a fine-tuned transformer model, and serve it through a web app.

## Dataset
- `cornell-movie-review-data/rotten_tomatoes` (Hugging Face)
- 10,662 review snippets: 8,530 train / 1,066 validation / 1,066 test
- 2 balanced classes (positive / negative)

## Model
- Pretrained model: [distilbert/distilbert-base-uncased](https://huggingface.co/distilbert/distilbert-base-uncased)
- Fine-tuned model: [aqsaahsan/sentiment-distilbert](https://huggingface.co/aqsaahsan/sentiment-distilbert)
- Training: 3 epochs, learning rate 2e-5, batch size 16, max length 128
- Test accuracy: 83.6%

## Project files
- `app.py` - Streamlit application
- `requirements.txt` - dependencies
- `notebooks/training.ipynb` - training notebook

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```

## Live app
https://movie-sentiment-aqsa.streamlit.app

## Limitations
The dataset has no neutral class, so neutral text is forced into positive or negative. The model may also carry biases inherited from DistilBERT's pretraining data.
