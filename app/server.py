from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import json

# Define input model for POST requests
class PredictionRequest(BaseModel):
    texts: list[str]

# Load the pre-trained model (in .keras format)
# Update the file path to match the container's filesystem
model = load_model("/code/saved_model.keras")


# Load tokenizer configuration from a saved file
with open("/code/tokenizer_config.json", "r") as f:
    tokenizer_config = json.load(f)

# Reinitialize the tokenizer manually
tokenizer = Tokenizer()
tokenizer.word_index = tokenizer_config.get("word_index", {})
tokenizer.index_word = tokenizer_config.get("index_word", {})
tokenizer.num_words = tokenizer_config.get("num_words", None)

# Predefined maximum length (ensure this matches training)
max_len = 100

# Text cleaning function
def clean(text):
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = text.lower().strip()
    return text

# FastAPI app initialization
app = FastAPI()

@app.get("/")
def home():
    return {"message": "API is running. Use the `/predict` endpoint for predictions."}

@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        # Clean and preprocess the text
        new_texts_cleaned = [clean(text) for text in request.texts]
        new_sequences = tokenizer.texts_to_sequences(new_texts_cleaned)
        new_padded = pad_sequences(new_sequences, maxlen=max_len, padding="post")

        # Make predictions
        new_predictions = model.predict(new_padded)
        new_labels = (new_predictions > 0.5).astype("int32").tolist()

        return {"predictions": new_labels}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host = "0.0.0.0", port=8000)


