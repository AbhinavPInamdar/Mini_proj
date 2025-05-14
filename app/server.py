from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from fastapi.middleware.cors import CORSMiddleware
import re
import json

class PredictionRequest(BaseModel):
    texts: list[str]

# Load model
model = load_model("/code/saved_model.keras")

# Load tokenizer config
with open("/code/tokenizer_config.json", "r") as f:
    tokenizer_config = json.load(f)

tokenizer = Tokenizer()
tokenizer.word_index = tokenizer_config.get("word_index", {})
tokenizer.index_word = tokenizer_config.get("index_word", {})
tokenizer.num_words = tokenizer_config.get("num_words", None)

max_len = 100

def clean(text):
    lines = text.splitlines()
    filtered = []

    for line in lines:
        line = line.strip()
        if (
            not line or
            "verified purchase" in line.lower() or
            "out of 5 stars" in line.lower() or
            line.lower().startswith("reviewed in") or
            line.lower().startswith("band colour") or
            line.lower() in {"helpful", "report"} or
            re.match(r"^\d+(\.\d+)? out of 5 stars", line, re.I) or
            re.match(r"^\w+\s*$", line)  # likely a name
        ):
            continue
        filtered.append(line)

    clean_text = " ".join(filtered)
    clean_text = re.sub(r"[^a-zA-Z0-9\s]", " ", clean_text)
    clean_text = re.sub(r"\s+", " ", clean_text)
    return clean_text.lower().strip()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/")
def home():
    return {"message": "API is running. Use the `/predict` endpoint for predictions."}

@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        new_texts_cleaned = [clean(text) for text in request.texts]
        new_sequences = tokenizer.texts_to_sequences(new_texts_cleaned)
        new_padded = pad_sequences(new_sequences, maxlen=max_len, padding="post")
        new_predictions = model.predict(new_padded)
        new_labels = (new_predictions > 0.5).astype("int32").tolist()
        return {"predictions": new_labels}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
