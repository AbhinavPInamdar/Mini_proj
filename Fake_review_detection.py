import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.callbacks import EarlyStopping

# Load and preprocess dataset
data = pd.read_csv(r"D:\Mini_proj\fake reviews dataset.csv")
data = data.dropna().drop_duplicates()

label_mapping = {"CG": 0, "OR": 1}  # Computer Generated and Original labels
data["label"] = data["label"].map(label_mapping)

if data["label"].isnull().any():
    raise ValueError("Some labels could not be mapped. Check your dataset!")

X_text = data["text_"].values
y = data["label"].values

# Text cleaning function
def clean(text):
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = text.lower().strip()
    return text

data["text_"] = data["text_"].apply(clean)

# Tokenization and padding
max_words = 5000
max_len = 100
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(data["text_"])
x_seq = tokenizer.texts_to_sequences(data["text_"])
x_pad = pad_sequences(x_seq, maxlen=max_len, padding="post")

x_train, x_test, y_train, y_test = train_test_split(x_pad, y, test_size=0.2, random_state=42)

# Compute class weights
class_weights = compute_class_weight(class_weight="balanced", classes=np.array([0, 1]), y=y)
class_weights_dict = dict(enumerate(class_weights))
print("Class weights:", class_weights_dict)

# Model definition
embedding_dim = 100
model = Sequential([
    Embedding(input_dim=max_words + 1, output_dim=embedding_dim, input_length=max_len),
    LSTM(128),
    Dropout(0.5),
    Dense(1, activation="sigmoid")
])

# Compile the model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

# Early stopping callback
early_stopping = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

# Train the model
history = model.fit(
    x_train, y_train,
    validation_split=0.2,
    batch_size=32,
    epochs=10,
    class_weight=class_weights_dict,
    callbacks=[early_stopping],
    verbose=1
)

# Save the model in .keras format
model.save('saved_model.keras')

# Load the saved model
loaded_model = load_model('saved_model.keras')
loaded_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])  # Optional

# Evaluate the loaded model
print("\nEvaluating the loaded model...")
test_loss, test_accuracy = loaded_model.evaluate(x_test, y_test, verbose=1)
print(f"Loaded Model - Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

# Predict new text samples
new_texts = ["This fan is really pretty and I actually use it.", "The worst purchase I ever made."]
new_texts_cleaned = [clean(text) for text in new_texts]
new_sequences = tokenizer.texts_to_sequences(new_texts_cleaned)
new_padded = pad_sequences(new_sequences, maxlen=max_len, padding="post")

new_predictions_prob = loaded_model.predict(new_padded)
for text, prob in zip(new_texts, new_predictions_prob):
    print(f"Text: {text}\nPrediction Probability: {prob[0]:.4f}\nPredicted Label: {int(prob[0] > 0.5)}\n")


import json
with open("tokenizer_config.json", "w") as f:
    f.write(tokenizer.to_json())




