import pandas as pd
import numpy as np
import re
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
from tensorflow.keras.callbacks import EarlyStopping

data = pd.read_csv(r"D:\Fake_review_detection\fake reviews dataset.csv")
data = data.dropna().drop_duplicates()

label_mapping = {
    "CG": 0,  # Computer Generated
    "OR": 1   # Original
}
data["label"] = data["label"].map(label_mapping)

if data["label"].isnull().any():
    raise ValueError("Some labels could not be mapped. Check your dataset!")

X_text = data["text_"].values
y = data["label"].values

def clean(text):
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)  # Retain alphanumeric and spaces
    text = re.sub(r"\s+", " ", text)             # Remove extra spaces
    text = text.lower().strip()                 # Convert to lowercase
    return text

data["text_"] = data["text_"].apply(clean)

# Tokenization and padding
max_words = 5000  # Maximum vocabulary size
max_len = 100     # Maximum sequence length
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(data["text_"])
x_seq = tokenizer.texts_to_sequences(data["text_"])
x_pad = pad_sequences(x_seq, maxlen=max_len, padding="post")

x_train, x_test, y_train, y_test = train_test_split(x_pad, y, test_size=0.2, random_state=42)

# Compute class weights
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.array([0, 1]),  # Explicitly include both classes
    y=y
)
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

# Evaluate the model
print("\nEvaluating on test data...")
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=1)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

# Classification report
y_pred = (model.predict(x_test) > 0.5).astype("int32")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

#%%
# Sample new text data
new_texts = ["This fan is really pretty and I actually use it.", "The worst purchase I ever made."]
# Clean and preprocess the text
new_texts_cleaned = [clean(text) for text in new_texts]
new_sequences = tokenizer.texts_to_sequences(new_texts_cleaned)
new_padded = pad_sequences(new_sequences, maxlen=max_len, padding="post")

# Predict using the trained model
new_predictions = model.predict(new_padded)
new_labels = (new_predictions > 0.5).astype("int32")

print(f"Predictions for new data: {new_labels}")



