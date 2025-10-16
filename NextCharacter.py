# Predicting next character in a sentence using SimpleRNN
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Embedding
from tensorflow.keras.utils import to_categorical

# --- Corpus & Vocabulary Mapping ---
text = "machine learning is amazing"
chars = sorted(list(set(text)))
char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for i, c in enumerate(chars)}

# Parameters
seq_length = 5   # Number of characters to look back
vocab_size = len(chars)

# --- Prepare dataset (Input sequences and next character) ---
X, Y = [], []
for i in range(len(text) - seq_length):
    seq = text[i:i + seq_length]
    next_char = text[i + seq_length]
    X.append([char_to_idx[c] for c in seq])
    Y.append(char_to_idx[next_char])

# Convert to NumPy arrays
X = np.array(X)
Y = to_categorical(Y, num_classes=vocab_size)

# --- Build the RNN model ---
model = Sequential([
    Embedding(vocab_size, 10, input_length=seq_length),
    SimpleRNN(50, activation='tanh'),
    Dense(vocab_size, activation='softmax')
])

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X, Y, epochs=200, verbose=0)

# --- Text generation function ---
def generate_text(seed_text, length=50):
    generated = seed_text
    for _ in range(length):
        x_pred = np.array([[char_to_idx[c] for c in generated[-seq_length:]]])
        pred = model.predict(x_pred, verbose=0)
        next_idx = np.argmax(pred)
        next_char = idx_to_char[next_idx]
        generated += next_char
    return generated

# --- Run text generation ---
seed = "machi"
print("Generated text:\n", generate_text(seed, length=40))
