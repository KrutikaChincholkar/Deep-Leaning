# Predicting next Character in a Sentence

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Embedding
from tensorflow.keras.utils import to_categorical

# Mapping
text = "machine learning is amazing"
chars = sorted(list(set(text)))
char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for i, c in enumerate(chars)}

# Parameters
seq_length = 5  # look at 5 chars to predict the next one
vocab_size = len(chars)

# Prepare dataset (building sequences)
X, Y = [], []
for i in range(len(text) - seq_length):
    seq_in = text[i:i + seq_length]
    next_char = text[i + seq_length]
    X.append([char_to_idx[c] for c in seq_in])
    Y.append(char_to_idx[next_char])

# Convert to arrays & one-hot encode outputs
X = np.array(X)
Y = to_categorical(Y, num_classes=vocab_size)

# Model definition
model = Sequential([
    Embedding(vocab_size, 10, input_length=seq_length),
    SimpleRNN(50, activation='tanh'),
    Dense(vocab_size, activation='softmax')
])

# Compile & train
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, Y, epochs=200, verbose=0)

# Function to predict next characters
def predict_next(seed_text, num_chars=50):
    result = seed_text
    for _ in range(num_chars):
        x_pred = np.array([[char_to_idx[c] for c in result[-seq_length:]]])
        pred = model.predict(x_pred, verbose=0)
        next_idx = np.argmax(pred)
        result += idx_to_char[next_idx]
    return result

# Example prediction
seed = "machi"
print("Seed:", seed)
print("Generated Text:", predict_next(seed, num_chars=30))
