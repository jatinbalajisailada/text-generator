import pickle

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical

#loading dataset to train
with open("faq_dataset.txt", "r", encoding="utf-8") as f:
    text = f.read()


#tokenize the text
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
total_words = len(tokenizer.word_index) + 1


#convert data into sequences for next word prediction
input_sequences = []

for line in text.split("\n"):
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)


#padding and splitting into X and y
max_seq_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen = max_seq_len, padding = "pre"))

X = input_sequences[:, :-1]
y = input_sequences[:, -1]
y = to_categorical(y, num_classes=total_words)


#architecture
model = Sequential([
    Embedding(total_words, 100, input_length = max_seq_len - 1),
    LSTM(150),
    Dense(total_words, activation = "softmax")
])

#compilation
model.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])
model.fit(X, y, epochs = 50, verbose = 1)


#export model and tokenizer
model.save("lstm_model.keras")
with open("tokenizer.pkl", "wb") as t:
    pickle.dump(tokenizer, t)

print("Training complete! Model and tokenizer saved.")