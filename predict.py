import pickle

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

#loading the model and tokenizer
model = load_model("lstm_model.keras")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)


#get required sequence length from the model
max_sequence_len = model.input_shape[1]

def generate_next_words(seed_text, next_words = 5):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen = max_sequence_len, padding = "pre")
        
        predicted_probs = model.predict(token_list, verbose = 0)
        predicted_index = np.argmax(predicted_probs, axis = -1)[0]
        
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted_index:
                output_word = word
                break
        
        if output_word == "":
            break
        seed_text += " " + output_word
        
    return seed_text

if __name__ == "__main__":
    print("\n--- Text Predictor ---")
    print("Type 'exit' to stop.\n")
    
    while True:
        text = input("Enter starting words: ")
        if text.lower() == "exit":
            break
        
        num_words = int(input("How many words to predict? "))
        result = generate_next_words(text, num_words)

        print("Output:", result)
        print("-" * 10)