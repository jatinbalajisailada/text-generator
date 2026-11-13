# Text Generator using LSTM

## Problem Statement
Design and implement an LSTM-based deep learning model that learns domain-specific linguistic patterns from a custom FAQ dataset and predicts the next word(s) in a given text sequence, enabling contextually relevant text completion and generation.

## Project Overview
This project uses a Deep Learning approach with an **Embedding Layer** and **LSTM (Long Short-Term Memory)** network built in TensorFlow/Keras. The model is trained on a custom FAQ dataset using N-gram sequence tokenization and pre-padding to predict subsequent words from a user-provided starting phrase.

To allow immediate testing without waiting for training epochs, the pre-trained model (`lstm_model.keras`) and fitted tokenizer (`tokenizer.pkl`) are included directly in this repository.

## Project Structure
```text
text-generator/
├── faq_dataset.txt          # Custom FAQ text dataset
├── train.py                 # Script to preprocess data, train the LSTM, and save files
├── predict.py               # Interactive terminal script to test inputs and view outputs
├── lstm_model.keras         # Pre-trained LSTM model weights
├── tokenizer.pkl            # Saved Keras tokenizer
├── text_generator.ipynb           # Original Jupyter Notebook used for experimentation
├── requirements.txt         # Required Python libraries (tensorflow, numpy)
└── README.md                # Project documentation and setup guide
```

## How to Run Locally
### 1. Clone the Repository
```code
git clone [https://github.com/jatinbalajisailada/next-word-text-generator.git](https://github.com/jatinbalajisailada/next-word-text-generator.git)
cd next-word-text-generator
```
### 2. Install Dependencies
```code
pip install -r requirements.txt
```

### 3. Run the Predictor (Instant Output)
Run the prediction script to give input phrases and see the generated output directly in the terminal:
```code
python predict.py
```

Example Terminal Interaction:
```text
--- FAQ Next-Word Predictor ---
Type 'exit' to stop.

Enter starting words: what is the total duration
How many words to predict? 3
Output: what is the total duration of the course
-----------------------------------
```

### 4. Retrain the Model (Optional)
If you modify faq_dataset.txt and want to train the model from scratch:
```text
python train.py
```

## Sample Inputs to Try
Since the model is trained on the CampusX Data Science Mentorship Program FAQ dataset, try these starting phrases and word counts for clean predictions:

`"What is the total duration"` (Predict 3 words → of the course)

`"Where can I find"` (Predict 3 words → the class schedule)

`"How will I be informed"` (Predict 4 words → about the upcoming class)

`"You have to make all your"` (Predict 5 words → monthly payments on our website)

`"What is the criteria"` (Predict 4 words → to get the certificate)

`"You get a 7 days"` (Predict 2 words → refund period)

`"We will be covering"` (Predict 3 words → the following modules)
