#THIS IS AN AI GENERATED INTERFACE FOR TESTING THE MODEL

import streamlit as st
from predict import generate_next_words

st.title("FAQ Next-Word Text Generator")
st.write("Enter a starting phrase from the FAQ domain to predict the upcoming words.")

seed_text = st.text_input("Enter starting phrase:", "How do I")
next_words = st.slider("Number of words to predict:", 1, 10, 4)

if st.button("Generate Output"):
    if seed_text.strip() != "":
        result = generate_next_words(seed_text, next_words)
        st.success(f"**Predicted Text:** {result}")
    else:
        st.warning("Please enter some starting words.")