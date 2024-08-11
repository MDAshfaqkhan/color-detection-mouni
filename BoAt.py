import streamlit as st
import pandas as pd 
import pickle
from PIL import Image
import io

model = pickle.load(open(r"C:\Users\Ashfaq Khan\283-Batch\Machine Learning Nagaraju sir\BoAt_Nirvana.pkl", "rb"))

st.image('innomatics_research_labs_logo.png')
st.image('nirvana_ion_image1.jpg')
st.image('nirvana_ion_image2.jpg')
st.image('nirvana_ion_image3.jpg')

st.title("Sentiment Analysis on BoAt Nirvana Ion Ear Buds")

# Load the CSV file
df = pd.read_csv("combined_file__.csv")

# Load the pre-trained model and vectorizer
model_path = "BoAt_Nirvana.pkl"
vectorizer_path = "Bagw_vectorization.pkl"

with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

with open(vectorizer_path, 'rb') as vectorizer_file:
    Bagw = pickle.load(vectorizer_file)

# Input review from user
review_input = st.text_input("Enter your opinion")

# Transform the input review using the loaded vectorizer
if st.button("Submit"):
    data = Bagw.transform([review_input]).toarray()
    pred = model.predict(data)[0]
    
    if pred == "Negative":
        neg = Image.open("negative.png")
        st.image(neg)
    elif pred == "Positive":
        pos = Image.open("positive.png")
        st.image(pos)
    else:
        st.write("Neutral")
