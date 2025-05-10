import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import nltk
import gradio as gr
import os
import logging
import tensorflow as tf

# Suppress TensorFlow warnings and set logging level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow info/warning logs
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Download NLTK data
nltk.download('punkt', quiet=True)

# Load the dataset
def load_data():
    file_path = "zomato_data.csv"
    try:
        zomato_data = pd.read_csv(file_path)
    except FileNotFoundError:
        print("Warning: zomato_data.csv not found. Using dummy dataset.")
        zomato_data = pd.DataFrame({
            'name': ['Sample Restaurant'],
            'online_order': ['Yes'],
            'book_table': ['No'],
            'rate': [4.0],
            'approx_cost': [1000.0],
            'listed_in(type)': ['Dining']
        })
    return zomato_data

# Preprocess data
def preprocess_data(zomato_data):
    zomato_data.rename(columns={'approx_cost(for two people)': 'approx_cost'}, inplace=True)
    zomato_data['approx_cost'] = zomato_data['approx_cost'].astype(str).str.replace(',', '', regex=True).astype(float)
    zomato_data['rate'] = zomato_data['rate'].astype(str).str.split('/').str[0].replace(['NEW', '-', 'nan'], None).astype(float)
    zomato_data.fillna("Unknown", inplace=True)
    zomato_data['search_text'] = (
        "Name: " + zomato_data['name'] + " | " +
        "Online Order: " + zomato_data['online_order'] + " | " +
        "Book Table: " + zomato_data['book_table'] + " | " +
        "Rate: " + zomato_data['rate'].astype(str) + " | " +
        "Cost for Two: ₹" + zomato_data['approx_cost'].astype(str) + " | " +
        "Cuisine Type: " + zomato_data['listed_in(type)']
    )
    return zomato_data

# Generate Q&A pairs
def generate_qa_pairs(zomato_data):
    qa_pairs = []
    for _, row in zomato_data.iterrows():
        qa_pairs.append((f"Can I order online at {row['name']}?", 
                         f"{'Yes' if row['online_order'] == 'Yes' else 'No'}, online ordering is {'available' if row['online_order'] == 'Yes' else 'not available'} at {row['name']}."))
        qa_pairs.append((f"Can I book a table at {row['name']}?", 
                         f"{'Yes' if row['book_table'] == 'Yes' else 'No'}, table booking is {'available' if row['book_table'] == 'Yes' else 'not available'} at {row['name']}."))
        qa_pairs.append((f"What is the rating of {row['name']}?", f"The rating of {row['name']} is {row['rate']}/5."))
        qa_pairs.append((f"What is the cost for two people at {row['name']}?", 
                         f"The approximate cost for two people at {row['name']} is ₹{row['approx_cost']}."))
        qa_pairs.append((f"What type of cuisine does {row['name']} serve?", 
                         f"{row['name']} is listed in {row['listed_in(type)']}."))
    return qa_pairs

# Initialize vectorizer
def init_vectorizer(zomato_data):
    vectorizer = TfidfVectorizer()
    vectorized_data = vectorizer.fit_transform(zomato_data['search_text'])
    return vectorizer, vectorized_data

# Load tokenizer and model
def load_model_and_tokenizer(zomato_data):
    model = None
    tokenizer = None
    try:
        model = load_model("chatbot_model.h5")
        tokenizer = Tokenizer()
        qa_pairs = generate_qa_pairs(zomato_data)
        questions, answers = zip(*qa_pairs)
        tokenizer.fit_on_texts(questions + answers)
        print("Successfully loaded model and initialized tokenizer.")
    except FileNotFoundError:
        print("Warning: chatbot_model.h5 not found. Falling back to TF-IDF-based responses.")
    except Exception as e:
        print(f"Error loading model: {e}. Falling back to TF-IDF-based responses.")
    return model, tokenizer

# Chatbot query using TF-IDF
def chatbot_query(query, vectorizer, vectorized_data, zomato_data):
    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, vectorized_data)
    most_similar_idx = np.argmax(similarities)
    return zomato_data['search_text'].iloc[most_similar_idx]

# Chatbot response using LSTM model
def chatbot_response(input_text, model, tokenizer, vectorizer, vectorized_data, zomato_data, max_len=20):
    if model and tokenizer:
        input_seq = tokenizer.texts_to_sequences([input_text])
        input_seq = pad_sequences(input_seq, maxlen=max_len, padding='post')
        try:
            pred = model.predict([input_seq, input_seq], verbose=0)
            pred_idx = np.argmax(pred[0], axis=1)
            response = " ".join([tokenizer.index_word.get(idx, "") for idx in pred_idx if idx > 0])
            if response.strip():
                return response
        except Exception as e:
            print(f"Model prediction failed: {e}")
    # Fallback to TF-IDF
    return chatbot_query(input_text, vectorizer, vectorized_data, zomato_data)

# Initialize everything
zomato_data = load_data()
zomato_data = preprocess_data(zomato_data)
vectorizer, vectorized_data = init_vectorizer(zomato_data)
model, tokenizer = load_model_and_tokenizer(zomato_data)

# Gradio interface
def gradio_chatbot(user_query):
    if not user_query.strip():
        return "Please enter a valid query."
    response = chatbot_response(user_query, model, tokenizer, vectorizer, vectorized_data, zomato_data)
    return response

# Create Gradio interface
iface = gr.Interface(
    fn=gradio_chatbot,
    inputs=gr.Textbox(lines=2, placeholder="Ask about a restaurant (e.g., 'Can I order online at Cafe XYZ?')"),
    outputs="text",
    title="Zomato Restaurant Chatbot",
    description="Ask questions about restaurants, such as online ordering, table booking, ratings, costs, or cuisine types."
)

# Launch the app
if _name_ == "_main_":
    iface.launch()
