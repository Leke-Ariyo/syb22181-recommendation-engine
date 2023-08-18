from flask import Flask, request, jsonify
from flask_cors import CORS
import nltk.data
import openai
import pandas as pd
import numpy as np
import nltk
import os
import re
import ssl
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from spacy import displacy

app = Flask(__name__)
CORS(app)

# OpenAI Initialization
openai_api_key = "<ENTER_OPENAPI_KEY>"
openai.api_key = openai_api_key

# Define a variable to keep track of the last recommendation
last_recommendation = None

# Data Loading and Preprocessing
df = pd.read_csv('<PATH_TO_DATA_FILE>')
nlp = spacy.load('en_core_web_sm')
lemmatizer = WordNetLemmatizer()

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    doc = nlp(str(text).lower())
    tokens = [lemmatizer.lemmatize(token.text) for token in doc if token.text not in stop_words]
    return ' '.join(tokens)

for column in df.columns:
    if df[column].dtype == 'object':
        df[column] = df[column].apply(preprocess_text)

df['combined_text'] = df.apply(lambda row: ' '.join(row.astype(str)), axis=1)
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['combined_text'])

def extract_price_limit(query):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(query.lower()) # Convert the query to lowercase

    # Look for terms related to budget or price limit
    for token in doc:
        if token.text in ["at", "budget", "within", "under", "below", "less than"]:
            # Explore the token's subtree to find currency symbol or related terms, and numeric value
            for desc in token.subtree:
                if desc.text in ["£", "pounds", "gbp"] and desc.nbor(1).like_num:
                    return float(desc.nbor(1).text)
    return None


def preprocess_price(price):
    # Remove any non-digit characters except for the decimal point
    price = re.sub(r'[^\d.]', '', str(price))
    return float(price) if price else None

def identify_constraints(query):
    doc = nlp(query)
    constraints = []
    
    for token in doc:
        # Look for negations or other indicators of constraints
        if token.dep_ == 'neg' or token.text in ["without", "no"]:
            # Find the corresponding noun or adjective
            for descendant in token.head.subtree:
                if descendant.pos_ in ["NOUN", "ADJ"]:
                    constraints.append(descendant.lemma_)
                    
    return constraints


def get_similar_listings(query, top_n=3):
    # Preprocess the user query
    processed_query = preprocess_text(query)

    # Transform the query using the same TF-IDF vectorizer
    query_vector = vectorizer.transform([processed_query])

    # Calculate cosine similarity
    cosine_sim = cosine_similarity(query_vector, tfidf_matrix)

    # Get top N similar listings
    similar_indices = cosine_sim.argsort().flatten()[-top_n:]
    similar_listings = df.iloc[similar_indices[::-1]].copy()  # Reverse to get the most similar at the top

    # Add cosine similarity values
    similar_listings['cosine_similarity'] = cosine_sim[0, similar_indices[::-1]]

    # Filter listings by price constraint
    price_limit = extract_price_limit(query)
    if price_limit:
        # Convert price to numeric (removing any non-numeric characters and 'pcm')
        similar_listings['Price'] = similar_listings['Price'].apply(preprocess_price)
        #similar_listings['Price'] = similar_listings['Price'].replace('[\£,a-zA-Z\s]', '', regex=True).astype(float)
        similar_listings = similar_listings[similar_listings['Price'] <= price_limit]
    similar_listings = similar_listings.drop(columns=['combined_text'])
    # Identify constraints from the query
    constraints = identify_constraints(query)

    for constraint in constraints:
        for col in similar_listings.columns:
            # Check if the column contains string data
            if similar_listings[col].dtype == 'object':
                similar_listings = similar_listings[~similar_listings[col].str.contains(constraint, na=False)]

    return similar_listings

# Define a variable to keep track of conversation history
conversation_history = []

def get_similar_listings_based_on_last_recommendation(query, top_n=3):
    global last_recommendation
    
    # Check if there was a last recommendation
    if last_recommendation is None:
        return pd.DataFrame()

    # Get the previous listing's details as a new query
    new_query = last_recommendation['combined_text']
    
    # Add any new criteria from the user's query
    # You can extend this by identifying specific criteria in the user's query
    new_query += " " + preprocess_text(query)

    # Now call the existing get_similar_listings function with this new query
    return get_similar_listings(new_query, top_n)


@app.route('/query/', methods=['POST'])
def query():
    global last_recommendation

    user_query = request.form['user_query']
    conversation_history.append(user_query)

    # Check if this is a follow-up query
    doc = nlp(user_query.lower())
    follow_up_query = False
    for token in doc:
        # Identify follow-up queries based on dependency parsing, e.g., referring to a previous subject
        if token.dep_ == 'nsubj' and token.text == "this":
            follow_up_query = True
            break

    if follow_up_query:
        similar_listings = get_similar_listings_based_on_last_recommendation(user_query)
    else:
        similar_listings = get_similar_listings(user_query)

    # If listings are found, store the top one as the last recommendation
    if not similar_listings.empty:
        last_recommendation = similar_listings.iloc[0]

    # OpenAI integration
    introduction = ("You are Riches, a chatbot specialized in recommending properties based on user preferences. "
                    "The user is seeking advice on real estate listings. If you are recommending a property, outside the limit explain why. Do not recommend directly from GPT")
    full_conversation = introduction + "\n" + "\n".join(conversation_history)
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=full_conversation,
        max_tokens=100
    )
    model_reply = response.choices[0].text.strip()
    conversation_history.append(model_reply)

    # Prepare the JSON response with the model reply and similar listings
    result = {
        "model_reply": model_reply,
        "similar_listings": similar_listings.to_dict(orient='records')
    }

    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True)
