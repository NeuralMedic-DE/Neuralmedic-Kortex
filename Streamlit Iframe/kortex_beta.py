import streamlit as st
import configparser
from typing import Optional, List
from sentence_transformers import SentenceTransformer
import faiss
import google.generativeai as genai
from google.generativeai import GenerativeModel

# Custom Google Gemini LLM
class GoogleGeminiLLM:
    def __init__(self, api_key: str):
        self.api_key = api_key
        genai.configure(api_key=self.api_key)
        self.model = GenerativeModel(model_name='gemini-pro')

    def call(self, prompt: str) -> str:
        response = self.model.generate_content(prompt)
        # Print the response attributes to inspect
        print(dir(response))
        # Assuming response has 'content' attribute
        return response.content

# Load environment variables from the .ini file
config = configparser.ConfigParser()
config.read('.ini')

# Access the API key from the environment
GOOGLE_API_KEY = config['api_key']['GOOGLE_API_KEY']

# Read context from a text file
def load_context(file_path):
    with open(file_path, 'r') as file:
        context = file.read()
    return context

# Path to the context text file
context_file_path = 'results.txt'
eeg_context = load_context(context_file_path)

# Create a vector store from your EEG context using Sentence Transformers
embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
context_embeddings = embedding_model.encode([eeg_context])

# Initialize FAISS index
d = context_embeddings.shape[1]
index = faiss.IndexFlatL2(d)
index.add(context_embeddings)

def retrieve_context(query: str, top_k: int = 1):
    query_embedding = embedding_model.encode([query])
    D, I = index.search(query_embedding, top_k)
    return [eeg_context]

# Initialize the Google Gemini LLM with the api_key
llm = GoogleGeminiLLM(api_key=GOOGLE_API_KEY)

st.title('Gemini-Kortex')

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Ask me Anything"
        }
    ]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Process and store Query and Response
def llm_function(query):
    # Retrieve context using FAISS
    context = retrieve_context(query)
    context_response = " ".join(context)

    # Generate response using Google Gemini API with context
    response = llm.call(context_response + " " + query)

    # Displaying the Assistant Message
    with st.chat_message("assistant"):
        st.markdown(response)

    # Storing the User Message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": query
        }
    )

    # Storing the Assistant Message
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response
        }
    )

# Accept user input
query = st.chat_input("")

# Calling the Function when Input is Provided
if query:
    # Displaying the User Message
    with st.chat_message("user"):
        st.markdown(query)

    llm_function(query)
