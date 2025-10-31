from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import pandas as pd
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import weaviate
from langchain_weaviate.vectorstores import WeaviateVectorStore
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyPDFLoader, DataFrameLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from rich.console import Console
from rich.markdown import Markdown

# Load environment variables
load_dotenv()

# Configure environment variables
# OLLAMA_HOST = "http://localhost:11434"  # Default Ollama host
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize Weaviate client
weaviate_client = weaviate.connect_to_local(skip_init_checks=True)

# Configure embeddings model
embeddings_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

# Initialize Weaviate vector store
class SentenceTransformerEmbeddings:
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()

    def embed_query(self, text):
        return self.model.encode(text).tolist()

# Initialize Weaviate vector stores for static and dynamic documents
static_db = WeaviateVectorStore(
    client=weaviate_client,
    embedding=SentenceTransformerEmbeddings(embeddings_model),
    index_name="StaticDocuments",  # Collection for static documents
    text_key="content"
)

dynamic_db = WeaviateVectorStore(
    client=weaviate_client,
    embedding=SentenceTransformerEmbeddings(embeddings_model),
    index_name="DynamicDocuments",  # Collection for dynamic documents
    text_key="content"
)

# Configure prompt for QA chain
prompt_template = """
ROLE: ACT AS A TEXT ANALYSER, SUMMARIZER, DEBUGGER AND INFORMATION DESK, SOFTWARE ENGINEER, TESTER
Use the following context to answer the question:
Context: {context}
Question: {question}
DO NOT ANSWER WHAT YOU ARE THINKING JUST Answer Should be precise to question asked and properly explained and Summarised with good length and flow and source of the document should be mentioned and also suggest the follow up questions for user to ask to get the better clarity. Answer in markdown format:
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# Initialize Ollama LLM
# llm = OllamaLLM(base_url=OLLAMA_HOST, model="llama2")

# Initialize RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    retriever=static_db.as_retriever(search_kwargs={"k": 3}),
    chain_type="stuff",
    llm=ChatGroq(api_key=groq_api_key, model="llama-3.3-70b-versatile"), #llama-3.3-70b-versatile
    chain_type_kwargs={"prompt": prompt}
)

# Define keywords for fact-based information
FACT_BASED_KEYWORDS = []

def contains_fact_based_keywords(query):
    """
    Check if the query contains any fact-based keywords.
    Args:
        query (str): The user query.
    Returns:
        bool: True if the query contains fact-based keywords, False otherwise.
    """
    return any(keyword.lower() in query.lower() for keyword in FACT_BASED_KEYWORDS)

def handle_query(query):
    """
    Handle the query based on whether it contains fact-based keywords.
    Args:
        query (str): The user query.
    Returns:
        str: The response from the LLM.
    """
    if contains_fact_based_keywords(query):
        # Use dynamic documents for fact-based answers
        print("Fetching fact-based information from dynamic documents...")
        qa_chain.retriever = dynamic_db.as_retriever(search_kwargs={"k": 10})
    else:
        # Use static documents for summarized or explanatory answers
        print("Fetching summarized information from static documents...")
        qa_chain.retriever = static_db.as_retriever(search_kwargs={"k": 10})

    # Invoke the QA chain
    response = qa_chain.invoke({"query": query})
    return response["result"]

app = Flask(__name__)
CORS(app)

@app.route('/query', methods=['POST'])
def query():
    data = request.json
    query_text = data.get('query')
    if not query_text:
        return jsonify({"error": "No query provided"}), 400
    response = handle_query(query_text)
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug=True)