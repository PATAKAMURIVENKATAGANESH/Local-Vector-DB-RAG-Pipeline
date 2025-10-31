import os
import pandas as pd
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import weaviate
from langchain_weaviate.vectorstores import WeaviateVectorStore
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from rich.console import Console
from rich.markdown import Markdown
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from llama_parse import LlamaParse  # Assuming LlamaParse is available as a library

# Load environment variables
load_dotenv()

# Configure environment variables
OLLAMA_HOST = "http://localhost:11434"  # Default Ollama host
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize Weaviate client
weaviate_client = weaviate.connect_to_local(skip_init_checks=True)
parser = LlamaParse(api_key="llx-KfM0zeantsWgMgMuBT7XipQ6UyVemRFrlqaclzodYwHT8dhn", parsing_instruction="The sheets contain information regarding the lab management system and maintenance.", result_type="markdown")
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
Use the following context to answer the question:
Context: {context}
Question: {question}
Answer Should be properly summarised with detailed step by step explanation and of good length and flow and suggest the follow up questions for user to ask to get the better clarity. Answer in markdown format:
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# Initialize Ollama LLM
llm = ChatGroq(api_key=groq_api_key, model="deepseek-r1-distill-llama-70b")

# Initialize RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    retriever=static_db.as_retriever(search_kwargs={"k": 3}),
    chain_type="stuff",
    llm=llm,
    chain_type_kwargs={"prompt": prompt}
)

# Define keywords for fact-based information
FACT_BASED_KEYWORDS = ["lab location", "pc assigned", "ETA", "vacation calendar", "leaves", "zero balancing", "A2-65 LAB", "A2-66 LAB", "jenkins automation", "BLR-ROW", "lab dashboard", "pc", "setup", "setup1", "setup2"]

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
        qa_chain.retriever = dynamic_db.as_retriever(search_kwargs={"k": 3})
    else:
        # Use static documents for summarized or explanatory answers
        print("Fetching summarized information from static documents...")
        qa_chain.retriever = static_db.as_retriever(search_kwargs={"k": 3})

    # Invoke the QA chain
    response = qa_chain.invoke({"query": query})
    return response["result"]

def process_folder(folder_path, document_type="static"):
    """
    Process all PDF and Excel files in a folder and store their embeddings in Weaviate.
    Args:
        folder_path (str): Path to the folder containing files.
        document_type (str): Type of documents ("static" or "dynamic").
    """
    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Split text into chunks of 1000 characters
        chunk_overlap=200,  # Overlap chunks by 200 characters for context
    )

    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        print(f"Processing file: {file_name}")

        if file_name.endswith(".pdf"):
            # Load PDF content
            loader = PyPDFLoader(file_path)
            pages = loader.load()

            # Split documents into smaller chunks
            chunks = text_splitter.split_documents(pages)

            # Store each chunk in Weaviate
            for chunk in chunks:
                metadata = {
                    "file_name": file_name,
                    "file_path": file_path,
                    "document_type": document_type
                }
                document = Document(page_content=chunk.page_content, metadata=metadata)
                if document_type == "static":
                    static_db.add_documents([document])
                else:
                    dynamic_db.add_documents([document])
            print(f"PDF documents embedded as {document_type}.")

        elif file_name.endswith(".xlsx") or file_name.endswith(".xls"):
            # Load Excel content using LlamaParse
            # parser = LlamaParse()
            llama_parse_documents = parser.load_data(file_path)

            # Convert parsed data into LangChain Documents
            with open('output.md', 'a') as f:  # Open the file in append mode ('a')
                for doc in llama_parse_documents:
                    f.write(doc.text + '\n')

            markdown_path = "output.md"
            loader = UnstructuredMarkdownLoader(markdown_path)

            documents = loader.load()
            # Split documents into smaller chunks
            # chunks = text_splitter.split_documents(documents)

            # Store each chunk in Weaviate
            # for chunk in chunks:
            #     metadata = {
            #         "file_name": file_name,
            #         "file_path": file_path,
            #         "document_type": document_type
            #     }
            #     document = Document(page_content=chunk.page_content, metadata=metadata)
            if document_type == "static":
                static_db.add_documents(documents)
            else:
                dynamic_db.add_documents(documents)
            print(f"Excel documents embedded as {document_type}.")

        else:
            print(f"Skipping unsupported file: {file_name}")
            continue

if __name__ == "__main__":
    # Folder containing the documents
    static_folder_path = "C:/Users/nxa24481/Downloads/ganesh/AI/docs/static"
    dynamic_folder_path = "C:/Users/nxa24481/Downloads/ganesh/AI/docs/dynamic"

    # Preprocess and store documents in Weaviate
    # process_folder(static_folder_path, document_type="static")
    process_folder(dynamic_folder_path, document_type="dynamic")

    # Query the LLM using the RetrievalQA chain
    query = "which test setup are connected to NRW93436?"
    print("Invoked")
    response = handle_query(query)

    # Print the response in a nicely formatted way
    console = Console()
    console.print("[bold green]LLM Response:[/bold green]")
    console.print(Markdown(response))