import os
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import weaviate
from langchain_weaviate.vectorstores import WeaviateVectorStore
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from rich.console import Console
from rich.markdown import Markdown
from weaviate.classes.query import Filter

# Load environment variables
load_dotenv()

# Configure environment variables
OLLAMA_HOST = "http://localhost:11434"  # Default Ollama host
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

# Initialize Weaviate vector store
db = WeaviateVectorStore(
    client=weaviate_client,
    embedding=SentenceTransformerEmbeddings(embeddings_model),  # Use the wrapper class
    index_name="Documents",  # Name of the Weaviate collection
    text_key="content"  # Property in the collection that contains the text
)
# Configure prompt for QA chain
prompt_template = """
Use the following context to answer the question:
Context: {context}
Question: {question}
Answer in markdown format:
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# Initialize Ollama LLM
llm = OllamaLLM(base_url=OLLAMA_HOST, model="llama2")

# Initialize RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    retriever=db.as_retriever(search_kwargs={"k": 3}),
    chain_type="stuff",
    llm=ChatGroq(api_key=groq_api_key, model="llama-3.1-8b-instant"),
    chain_type_kwargs={"prompt": prompt}
)


def process_pdf_folder(folder_path):
    """
    Process all PDF files in a folder and store their embeddings in Weaviate.
    Args:
        folder_path (str): Path to the folder containing PDF files.
    """
    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Split text into chunks of 1000 characters
        chunk_overlap=200,  # Overlap chunks by 200 characters for context
    )

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".pdf"):
            file_path = os.path.join(folder_path, file_name)
            print(f"Processing file: {file_name}")

            # Load PDF content
            loader = PyPDFLoader(file_path)
            pages = loader.load()

            # Split documents into smaller chunks
            chunks = text_splitter.split_documents(pages)

            # Store each chunk in Weaviate
            for chunk in chunks:
                metadata = {"file_name": file_name, "file_path": file_path}
                document = Document(page_content=chunk.page_content, metadata=metadata)
                db.add_documents([document])
            print("documents_embedded")

if __name__ == "__main__":
    # Folder containing the PDFs
    folder_path = "C:/Users/nxa24481/Downloads/ganesh/AI/docs"

    # Preprocess and store documents in Weaviate
    # process_pdf_folder(folder_path)

    # Query the LLM using the RetrievalQA chain
    