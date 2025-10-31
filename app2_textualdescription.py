import os
import pandas as pd
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
# from langchain_ollama.llms import OllamaLLM
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import weaviate
from langchain_weaviate.vectorstores import WeaviateVectorStore
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyPDFLoader, DataFrameLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from rich.console import Console
from rich.markdown import Markdown
import json
from pathlib import Path
from langchain_community.document_loaders import BSHTMLLoader



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
dynamic_db = WeaviateVectorStore(
    client=weaviate_client,
    embedding=SentenceTransformerEmbeddings(embeddings_model),
    index_name="DynamicDocuments",  # Collection for dynamic documents
    text_key="content"
)
log_db = WeaviateVectorStore(
    client=weaviate_client,
    embedding=SentenceTransformerEmbeddings(embeddings_model),
    index_name="LogDocuments",  # Collection for dynamic documents
    text_key="content"
)

# Configure prompt for QA chain
# Use the following context to answer the question:
# Context: {context}
# Question: {question}
"""Answer Should be properly summarised with detailed step by step explanation and of good length and flow and suggest the follow up questions for user to ask to get the better clarity. Answer in markdown format:"""
prompt_template = """
YOU ARE HTML FILE LOG REVIEWER and review the files based on the values inside the tables of the files. 
Use the following context to answer the question:
Context: {context}
Question: {question}
 Answer in markdown format:
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
    # if contains_fact_based_keywords(query):
    #     # Use dynamic documents for fact-based answers
    #     print("Fetching fact-based information from dynamic documents...")
    #     qa_chain.retriever = dynamic_db.as_retriever(search_kwargs={"k": 3})
    # else:
    #     # Use static documents for summarized or explanatory answers
    #     print("Fetching summarized information from static documents...")
    #     qa_chain.retriever = static_db.as_retriever(search_kwargs={"k": 3})

    # Invoke the QA chain
    response = qa_chain.invoke({"query": query})
    return response["result"]

def generate_textual_description(row_data):
    """
    Generate a textual description for a row of data using the LLM.
    Args:
        row_data (dict): A dictionary representing a row of data.
    Returns:
        str: A natural language description of the row.
    """
    prompt = f"""
    Generate a concise and informative textual description for the following data and return back in jsonl format only:
    {json.dumps(row_data, indent=2)}
    """
    response = llm.invoke(prompt)
    return response.content.strip()


def process_folder2(folder_path, document_type="static"):
    """
    Recursively process all PDF and AIDL files in a folder and its subfolders.
    
    Args:
        folder_path (str): Path to the root folder containing files.
        document_type (str): Type of documents ("static" or "dynamic").
    """
    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Split text into chunks of 1000 characters
        chunk_overlap=200,  # Overlap chunks by 200 characters for context
    )

    # Recursively traverse directories
    for file_path in Path(folder_path).rglob("*"):
        if file_path.suffix.lower() in [".pdf", ".aidl"]:  # Check for PDF or AIDL files
            print(f"Processing file: {file_path}")

            if file_path.suffix.lower() == ".aidl":
                # Load PDF content
                loader = TextLoader(file_path)
                pages = loader.load()

                # Split documents into smaller chunks
                chunks = text_splitter.split_documents(pages)

                for chunk in chunks:
                    metadata = {
                        "file_name": file_path.suffix.lower(),
                        "file_path": file_path,
                        "document_type": document_type  # Add document type to metadata
                    }
                    document = Document(page_content=chunk.page_content, metadata=metadata)
                    if document_type == "static":
                        static_db.add_documents([document])
                    else:
                        dynamic_db.add_documents([document])
                print(f"Documents embedded as {document_type}.")
            

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
        print(file_path)
        print(f"Processing file: {file_name}")

        if file_name.endswith(".pdf"):
            # Load PDF content
            loader = PyPDFLoader(file_path)
            pages = loader.load()

            # Split documents into smaller chunks
            chunks = text_splitter.split_documents(pages)

        # elif file_name.endswith(".xlsx") or file_name.endswith(".xls"):
        #     # Load Excel content
        #     df = pd.read_excel(file_path)
        #     documents = []
        #     for _, row in df.iterrows():
        #         page_content = ", ".join([f"{col}: {row[col]}" for col in df.columns])
        #         print(page_content)
        #         metadata = {"file_name": file_name, "file_path": file_path, "document_type": document_type}
        #         document = Document(page_content=page_content, metadata=metadata)
        #         documents.append(document)
        #     chunks = text_splitter.split_documents(documents)
        elif file_name.endswith(".html"):
            # Load PDF content
            loader = BSHTMLLoader(file_path)
            pages = loader.load()

            # Split documents into smaller chunks
            chunks = text_splitter.split_documents(pages)

        # elif file_name.endswith(".xlsx") or file_name.endswith(".xls"):
        #     # Load Excel content
        #     df = pd.read_excel(file_path)
        #     documents = []
        #     for _, row in df.iterrows():
        #         page_content = ", ".join([f"{col}: {row[col]}" for col in df.columns])
        #         print(page_content)
        #         metadata = {"file_name": file_name, "file_path": file_path, "document_type": document_type}
        #         document = Document(page_content=page_content, metadata=metadata)
        #         documents.append(document)
        #     chunks = text_splitter.split_documents(documents)

        elif file_name.endswith(".xlsx") or file_name.endswith(".xls"):
            # Load Excel content
            df = pd.read_excel(file_path)
            documents = []

            # Convert each row to JSONL with an empty textual_description
            jsonl_data = []
            for _, row in df.iterrows():
                row_data = row.to_dict()
                row_data["textual_description"] = "" 
                # print(row_data) # Add empty textual_description
                jsonl_data.append(row_data)

            # Send JSONL to LLM to fill textual_description
            for row_data in jsonl_data:
                row_data["textual_description"] = generate_textual_description(row_data)
                print(row_data)

            # Store enriched JSONL in Weaviate
            for row_data in jsonl_data:
                metadata = {
                    "file_name": file_name,
                      # Assuming single sheet
                    "row_number": _ + 2,  # Excel rows start from 1, header is row 1
                    "document_type": document_type
                }

                # Create document
                document = Document(
                    page_content=json.dumps(row_data),  # Store the entire JSONL row
                    metadata=metadata
                )
                documents.append(document)

            # Split documents into smaller chunks
            chunks = documents

        else:
            print(f"Skipping unsupported file: {file_name}")
            continue

        # Store each chunk in Weaviate
        for chunk in chunks:
            metadata = {
                "file_name": file_name,
                "file_path": file_path,
                "document_type": document_type  # Add document type to metadata
            }
            document = Document(page_content=chunk.page_content, metadata=metadata)
            # if document_type == "static":
            #     static_db.add_documents([document])
            # else:
            #     dynamic_db.add_documents([document])
            log_db.add_documents([document])
        print(f"Documents embedded as {document_type}.")

if __name__ == "__main__":
    # Folder containing the documents
    static_folder_path = "C:/Users/nxa24481/Downloads/ganesh/AI/docs/dynamic-2"
    aosp_path = "C:/Users/nxa24481/Downloads/interfaces-756b1b3d88a85162efe4b0a2d369e9b26d1b4b5e-security.tar/interfaces-756b1b3d88a85162efe4b0a2d369e9b26d1b4b5e-security"
    # dynamic_folder_path = "C:/Users/nxa24481/Downloads/ganesh/AI/docs/dynamic"
    
    # # Preprocess and store documents in Weaviate
    process_folder(static_folder_path, document_type="static")
    # # process_folder(dynamic_folder_path, document_type="dynamic")

    # # Query the LLM using the RetrievalQA chain
    query = "compare the both DPAS ROW EOS V2_2025-03-11_14-11-38_HTML.html file and logDataBaseIndex.html file and give me all the tests which are new , had different verdicts, which are missing in which file, the differences should be clearly highlighted."
    print("Invoked")
    response = handle_query(query)

    # # # Print the response in a nicely formatted way
    console = Console()
    console.print("[bold green]LLM Response:[/bold green]")
    console.print(Markdown(response))

    # query = "what is delete command"
    # print("Invoked")
    # response = handle_query(query)

    # # Print the response in a nicely formatted way
    # console = Console()
    # console.print("[bold green]LLM Response:[/bold green]")
    # console.print(Markdown(response))

    # query = "which setup is connected to NRW93754"
    # print("Invoked")
    # response = handle_query(query)

    # # Print the response in a nicely formatted way
    # console = Console()
    # console.print("[bold green]LLM Response:[/bold green]")
    # console.print(Markdown(response))


    # query = "what is install command"
    # print("Invoked")
    # response = handle_query(query)

    # # Print the response in a nicely formatted way
    # console = Console()
    # console.print("[bold green]LLM Response:[/bold green]")
    # console.print(Markdown(response))

