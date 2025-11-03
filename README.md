# Local Vector Database RAG Pipeline

A comprehensive Retrieval Augmented Generation (RAG) system using Weaviate vector database, LangChain, and multiple AI models for document processing and intelligent querying.

## 🚀 Features

- **Multi-Document Processing**: Support for PDF, Excel, and HTML documents
- **Vector Database**: Local Weaviate instance for efficient semantic search
- **Dual Document Collections**: Separate handling for static and dynamic documents
- **Smart Query Routing**: Automatic routing based on keyword detection
- **Multiple AI Models**: Integration with Groq and Ollama LLMs
- **Web Interface**: Flask API with CORS support
- **Rich Output**: Markdown-formatted responses with follow-up questions
- **Advanced Document Parsing**: LlamaParse integration for complex documents

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Documents     │    │   Vector Store   │    │   AI Models     │
│   (PDF/Excel/   │───▶│   (Weaviate)     │───▶│   (Groq/Ollama) │
│    HTML)        │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │   Flask API      │
                       │   (REST)         │
                       └──────────────────┘
```

## 📋 Prerequisites

- Python 3.8+
- Docker & Docker Compose
- 4GB+ RAM (recommended)

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd Local-Vector-DB-RAG-Pipeline
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Up Environment Variables

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_groq_api_key_here
LLAMA_PARSE_API_KEY=your_llama_parse_api_key_here
```

### 4. Start Weaviate Database

```bash
docker compose up -d
```

This will start Weaviate on:
- HTTP: `http://localhost:8080`
- gRPC: `localhost:50051`

## 🚦 Quick Start

### 1. Document Processing

```python
# Process documents and store in vector database
python app.py
```

### 2. Start Flask API Server

```python
# Start the REST API server
python app4.py
```

The API will be available at `http://localhost:5000`

### 3. Query the System

**Using cURL:**
```bash
curl -X POST http://localhost:5000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Explain machine learning concepts"}'
```

**Using Python:**
```python
import requests

response = requests.post('http://localhost:5000/query', 
                        json={"query": "What hardware is used in setup1?"})
print(response.json()["response"])
```

## 📁 File Structure

```
├── app.py                          # Main document processing script
├── app2_textualdescription.py      # Excel to text conversion
├── app3.py                         # Basic RAG implementation
├── app4.py                         # Flask API server
├── app5_llamaparse.py              # LlamaParse integration
├── docker-compose.yml              # Weaviate container setup
├── requirements.txt                # Python dependencies
├── frontend/                       # React frontend (optional)
│   ├── src/
│   │   ├── components/
│   │   │   ├── QAAgent.js
│   │   │   ├── QuizAgent.js
│   │   │   └── Dashboard.js
│   │   └── services/
│   │       └── api.js
└── README.md
```

## 🔧 Configuration

### Document Collections

The system uses three main collections in Weaviate:

1. **StaticDocuments**: For general knowledge and explanatory content
2. **DynamicDocuments**: For fact-based, specific information
3. **abcDocuments**: For HTML-processed documents

### Keywords for Smart Routing

The system automatically routes queries based on predefined keywords:

```python
FACT_BASED_KEYWORDS = [
    "lab location", "pc assigned", "ETA", "vacation calendar",
    "A2-65 LAB", "A2-66 LAB", "jenkins automation", "BLR-ROW",
    "hardware", "setup", "SN300", "SN330", "vulcan"
]
```

### Supported Models

- **Groq**: `llama-3.3-70b-versatile`, `deepseek-r1-distill-llama-70b`
- **Ollama**: `llama2` (local)
- **Embeddings**: `sentence-transformers/all-mpnet-base-v2`

## 📖 API Documentation

### Endpoints

#### POST /query
Query the RAG system with natural language questions.

**Request:**
```json
{
    "query": "Your question here"
}
```

**Response:**
```json
{
    "response": "Markdown formatted answer with follow-up questions"
}
```

**Example Queries:**

- **General Questions**: "Explain software testing methodologies"
- **Fact-based Questions**: "Which PC is assigned to A2-65 LAB?"
- **Technical Questions**: "What hardware is used in setup1?"

## 🧪 Testing

### Using Postman

1. **Method**: POST
2. **URL**: `http://localhost:5000/query`
3. **Headers**: `Content-Type: application/json`
4. **Body**:
```json
{
    "query": "Test question"
}
```

### Using Frontend

Start the React frontend:

```bash
cd frontend
npm install
npm start
```

Access the web interface at `http://localhost:3000`

## 📊 Document Processing Workflow

### 1. PDF Processing
- Extracts text using PyPDFLoader
- Splits into chunks (1000 chars, 200 overlap)
- Generates embeddings using SentenceTransformers
- Stores in Weaviate collections

### 2. Excel Processing
- Reads data using pandas
- Converts rows to textual descriptions using LLM
- Processes as natural language text
- Embeds in vector database

### 3. HTML Processing
- Parses using UnstructuredHTMLLoader
- Extracts clean text content
- Chunks and embeds similar to PDFs

## 🔍 Advanced Features

### LlamaParse Integration

For complex document parsing:

```python
parser = LlamaParse(
    api_key="your_api_key",
    parsing_instruction="Custom parsing instructions",
    result_type="markdown"
)
```

### Custom Embeddings

The system uses a custom wrapper for SentenceTransformers:

```python
class SentenceTransformerEmbeddings:
    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()
    
    def embed_query(self, text):
        return self.model.encode(text).tolist()
```

## 🐛 Troubleshooting

### Common Issues

1. **Weaviate Connection Error**
   ```bash
   # Check if Weaviate is running
   docker ps
   # Restart if needed
   docker compose restart
   ```

2. **Memory Issues**
   ```bash
   # Monitor Docker memory usage
   docker stats
   ```

3. **API Key Issues**
   ```bash
   # Verify environment variables
   echo $GROQ_API_KEY
   ```

### Performance Optimization

- **Chunk Size**: Adjust `chunk_size` based on document complexity
- **Retrieval Count**: Modify `k` parameter for more/fewer context chunks
- **Model Selection**: Choose appropriate models based on speed vs accuracy needs

## 📈 Monitoring

### Database Status

```python
# Check Weaviate collections
import weaviate
client = weaviate.connect_to_local()
print(client.collections.list_all())
```

### Query Performance

Monitor response times and adjust retrieval parameters accordingly.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📄 License

This project is licensed under the MIT License.

## 🆘 Support

For issues and questions:
- Create an issue in the GitHub repository
- Check the troubleshooting section
- Review the API documentation

## 🔮 Future Enhancements

- [ ] Multi-language support
- [ ] Real-time document updates
- [ ] Advanced query analytics
- [ ] Custom model fine-tuning
- [ ] Distributed vector storage
- [ ] Authentication and user management

---

**Built with ❤️ using LangChain, Weaviate, and modern AI technologies**
