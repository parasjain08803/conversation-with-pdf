# Chat with PDF

A powerful Streamlit application that allows you to upload PDF documents and chat with them using AI-powered RAG (Retrieval-Augmented Generation).

## Features

- 📄 **PDF Upload & Processing**: Upload any PDF file for intelligent conversation
- 🧠 **AI-Powered Chat**: Uses Groq's Qwen model for intelligent responses
- 🔍 **Smart Retrieval**: Advanced RAG with history-aware question rewriting
- ⚙️ **Customizable Settings**: Adjust chunk size and overlap for better results
- 💾 **Chat History**: Persistent conversation memory during session
- 📊 **Document Insights**: View document statistics and processing info
- 📥 **Export Chat**: Download your conversation history
- 🔄 **Error Handling**: Robust error handling with user-friendly messages

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd "Chat with pdf"
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

## Usage

1. Run the application:
```bash
streamlit run app.py
```

2. Open your browser to the provided URL (usually `http://localhost:8501`)

3. Upload a PDF file using the file uploader

4. Start chatting with your document!

## Configuration

### Environment Variables

- `GROQ_API_KEY`: Required - Your Groq API key (get one from [Groq Console](https://console.groq.com/))
- `LANGCHAIN_API_KEY`: Optional - For LangSmith tracing
- `LANGCHAIN_TRACING_V2`: Optional - Enable LangSmith tracing
- `LANGCHAIN_PROJECT`: Optional - LangSmith project name

### Settings

Use the sidebar to adjust:
- **Chunk Size**: Controls text chunking size (500-2000 characters)
- **Chunk Overlap**: Overlap between chunks (50-500 characters)

## Architecture

The application uses:
- **LangChain**: For RAG pipeline and LLM integration
- **ChromaDB**: For vector storage and similarity search
- **HuggingFace Embeddings**: For text embeddings
- **Groq**: For fast LLM inference
- **Streamlit**: For the web interface

## Error Handling

The application includes comprehensive error handling for:
- PDF processing failures
- API connection issues
- Invalid file formats
- Memory management

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is open source and available under the MIT License.