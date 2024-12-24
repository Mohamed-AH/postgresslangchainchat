# PostgreSQL Source Chat with LangChain

A proof of concept demonstrating how to build a chatbot that answers questions using content stored in a PostgreSQL database, powered by LangChain and Gemini.

## What is this?

This POC shows how to:
- Store content in PostgreSQL
- Generate embeddings using Cohere
- Use FAISS for vector search
- Generate answers using Google's Gemini model
- Chain everything together with LangChain

## Important Notes

⚠️ This is a **proof of concept** with several limitations:
- No error handling for API failures
- In-memory FAISS index (resets on restart)
- Basic text chunking that might split content inappropriately
- No input validation
- No security measures
- No tests
- Not suitable for production use

## Getting Started

### 1. Install Dependencies
```bash
pip install langchain langchain-google-genai langchain-cohere python-dotenv psycopg2-binary pandas sqlalchemy
```

### 2. Set Up Environment
Create a `.env` file:
```env
DATABASE_URL=postgresql://user:password@host:port/dbname
COHERE_API_KEY=your_key
GOOGLE_API_KEY=your_key
```

### 3. Load Sample Data
```bash
python setup_mock_data.py
```

### 4. Run the System
```bash
python qa_system.py
```

## Project Files

- `setup_mock_data.py`: Loads markdown content into PostgreSQL
- `qa_system.py`: Basic QA implementation
- `content.md`: Sample AWS networking content
- `.env.example`: Environment variables template

## Example Usage

```bash
$ python qa_system.py
QA System is ready! Type 'exit' to quit.

What would you like to know? What is a VPC?
[System provides answer with sources]
```

## Learning Resources

If you're interested in building something more robust, check out:
- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)
- [RAG Best Practices](https://www.pinecone.io/learn/retrieval-augmented-generation/)

## Contributing

This is a learning project. Feel free to experiment with it! Some interesting areas to explore:
- Better text chunking strategies
- Improved prompt engineering
- Error handling
- Testing approaches
- Alternative vector stores
- Different embedding models

## License

MIT 