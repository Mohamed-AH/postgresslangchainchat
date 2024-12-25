# PostgreSQL Source Chat with LangChain

A proof of concept demonstrating how to build a chatbot that answers questions using content stored in a PostgreSQL database, powered by LangChain and Gemini.

## What is this?

This POC shows how to:
- Store content in PostgreSQL
- Generate embeddings using Cohere
- Use Pinecone for vector storage and search
- Generate answers using Google's Gemini model
- Chain everything together with LangChain

## Important Notes

⚠️ This is a **proof of concept** with several limitations:
- Basic error handling
- Simplified document processing
- No input validation
- Limited security measures
- No tests
- Not suitable for production use without further development

## Getting Started

### 1. Install Dependencies
```bash
pip install langchain langchain-google-genai langchain-cohere langchain-pinecone python-dotenv psycopg2-binary pandas sqlalchemy pinecone-client
```

### 2. Set Up Environment
Create a `.env` file:
```env
DATABASE_URL=postgresql://user:password@host:port/dbname
COHERE_API_KEY=your_key
GOOGLE_API_KEY=your_key
PINECONE_API_KEY=your_key
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

- `setup_mock_data.py`: Loads content into PostgreSQL
- `qa_system.py`: QA system implementation with Pinecone integration
- `.env.example`: Environment variables template

## Example Usage

```bash
$ python qa_system.py
Starting QA System setup...
[System initialization messages]
QA System is ready! Type 'exit' to quit.

What would you like to know? What is a VPC?
[System provides answer with sources]
```

## Key Features

- **Quick Setup**: Simple instructions to launch your chatbot with PostgreSQL, LangChain, and Pinecone.
- **Efficient Data Loading**: Loads PostgreSQL data in chunks to optimize memory.
- **Powerful Integrations**: Uses Cohere for embeddings and Pinecone for vector storage.
- **Accurate Answers**: Utilizes Google's Gemini model for dynamic question answering.
- **User-Friendly CLI**: Easy command-line interface for interaction.
- **Customizable Architecture**: Modular design for easy experimentation.


## Learning Resources

If you're interested in building something more robust, check out:
- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)
- [RAG Best Practices](https://www.pinecone.io/learn/retrieval-augmented-generation/)
- [Pinecone Documentation](https://docs.pinecone.io/docs/overview)

## Contributing

This is a learning project. Feel free to experiment with it! Some interesting areas to explore:
- Advanced text chunking strategies
- Improved prompt engineering
- Comprehensive error handling
- Testing approaches
- Fine-tuning vector search parameters
- Experimenting with different embedding models

## License

MIT