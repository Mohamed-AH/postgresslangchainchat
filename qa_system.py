import os
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_pinecone import PineconeVectorStore
from langchain_cohere import CohereEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import DataFrameLoader
from sqlalchemy import create_engine
import pandas as pd
from pinecone import Pinecone, ServerlessSpec
import time

load_dotenv()

# Check all environment variables at once
required_vars = ['DATABASE_URL', 'COHERE_API_KEY', 'GOOGLE_API_KEY', 'PINECONE_API_KEY']
missing_vars = [var for var in required_vars if not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Missing environment variables: {', '.join(missing_vars)}")

def initialize_pinecone():
    """Initialize Pinecone client and ensure index exists"""
    print("Initializing Pinecone...")
    pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
    
    # Index configuration
    index_name = "qa-knowledge-base"
    dimension = 1024  # Dimension for Cohere embed-english-v3.0
    metric = "cosine"
    
    # Check if index exists
    if index_name not in pc.list_indexes().names():
        print(f"Creating new index: {index_name}")
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric=metric,
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'  # Only supported region for free tier
            )
        )
        print(f"Serverless index '{index_name}' created successfully in us-east-1 (AWS).")
        
        # Wait for index to be ready
        while not pc.describe_index(index_name).status['ready']:
            print("Waiting for index to be ready...")
            time.sleep(1)
    else:
        print(f"Index '{index_name}' already exists.")
        
    # Print index description
    index_description = pc.describe_index(index_name)
    print(f"Index description: {index_description}")
            
    return pc, index_name

def is_index_empty(index):
    stats = index.describe_index_stats()
    return stats.total_vector_count == 0

def setup_qa_system():
    # Initialize Pinecone and get index
    pc, index_name = initialize_pinecone()
    index = pc.Index(index_name)

    # Setup embeddings
    print("Setting up embeddings...")
    embeddings = CohereEmbeddings(
        cohere_api_key=os.getenv('COHERE_API_KEY'),
        model="embed-english-v3.0"
    )

    # Initialize vector store with Pinecone
    print("Initializing vector store...")
    vector_store = PineconeVectorStore(
        index=index,
        embedding=embeddings,
        text_key="text"
    )

    # Only load and add documents if the index is empty
    if is_index_empty(index):
        print("Vector store is empty. Loading and adding documents...")
        engine = create_engine(os.getenv('DATABASE_URL'))
        query = "SELECT title || ': ' || content as text_content FROM knowledge_base"
        with engine.connect() as connection:
            for chunk in pd.read_sql(query, connection, chunksize=1000):
                docs = DataFrameLoader(chunk, page_content_column="text_content").load()
                vector_store.add_documents(docs)
    else:
        print("Vector store already contains documents. Skipping document addition.")

    # Setup retriever and LLM
    print("Setting up retriever and LLM...")
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=os.getenv('GOOGLE_API_KEY'),
        temperature=0
    )

    return RetrievalQA.from_chain_type(
        llm,
        retriever=retriever,
        return_source_documents=True
    )

if __name__ == '__main__':
    try:
        print("Starting QA System setup...")
        qa_chain = setup_qa_system()
        print("\nQA System is ready! Type 'exit' to quit.")
        
        while True:
            question = input('\nWhat would you like to know? ')
            if question.lower() == 'exit':
                break
            
            try:
                answer = qa_chain.invoke(question)
                print('\nAnswer:')
                print(answer['result'])
                
                print('\nSources:')
                for doc in answer['source_documents']:
                    print(f"- {doc.page_content[:200]}...")
                    
            except Exception as e:
                print(f"Error during question processing: {str(e)}")
                
    except Exception as e:
        print(f"Error during setup: {str(e)}")
