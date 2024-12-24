import os
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_cohere import CohereEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DataFrameLoader
from sqlalchemy import create_engine
import pandas as pd

load_dotenv()

def setup_qa_system():
    # Create SQLAlchemy engine
    database_url = os.getenv('DATABASE_URL')
    engine = create_engine(database_url)
    
    # Modified query to remove category
    query = """
    SELECT 
        title || ': ' || content as text_content
    FROM knowledge_base
    """
    df = pd.read_sql(query, engine)
    
    # Prepare documents
    loader = DataFrameLoader(df, page_content_column="text_content")
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )
    chunks = text_splitter.split_documents(docs)
    
    # Setup embeddings
    embeddings = CohereEmbeddings(
        cohere_api_key=os.getenv('COHERE_API_KEY'),
        model="embed-english-v3.0"
    )
    
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    # Setup retriever and LLM
    retriever = vector_store.as_retriever(
        search_kwargs={"k": 3}
    )
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",  # Using Gemini 1.5 Flash model
        google_api_key=os.getenv('GOOGLE_API_KEY'),
        temperature=0
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=retriever,
        return_source_documents=True
    )
    
    return qa_chain

if __name__ == '__main__':
    try:
        print("Initializing QA System...")
        qa_chain = setup_qa_system()
        print("QA System is ready! Type 'exit' to quit.")
        
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