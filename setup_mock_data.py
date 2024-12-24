import re
from dotenv import load_dotenv
import os
import psycopg2

# Load environment variables
load_dotenv()
DATABASE_URL = os.getenv('DATABASE_URL')

def extract_sections(md_text):
    sections = []
    current_title = None
    current_content = []
    
    # Split text into lines and process
    lines = md_text.split('\n')
    for line in lines:
        # Skip image references
        if line.strip().startswith('!['):
            continue
            
        # Check for headings
        if line.strip().startswith('#'):
            # Save previous section if exists
            if current_title and current_content:
                content = ' '.join(current_content).strip()
                if content:
                    sections.append({
                        'title': current_title,
                        'content': content
                    })
            # Start new section
            current_title = line.strip('#').strip()
            current_content = []
        else:
            # Add content lines
            if line.strip():
                current_content.append(line.strip())
    
    # Add the last section
    if current_title and current_content:
        content = ' '.join(current_content).strip()
        if content:
            sections.append({
                'title': current_title,
                'content': content
            })
    
    return sections

def setup_database():
    try:
        print(f"Connecting to database...")
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        
        print("Creating knowledge_base table...")
        cur.execute("""
        DROP TABLE IF EXISTS knowledge_base;
        CREATE TABLE knowledge_base (
            id SERIAL PRIMARY KEY,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)
        
        return conn, cur
    except Exception as e:
        print(f"Database setup error: {str(e)}")
        raise

def insert_sections(conn, cur, sections):
    try:
        print(f"Inserting {len(sections)} sections...")
        for section in sections:
            print(f"Inserting: {section['title']}")
            cur.execute(
                "INSERT INTO knowledge_base (title, content) VALUES (%s, %s)",
                (section['title'], section['content'])
            )
        
        conn.commit()
        print("All sections inserted successfully")
        
        # Verify data
        cur.execute("SELECT title FROM knowledge_base")
        print("\nCreated articles:")
        for row in cur.fetchall():
            print(f"- {row[0]}")
    
    except Exception as e:
        print(f"Data insertion error: {str(e)}")
        raise
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()

if __name__ == "__main__":
    try:
        print("Starting data import process...")
        
        # Read the markdown content from file
        with open('content.md', 'r', encoding='utf-8') as file:
            md_content = file.read()
        
        # Extract sections
        sections = extract_sections(md_content)
        
        # Setup database and insert sections
        conn, cur = setup_database()
        insert_sections(conn, cur, sections)
        
        print("\nMock data setup completed successfully!")
        
    except FileNotFoundError:
        print("Error: Please save the markdown content in 'content.md'")
    except Exception as e:
        print(f"Error: {str(e)}")