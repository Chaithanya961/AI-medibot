from dotenv import load_dotenv
import os
from src.helper import load_pdf_file, filter_to_minimal_docs, text_split, download_hugging_face_embeddings
from pinecone import Pinecone, ServerlessSpec 
from langchain_pinecone import PineconeVectorStore

load_dotenv()

# 1. Load All Keys
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
# Added Groq Key to environment
GROQ_API_KEY = os.environ.get('GROQ_API_KEY') 

# 2. Set Environment Variables for LangChain
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# 3. Data Processing
extracted_data = load_pdf_file(data='data/')
filter_data = filter_to_minimal_docs(extracted_data)
text_chunks = text_split(filter_data)

# 4. Initialize Embeddings (Sentence-Transformers)
embeddings = download_hugging_face_embeddings()

# 5. Initialize Pinecone Client
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "medical-chatbot"

# 6. Create Index if it doesn't exist
# IMPORTANT: Dimension must be 384 for 'all-MiniLM-L6-v2'
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384, 
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

# 7. Ingest Documents into Pinecone
# This will embed your chunks and upload them
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings, 
)

print("Ingestion complete. Your Pinecone index is ready for Groq!")