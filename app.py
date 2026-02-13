from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
# CHANGED: Import Groq instead of OpenAI
from langchain_groq import ChatGroq 
# from langchain.chains import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from newsapi import NewsApiClient

app = Flask(__name__)

load_dotenv()
NEWS_API_KEY = os.environ.get('NEWS_API_KEY', "") 
newsapi = NewsApiClient(api_key=NEWS_API_KEY) #
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
GROQ_API_KEY = os.environ.get('GROQ_API_KEY') # Added Groq Key
print(f"Is News API Key loaded? {'Yes' if os.environ.get('NEWS_API_KEY') else 'No'}")
print(f"NEWS_API_KEY: {os.environ.get('NEWS_API_KEY')}")
# Ensure keys are in environment for LangChain
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

embeddings = download_hugging_face_embeddings()

index_name = "medical-chatbot" 
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})

# CHANGED: Use ChatGroq instead of ChatOpenAI
chatModel = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.4 # Slightly lower for more factual medical responses
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

@app.route("/")
def index():
    articles = []
    if NEWS_API_KEY:
        try:
            # Fetch Health News
            health_news = newsapi.get_top_headlines(category='health', language='en', country='us')
            # Fetch Science News (often contains medical breakthroughs)
            science_news = newsapi.get_top_headlines(category='science', language='en', country='us')
            
            # Combine and limit to 6 articles
            articles = (health_news.get('articles', []) + science_news.get('articles', []))[:30]
        except Exception as e:
            print(f"NewsAPI Error: {e}")
    
    return render_template('chat.html', articles=articles)

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    print(f"User Input: {msg}")
    
    # The RAG chain logic stays exactly the same
    response = rag_chain.invoke({"input": msg})
    print("Response : ", response["answer"])
    return str(response["answer"])

@app.route("/health-centers")
def health_centers():
    return render_template('health_centers.html')

@app.route("/about-ai")
def about_ai():
    return render_template('about_ai.html')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)