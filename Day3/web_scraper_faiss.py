import requests
from bs4 import BeautifulSoup
from langchain_ollama import OllamaLLM
import streamlit as st
import faiss
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
import numpy as np

llm = OllamaLLM(model= "llama3.2")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

index = faiss.IndexFlatL2(384)
vectorstore = {}

def scrape_website(url):
    try:
        st.write(f"Scarping websits: {url}")
        headers = {"User-Agent": "Mozialla/5.0"}
        response = requests.get(url, headers=headers)
        
        if response.status_code != 200:
            return f"Failed to fetch {url}"
        
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        text = " ".join([p.get_text() for p in paragraphs])
        
        return text[:5000]
    except Exception as e:
        return f"Error: {str(e)}"
    
def store_in_faiss(text, url):
    global index, vectorstore
    st.write("Storing content in FAISS....")
    
    
    splitter = CharacterTextSplitter(chunk_size = 500, chunk_overlap = 100)
    texts = splitter.split_text(text)
    
    vectors = embeddings.embed_documents(texts)
    vectors = np.array(vectors, dtype=np.float32)
    
    index.add(vectors)
    vectorstore[len(vectorstore)] = (url, texts)
    
    return "Data stored successfully in FAISS!"

def retrieve_from_faiss(query):
    global index, vectorstore
    
    query_vector = np.array(embeddings.embed_query(query), dtype=np.float32).reshape(1, -1)
    D,I = index.search(query_vector, k=2)
    
    context = ""
    for idx in I[0]:
        if idx in vectorstore:
            context += " ".join(vectorstore[idx][1]) + "\n\n"
            
    if not context:
        return "No relevant data found!"
    
    return llm.invoke(f"Based on the following context, answer the question: \n\n{context}\nQuestion: {query}\nAnswer:")


st.title("AI-Powered Web Scraper with FAISS!")
st.write("Enter a URL to scrape, store in FAISS, and retrieve information.")

url = st.text_input("Enter the Website URL: ")
if url:
    content = scrape_website(url)
    if "Failed" in content or "Error" in content:
        st.write(content)
    else:
        store_message = store_in_faiss(content, url)
        st.write(store_message)
        
query = st.text_input("Ask a question about the stored content: ")
if query:
    answer = retrieve_from_faiss(query)
    st.subheader("AI's Answer")
    st.write(answer)