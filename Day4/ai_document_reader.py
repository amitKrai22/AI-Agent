import streamlit as st
import PyPDF2
import numpy as np
import faiss
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter

llm = OllamaLLM(model="llama3.2")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

index = faiss.IndexFlatL2(384)
vectorstore = {}
summary_text = ""

def extract_text(uploaded_file):
    if uploaded_file is not None:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    else:
        return ""


def store_in_faiss(text, filename):
    global index, vextorestore
    st.write("Storing {filename} in FAISS")
    
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    texts = splitter.split_text(text)
    
    vectors = embeddings.embed_documents(texts)
    vectors = np.array(vectors, dtype=np.float32)
    
    index.add(vectors)
    vectorstore[len(vectorstore)] = (filename, texts)
    
    return "Document stored successfully!"

def summarize_text(text):
    global summary_text
    st.write("Summarizing the document...")
    summary_text = llm.invoke(f"Summarize the following text: \n\n{text[:3000]}")
    return summary_text
    
def retrieve_and_answer(query):
    global index, vrctorestore
    st.write("Retrieving data from FAISS..")
    
    
    query_vector = np.array(embeddings.embed_query(query), dtype=np.float32).reshape(1, -1)
    D,I = index.search(query_vector, k=2)
    
    context = ""
    for idx in I[0]:
        if idx in vectorstore:
            context += " ".join(vectorstore[idx][1]) + "\n\n"
            
    if not context:
        return "No relevant data found!"
    
    return llm.invoke(f"Based on the following context, answer the question: \n\n{context}\nQuestion: {query}\nAnswer:")

def download_file():
    if summary_text:
        st.download_button(
            label="Download Summary",
            data=summary_text,
            file_name="AI_summary.txt",
            mime="text/plain"
        )

st.title("AI Document Reader & Q&A bot")
st.write("Upload a PDF and get AI-generated summary ask questions about its content.")

uploaded_file = st.file_uploader("Upload PDF", type="pdf")
if uploaded_file:
    text = extract_text(uploaded_file)
    store_message = store_in_faiss(text, uploaded_file.name)
    st.write(store_message)
    
    summary = summarize_text(text)
    st.subheader("AI-Summary:")
    st.write(summary)
    
    download_file()
    
query = st.text_input("Ask a question about the document: ")
if query:
    answer = retrieve_and_answer(query)
    st.subheader("AI's Answer:")
    st.write(answer)