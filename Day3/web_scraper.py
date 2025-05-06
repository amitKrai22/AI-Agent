import requests
from bs4 import BeautifulSoup
from langchain_ollama import OllamaLLM
import streamlit as st 

llm = OllamaLLM(model="llama3.2")


def scrape_website(url):
    try:
        st.write(f"Scraping {url}")
        headers = {"User-Agent": "Mozialla/5.0"}
        response = requests.get(url, headers=headers)
        
        if response.status_code != 200:
            return f"failed to featch {url}"
        
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        text = " ".join([p.get_text() for p in paragraphs])
        print(text)
        
        return text[:2000]
    
    except Exception as e:
        return f"Error: {str(e)}"
    
def summerize_content(content):
    st.write("Summarizing_content....")
    return llm.invoke(f"Summarize the following content: \n\n{content[:1000]}")

st.title("AI-Powered Web Scraper!")
st.write("Enter a URL to scrape and summarise its content.")

url = st.text_input("Enter the URL to scrape: ")
if url:
    content = scrape_website(url)
    if "Failed" in content or "Error" in content:
        st.write(content)
    else: 
        summary = summerize_content(content)
        st.subheader("Website Summary")
        st.write(summary)
        