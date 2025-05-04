import streamlit as st 
import speech_recognition as sr
import pyttsx3
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

llm = OllamaLLM(model="llama3.2")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = ChatMessageHistory()

engine = pyttsx3.init()
engine.setProperty('rate', 150) # Speed of speech

recognizer = sr.Recognizer()

def speak(text):
    engine.say(text)
    engine.runAndWait()

def listen():
    with sr.Microphone() as source:
        st.write("Listening...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
    try:
        query = recognizer.recognize_google(audio)
        st.write(f"You said: {query}")
        return query.lower()
    except sr.UnknownValueError:
        st.write("Sorry, I did not understand that. Try again!")
        return ""
    except sr.RequestError:
        st.write("Sorry, there was an error with the speech recognition service.")
        return ""
    
prompt = PromptTemplate(
    input_variables=["chat_history", "question"],
    template="Previous conversation: {chat_history}\nUser: {question}\nAI:"
)

def run_chain(question):
    chat_history_text = "\n".join([f"{message.type.capitalize()}: {message.content}" for message in st.session_state.chat_history.messages])
    
    response = llm.invoke(prompt.format(chat_history=chat_history_text, question=question))
    
    st.session_state.chat_history.add_user_message(question)
    st.session_state.chat_history.add_ai_message(response)
    
    return response

st.title("👦🏻 Welcome AI Voice Assistant!🪄")
st.write("👁 click the button below to speak with AI Assistant!")


if st.button("🎤 Start Listening"):
    user_query = listen() 
    if user_query:
        ai_response = run_chain(user_query)
        st.write(f"**You:** {user_query}")
        st.write(f"**AI:** {ai_response}")
        speak(ai_response)
        
st.subheader("📜 Chat History")
for message in st.session_state.chat_history.messages:
    st.write(f"**{message.type.capitalize()}**: {message.content}")
    