import speech_recognition as sr
import pyttsx3
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
import os
import contextlib

@contextlib.contextmanager
def suppress_alsa_errors():
    with open(os.devnull, 'w') as devnull:
        old_stderr = os.dup(2)
        os.dup2(devnull.fileno(), 2)
        try:
            yield
        finally:
            os.dup2(old_stderr, 2)

llm = OllamaLLM(model="llama3.2") # you can use mistral, or any other model from the Ollama library

chat_history = ChatMessageHistory()

engine = pyttsx3.init()
engine.setProperty('rate', 150) # Speed of speech
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)

recognizer = sr.Recognizer()

def speak(text):
    with suppress_alsa_errors():
        engine.say(text)
        engine.runAndWait()

def listen():
    with suppress_alsa_errors():
        with sr.Microphone() as source:
            print("Listening...")
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source)
    try:
        with suppress_alsa_errors():
            query = recognizer.recognize_google(audio)
        print(f"You said: {query}")
        return query.lower()
    except sr.UnknownValueError:
        print("Sorry, I did not understand that. Try again!")
        return ""
    except sr.RequestError:
        print("Sorry, there was an error with the speech recognition service.")
        return ""
    
prompt = PromptTemplate(
    input_variables=["chat_history", "question"],
    template="Previous conversation: {chat_history}\nUser: {question}\nAI:"
)

def run_chain(question):
    chat_history_text = "\n".join([f"{message.type.capitalize()}: {message.content}" for message in chat_history.messages])
    
    response = llm.invoke(prompt.format(chat_history=chat_history_text, question=question))
    
    chat_history.add_user_message(question)
    chat_history.add_ai_message(response)
    
    return response

speak("Hello! I am your AI voice assistant. How can I help you today?")

while True:
    query = listen()
    if "exit" in query or "stop" in query:
        speak("Goodbye! Have a great day!")
        break
    if query:
        response = run_chain(query)
        print(f"AI: {response}")
        speak(response)