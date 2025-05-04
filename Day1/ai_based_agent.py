import streamlit as st
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

llm = OllamaLLM(model="llama3.2") # you can use 'mistral' or 'Gemma3:4b'

if "chat_history" not in st.session_state:
    st.session_state.chat_history = ChatMessageHistory()
    
prompt = PromptTemplate(
    input_variables=["chat_history", "question"],
    template="Previous conversation: {chat_history}\nUser: {question}\nAI:"
)

def run_chain(question):
    chat_history_text = "\n".join([f"{message.type.capitalize()}: {message.content}" for message in st.session_state.chat_history.messages])
    
    response = llm.invoke(prompt.format(chat_history = chat_history_text, question=question))
    
    st.session_state.chat_history.add_user_message(question)
    st.session_state.chat_history.add_ai_message(response)
    
    return response

st.title("👦🏻 Welcome AI Agent Chatbot!🪄")
st.write("👁 Ask me Anything")

user_input = st.text_input("🙋 Your Question: ")
if user_input:
    response = run_chain(user_input)
    st.write(f" **You:** {user_input}")
    st.write(f"**AI:** {response}")
    
st.subheader("📜 Chat History")
for message in st.session_state.chat_history.messages:
    st.write(f"**{message.type.capitalize()}**: {message.content}")
        






# BASIC AI AGENT WITH MEMORY

# from langchain_community.chat_message_histories import ChatMessageHistory
# from langchain_core.prompts import PromptTemplate
# from langchain_ollama import OllamaLLM

# llm = OllamaLLM(model="llama3.2")

# chat_history = ChatMessageHistory()
# prompt = PromptTemplate(
#     input_variables=["chat_history", "question"],
#     template="Previous conversation: {chat_history}\nUser: {question}\nAI:"
# )

# def run_chain(question):
#     chat_history_text = "\n".join([f"{message.type.capitalize()}: {message.content}" for message in chat_history.messages])
    
#     response = llm.invoke(prompt.format(chat_history=chat_history_text, question=question))
    
#     chat_history.add_user_message(question)
#     chat_history.add_ai_message(response)
    
#     return response

# print("\n Welcome to AI Agent with Memory! Ask me anything")
# print("Type 'exit' to quit.")

# while True:
#     user_input = input("\nYou: ")
#     if user_input.lower() == 'exit':
#         print("Goodbye!")
#         break
#     ai_response = run_chain(user_input)
#     print(f"AI: {ai_response}")


# BASIC AI AGENT WITHOUT MEMORY

# from langchain_ollama import OllamaLLM

# llm = OllamaLLM(model="llama3.2")

# print("\n Welcome to AI Agent! Ask me anything.")

# while True:
#     question = input("Your question (type 'exit' to quit): ")
#     if question.lower() == 'exit':
#         print("Goodbye!")
#         break
#     response = llm.invoke(question)
#     print(f"AI response: {response}")
