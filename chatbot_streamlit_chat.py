# AI Chatbot using ollama that shows chat history with streamlit_chat module
from langchain_ollama import ChatOllama
from langchain.schema import SystemMessage, HumanMessage, AIMessage
import streamlit as st
from streamlit_chat import message
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

MODEL = "llama3.2"

def main():
    st.set_page_config(page_title="My Chatbot", page_icon="🤖")
    st.header("My Chatbot 🤖")

    # Checks if previous "messages" are in session_state, otherwise "messages" are remembered for the current conversation
    if "messages" not in st.session_state:    
        st.session_state.messages = [SystemMessage(content="Hello! How can I help you today?")]

    # Ollama llm setup
    llm = ChatOllama(model=MODEL)
    
    # Take user input
    user_input = st.chat_input("Talk to the chatbot")
    if user_input:
        logging.info("Message sent...")

        # Sends user input to llm model and returns content
        st.session_state.messages.append(HumanMessage(content=user_input))
        logging.info("Processing message...")
        response = llm.invoke(st.session_state.messages)

        # Appends all messages to "messages" session_state
        st.session_state.messages.append(AIMessage(content=response.content))
        logging.info("Response sent...")

    # Sets "messages" in session_state to empty list
    messages = st.session_state.get("messages", [])

    # Keeps previous messages for chat history
    for i, msg in enumerate(messages):
        if i % 2 == 0:
            message(msg.content, is_user=False, key=str(i) + "_ai")
        else:
            message(msg.content, is_user=True, key=str(i) + "_user")



if __name__ == '__main__':
    main()