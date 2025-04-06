# Chatbot using ollama
from langchain_ollama import ChatOllama
from langchain.schema import SystemMessage, HumanMessage, AIMessage
import streamlit as st
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

MODEL = "llama3.2"

def main():
    st.set_page_config(page_title="My Chatbot", page_icon="🤖")
    st.header("My Chatbot 🤖")

    # Aligns chatbot text and icon to right of screen
    st.markdown("""
        <style>
            .st-emotion-cache-4oy321 {
                flex-direction: row-reverse;
                text-align: right;
            }
        </style>
    """,
    unsafe_allow_html=True)

    # Checks if previous "messages" are in session_state, otherwise "messages" are remembered for the current conversation
    if "messages" not in st.session_state:    
        st.session_state.messages = [SystemMessage(content="Hello! How can I help you today?")]

    # Ollama llm setup
    llm = ChatOllama(model=MODEL)

    chatbot_message = st.chat_message("assistant")
    chatbot_message.write("Hello user!")
    
    # Take user input
    user_input = st.chat_input("Talk to the chatbot")
    if user_input:
        user_message = st.chat_message("human")
        user_message.write(user_input)
        logging.info("Message sent...")
        chatbot_message = st.chat_message("assistant")

        # Sends user input to llm model and returns content
        st.session_state.messages.append(HumanMessage(content=user_input))
        logging.info("Processing message...")
        response = llm.invoke(st.session_state.messages)

        # Appends all messages to "messages" session_state
        st.session_state.messages.append(AIMessage(content=response.content))
        chatbot_message.write(response.content)
        logging.info("Response sent...")

if __name__ == '__main__':
    main()