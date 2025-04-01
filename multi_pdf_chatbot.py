# Mutltiple PDF File Chatbot with Streamlit and Ollama
from langchain.text_splitter import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import streamlit as st
from PyPDF2 import PdfReader
import ollama, logging, chromadb
from html_template import css, bot_template, user_template

# Configure logging
logging.basicConfig(level=logging.INFO)

MODEL = "llama3.2"
EMBEDDING = "nomic-embed-text"
PERSIST_DIRECTORY = "./chroma_db"
VECTOR_STORE = "simple-rag"

def doc_loader(doc_path):
    """Loads and returns multiple documents for processing"""
    text = ""
    for pdf in doc_path:
        loader = PdfReader(pdf)
        for page in loader.pages:
            text += page.extract_text()
    logging.info("Loading documents...")
    return text

def split_chunk_doc(text):
    """Splits document into chunks and returns them"""
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    logging.info("Splitting completed!")
    return chunks


def create_vector_db(text_chunks):
    """Creates and returns vector database with ollama model"""
    ollama.pull(EMBEDDING)
    embeddings = OllamaEmbeddings(model=EMBEDDING)
    vector_db = Chroma.from_texts(
        texts=text_chunks, 
        embedding=embeddings
    )
    logging.info("Successfully added to Vector Database!")
    return vector_db


def conversation_chain(vector_db):
    """Puts chat messages into memory variable for llm to use"""
    llm = ChatOllama(model=MODEL)
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_db.as_retriever(),
        memory=memory
    )
    logging.info("Conversation chain created!")
    return conversation_chain


def handle_input(user_question):
    """Takes"""
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    st.set_page_config(page_title="Multiple PDF Chatbot 📄", page_icon="./imgs/favicon.png")
    st.write(css, unsafe_allow_html=True)
    st.header("PDF Chatbot :books:")
    
    # Fixes "ValueError: Could not connect to tenant default_tenant. Are you sure it exists?"
    chromadb.api.client.SharedSystemClient.clear_system_cache()

    # Checks if "conversation" and "chat_history" are in session_state, otherwise set to None
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_input(user_question)

    pdf_docs = st.file_uploader(
        "Upload your PDFs Here Then Click on Submit", 
        type="pdf", 
        accept_multiple_files=True
    )
    if st.button("Process"):
        with st.spinner("Processing"):
            try:
                # Get text from PDF
                text = doc_loader(pdf_docs)

                # Split into chunks
                chunks = split_chunk_doc(text)

                # Create chroma db with chunks
                vector_db = create_vector_db(chunks)

                # Creates conversation chain stores in session_state
                logging.info("Creating chain...")
                st.session_state.conversation = conversation_chain(vector_db)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

if __name__ == '__main__':
    main()