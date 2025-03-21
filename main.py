# Ollama PDF chatbot with Streamlit
from langchain.text_splitter import CharacterTextSplitter
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
from PyPDF2 import PdfReader
import ollama, logging, chromadb

# Configure logging
logging.basicConfig(level=logging.INFO)

MODEL = "llama3.2"
EMBEDDING = "nomic-embed-text"
PERSIST_DIRECTORY = "./chroma_db"
VECTOR_STORE = "simple-rag"

def create_vector_db():
    """Creates and returns vector database with ollama model"""
    # Pull the embedding model if not already available
    ollama.pull(EMBEDDING)

    embedding = OllamaEmbeddings(model=EMBEDDING)

    vector_db = Chroma(
        embedding_function=embedding,
        collection_name=VECTOR_STORE,
        persist_directory=PERSIST_DIRECTORY,
    )
    logging.info("Successfully added to Vector Database!")
    return vector_db

def ollama_retriever(vector_db, llm):
    """Returns relevant chunks of data from vector database based on user's query"""
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI language model assistant. Your task is to generate five
        different versions of the given user question to retrieve relevant documents from
        a vector database. By generating multiple perspectives on the user question, your
        goal is to help the user overcome some of the limitations of the distance-based
        similarity search. Provide these alternative questions separated by newlines.
        Original question: {question}""",
    )
    # Transforms the db into a retriever to pass questions to the llm using the QUERY_PROMPT 
    retriever = MultiQueryRetriever.from_llm(
        vector_db.as_retriever(), llm, prompt=QUERY_PROMPT
    )
    logging.info("Query Retriever successfully created!")
    return retriever

def create_chain(retriever, llm):
    """Passes retriever chunks to llm and returns an answer to the user's query"""
    # RAG prompt
    template = """Answer the question based ONLY on the following context:
    {context}
    Question: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    logging.info("RAG Chain created successfully!")
    return chain

def split_chunk_doc(documents):
    """Splits document into chunks and returns them"""
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000, 
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(documents)
    logging.info("Splitting completed!")
    return chunks

def doc_loader(doc_path):
    """Loads and returns document for processing"""
    loader = PdfReader(doc_path)
    text = ""
    for page in loader.pages:
        text += page.extract_text()
    logging.info("Loading document...")
    return text
    
def main():
    st.set_page_config(page_title="PDF Chatbot 📄", page_icon="favicon.png")
    col1, col2 = st.columns([2,10])
    with col1:
        st.image("image.png")
    with col2:
        st.title("PDF Chatbot")

    # Fixes "ValueError: Could not connect to tenant default_tenant. Are you sure it exists?"
    chromadb.api.client.SharedSystemClient.clear_system_cache()

    # Upload PDF
    pdf = st.file_uploader("Upload your PDF", type="pdf")
    
    # Gets text from file and splits it into chunks
    if pdf is not None:
        with st.spinner("Processing..."):
            try:
                text = doc_loader(pdf)
                
                # Split into chunks
                chunks = split_chunk_doc(text)

                # Create ollama embeddings
                ollama.pull(EMBEDDING)
                llm = ChatOllama(model=MODEL)
                embeddings = OllamaEmbeddings(model=EMBEDDING)

                # Create chroma db with chunks and embeddings
                store = Chroma.from_texts(chunks, embeddings)
                vector_db = create_vector_db()
                logging.info("Creating chain...")
                retriever = ollama_retriever(vector_db, llm)
                
                # Take user input
                user_input = st.text_input("Ask a question about your PDF:")
                if user_input:
                    docs = store.similarity_search(user_input)
                    chain = create_chain(retriever, llm)
                    logging.info("Querying text...")
                    response = chain.invoke(input=docs)
                    st.write(response)

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    else:
        st.info("Please upload a file to get started.")

if __name__ == '__main__':
    main()