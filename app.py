
    
import streamlit as st
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.vectorstores import FAISS
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain.embeddings import OllamaEmbeddings
# from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.runnables.history import RunnableWithMessageHistory
import os
from dotenv import load_dotenv
from langchain.output_parsers import StructuredOutputParser


# Load environment variables
load_dotenv()
os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')
groq_api_key = os.getenv('GROQ_API_KEY')

# Set up embeddings
# embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
embeddings=OllamaEmbeddings(model="gemma:2b")

# Set up Streamlit app
st.title('Code Conversion App')
st.write("Upload a PDF and chat")

# Set up LLM
llm = ChatGroq(groq_api_key=groq_api_key, model_name='gemma2-9b-it')
session_id = st.text_input("Session ID", value="default_session")

# Statefully manage chat history
if "store" not in st.session_state:
    st.session_state.store = {}

upload_files = st.file_uploader("Upload PDF", type="pdf", accept_multiple_files=False)

# Process file
if upload_files:
    documents = []
    temppdf = f"./temp.pdf"

    # Save uploaded file
    with open(temppdf, "wb") as file:
        file.write(upload_files.getvalue())
        file_name = upload_files.name

    # Load and split documents
    loader = PyMuPDFLoader(temppdf)
    docs = loader.load()
    documents.extend(docs)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    splits = text_splitter.split_documents(documents=documents)

    # Create vector store and retriever
    vector_store = FAISS.from_documents(documents=splits, embedding=embeddings)
    retriever = vector_store.as_retriever()

    # Set up context prompt
    context_prompt_system = (
        "Given a chat history and the latest user question, "
        "which might reference context in the chat history, "
        "formulate a standalone question that can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed, otherwise return it as is."
    )
    context_prompt = ChatPromptTemplate.from_messages([
        ("system", context_prompt_system),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    # Create history-aware retriever
    history_aware_retriever = create_history_aware_retriever(llm, retriever, context_prompt)

    # Set up QA prompt
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise.\n\n{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    # Create QA and retrieval chains
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # Function to manage session history
    def get_session_history(session: str) -> BaseChatMessageHistory:
        if session not in st.session_state.store:
            st.session_state.store[session] = ChatMessageHistory()
        return st.session_state.store[session]

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
        get_session_history=get_session_history
    )

    # Get user input
    user_input = st.text_input("Your question:")
    if user_input:
        session_history = get_session_history(session_id)
        response = conversational_rag_chain.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}}
        )
        st.write(st.session_state.store)
     

        st.write("### Assistance:")
        st.write(response["answer"])
        st.write("### Chat History:")
        st.write("### Chat History:")
        # Format and display chat history

        for message in session_history.messages:
            st.write(message)
          
            
              

     
