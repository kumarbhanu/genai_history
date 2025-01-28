import streamlit as st
import os
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# Load environment variables
load_dotenv()
# groq_api_key = os.getenv('GROQ_API_KEY')
# os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')
DATA_FILE = os.path.join(os.getcwd(), "data/eds_data.txt")

groq_api_key = st.secrets["GROQ_API_KEY"]
# Initialize session state if not already present
if 'store' not in st.session_state:
    st.session_state.store = {}

# Load and preprocess text data
def load_and_preprocess_data():
    loader = TextLoader('eds_data.txt')
    loaded_data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(loaded_data)

documents = load_and_preprocess_data()

# Initialize embeddings and database
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = FAISS.from_documents(documents=documents, embedding=embeddings)
retriever = db.as_retriever()

# Function to get session history
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]

context_prompt_system = (
        "Given a chat history and the latest user question, "
        "which might reference context in the chat history, "
        "formulate a standalone question that can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed, otherwise return it as is."
    )

# Define chat history context with enhanced system prompts
history_context = ChatPromptTemplate([
    ("system", context_prompt_system),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

# Initialize LLM (Language Model)
llm = ChatGroq(groq_api_key=groq_api_key, model_name="gemma2-9b-it")

# Create history aware retriever
chat_history_retriever = create_history_aware_retriever(llm, retriever, history_context)

# Main QA retriever
system_prompt = (
        "You are an assistant that converts any code-related queries into EDS UI code, "
        "which is the company's custom UI library 'eds'. "
        "If the user asks for code examples, convert them into the relevant 'eds' code. "
        "Use the following pieces of context to guide your conversion. "
        "If you're unsure, say 'I don't know'. Keep answers concise and relevant to the query.\n\n{context}"
    )

qa_context = ChatPromptTemplate([
    ("system", system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

qa_retriever = create_stuff_documents_chain(llm, qa_context)

# Combine the retrievers
merge_retriever = create_retrieval_chain(chat_history_retriever, qa_retriever)

# Create runnable retriever with message history
rag_retriever = RunnableWithMessageHistory(
    merge_retriever,
    get_session_history=get_session_history,
    input_messages_key="input",
    output_messages_key="answer",
    history_messages_key="chat_history"
)

# Streamlit UI setup
st.title("EDS CODE TRANSLATOR")

# Input text box for user query
input_text = st.text_input("Enter your query")

if input_text:
    session_history = get_session_history(session_id="test123")
    
    # Get response and related documents
    response = rag_retriever.invoke({"input": input_text}, config={"configurable": {"session_id": "test123"}})
    
    # Display the answer
    st.subheader("Answer:")
    st.write(response["answer"])
    
    # Display related documents
    st.subheader("Related Documents:")
  
    source_docs = response.get("context", [])
    if source_docs:
        with st.expander("View Source Documents"):
            for i, doc in enumerate(source_docs):
                st.write(f"#### Document {i + 1}:")
                st.write(doc.page_content)

    # Display chat history in a formatted way
    st.subheader("Chat History:")
    for message in session_history:
        st.write(message)
    
