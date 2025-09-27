import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.chat_history import BaseChatMessageHistory
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_groq import ChatGroq
import uuid


load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"]="Conversation with pdf"


st.set_page_config(page_title="Chat with your PDF", page_icon="📄", layout="wide")

st.markdown("<h2 style='text-align: center;'>📄 Chat with Your PDF</h2>", unsafe_allow_html=True)
st.divider()

with st.sidebar:
    st.header("⚙️ Upload & Settings")
    file = st.file_uploader("Upload a PDF file", type="pdf")
    st.info("💡 Ask anything about your uploaded PDF. If not in the PDF, I'll use outside knowledge (marked as **Outside Context**).")

    

if file:
    # Reset everything if a new file is uploaded
    if "last_uploaded" not in st.session_state or st.session_state.last_uploaded != file.name:
        st.session_state.vector = None
        st.session_state.messages = []
        st.session_state.store = {}
        st.session_state.last_uploaded = file.name   # track current file

    with st.spinner("Processing your PDF......."):
        if st.session_state.vector is None:
            with open("file.pdf", "wb") as f:
                f.write(file.getvalue())

            loader = PyPDFLoader("file.pdf")
            pdf = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            split = text_splitter.split_documents(pdf)

            embeddings = HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            st.session_state.vector = Chroma.from_documents(split, embedding=embeddings)

        retriever = st.session_state.vector.as_retriever()

        
        system_prompt_retriever = """
        You are a helpful assistant that reformulates user questions based on chat history 
        to improve document retrieval.
        """
        human_prompt_retriever = """
        Conversation history:
        {chat_history}

        User's question:
        {input}

        Reformulated standalone question:
        """
        retriever_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt_retriever),
            ("human", human_prompt_retriever)
        ])

        system_prompt_qa = """
        You are given a context and a user query.

        1. First, try to answer using only the provided context.
        2. If the context does not contain enough information, do **not** refuse or apologize. Instead, answer the query using outside knowledge.
        3. When you use outside knowledge, clearly mark it with: **“(Outside Context)”** before giving that part of the answer.
        4. Never say “the context does not contain this.” Always either answer from context or provide outside knowledge with the marker.

        Keep the answer clear, concise, and user-friendly.
        """
        human_prompt_qa = """
        Conversation history:
        {chat_history}

        User's question:
        {input}

        Relevant context from documents:
        {context}

        Answer:
        """
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt_qa),
            ("human", human_prompt_qa)
        ])

        llm = ChatGroq(model_name="gemma2-9b-it")
        history_aware_retriever = create_history_aware_retriever(llm, retriever, retriever_prompt)
        qa_chain = create_stuff_documents_chain(llm, qa_prompt)
        retrieval_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

    

        def get_session_history(session_id: str) -> BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id] = ChatMessageHistory()
            return st.session_state.store[session_id]

        final_chain = RunnableWithMessageHistory(
            retrieval_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )


        if "session_id" not in st.session_state:
            st.session_state.session_id = str(uuid.uuid4())  # random unique ID

        config = {"configurable": {"session_id": st.session_state.session_id}}


    
    

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if query := st.chat_input("💬 Ask a question about your PDF"):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)


        response = final_chain.invoke({"input": query}, config=config)
        answer = response["answer"]

        st.session_state.messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)
