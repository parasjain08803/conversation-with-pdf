import os
import tempfile
import shutil
from typing import List, Dict, Any
import logging

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from langchain_groq import ChatGroq

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Chat with PDF", page_icon="📄", layout="centered")
st.title("📄 Chat with Your PDF")

with st.sidebar:
    st.header("📊 Document Info")
    if "doc_info" in st.session_state:
        st.write(f"**File:** {st.session_state.doc_info.get('name', 'N/A')}")
        st.write(f"**Pages:** {st.session_state.doc_info.get('pages', 'N/A')}")
        st.write(f"**Chunks:** {st.session_state.doc_info.get('chunks', 'N/A')}")
    
    st.header("⚙️ Settings")
    chunk_size = st.slider("Chunk Size", 500, 2000, 1000)
    chunk_overlap = st.slider("Chunk Overlap", 50, 500, 200)
    
    st.header("🔧 Actions")
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        if st.session_state.session_id in st.session_state.store:
            st.session_state.store[st.session_state.session_id] = InMemoryChatMessageHistory()
        st.rerun()
    
    if st.button("Reset All"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

uploaded = st.file_uploader("Upload a PDF file", type="pdf")

if "chain" not in st.session_state:
    st.session_state.chain = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    st.session_state.session_id = "user_1"

if "current_file_name" not in st.session_state:
    st.session_state.current_file_name = None

if "doc_info" not in st.session_state:
    st.session_state.doc_info = {}
if "store" not in st.session_state:
    st.session_state.store = {}


def process_pdf(uploaded_file, chunk_size: int, chunk_overlap: int) -> Dict[str, Any]:
    """Process uploaded PDF and return document info and retriever."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        chunks = splitter.split_documents(docs)
        
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        vectorstore = Chroma.from_documents(chunks, embedding=embeddings)
        retriever = vectorstore.as_retriever()
        
        os.unlink(tmp_path)
        
        return {
            "docs": docs,
            "chunks": chunks,
            "retriever": retriever,
            "pages": len(docs),
            "num_chunks": len(chunks)
        }
        
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        st.error(f"Error processing PDF: {str(e)}")
        return None


if uploaded:
    settings_changed = (
        st.session_state.chain is None
        or st.session_state.current_file_name != uploaded.name
    )
    
    if settings_changed:
        st.session_state.current_file_name = uploaded.name
        st.session_state.messages = []
        
        with st.spinner("Processing your PDF ........"):
            result = process_pdf(uploaded, chunk_size, chunk_overlap)
            
            if result is None:
                st.error("Failed to process PDF. Please try again.")
                st.stop()
            
            st.session_state.doc_info = {
                "name": uploaded.name,
                "pages": result["pages"],
                "chunks": result["num_chunks"]
            }
            
            retriever = result["retriever"]
            
            llm = ChatGroq(model_name="qwen/qwen3-32b")

            rewrite_prompt = ChatPromptTemplate.from_messages([
                (
                    "system",
                    "Rewrite the user's question as a standalone question using the chat history."
                ),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ])

            question_rewriter = (
                rewrite_prompt
                | llm
                | StrOutputParser()
            )

            def retrieve_with_history(inputs: dict):
                """inputs: {'input': str, 'chat_history': list[BaseMessage]}"""
                standalone_q = question_rewriter.invoke(inputs)
                docs = retriever.invoke(standalone_q)
                return docs

            retrieve_runnable = RunnableLambda(retrieve_with_history)

            qa_prompt = ChatPromptTemplate.from_messages([
                (
                    "system",
                    """
You are a helpful assistant.

Use ONLY the given context to answer when possible.
If the context is not enough, use your outside knowledge but clearly mark it with **(Outside Context)**.
Never say that the context does not contain the information; either answer from context, or add outside knowledge with the marker.and give only content skip complete think tag.
                    """,
                ),
                MessagesPlaceholder("chat_history"),
                (
                    "human",
                    """
Question: {input}

Context:
{context}

Answer:
                    """,
                ),
            ])

            base_chain = (
                {
                    "context": retrieve_runnable,
                    "chat_history": lambda x: x["chat_history"],
                    "input": lambda x: x["input"],
                }
                | qa_prompt
                | llm
                | StrOutputParser()
            )

            def get_history(session_id: str) -> InMemoryChatMessageHistory:
                if session_id not in st.session_state.store:
                    st.session_state.store[session_id] = InMemoryChatMessageHistory()
                return st.session_state.store[session_id]

            final_chain = RunnableWithMessageHistory(
                base_chain,
                get_history,
                input_messages_key="input",
                history_messages_key="chat_history",
            )

            st.session_state.chain = final_chain

        st.success(f"✅ Loaded: {st.session_state.current_file_name}")
    else:
        st.success(f"✅ Already loaded: {st.session_state.current_file_name}")
else:
    st.info("👆 Please upload a PDF file to start chatting.")
    

chain = st.session_state.chain

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if chain is not None:
    if user_input := st.chat_input("Ask something about your PDF..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        try:
            with st.spinner("Thinking..."):
                answer = chain.invoke(
                    {"input": user_input},
                    config={"configurable": {"session_id": st.session_state.session_id}},
                )

            st.session_state.messages.append({"role": "assistant", "content": answer})
            with st.chat_message("assistant"):
                st.markdown(answer)
                
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            st.error(f"Error generating response: {str(e)}")
            
            error_msg = f"Sorry, I encountered an error: {str(e)}"
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
            with st.chat_message("assistant"):
                st.markdown(error_msg)

if st.session_state.chain is not None:
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📥 Export Chat"):
            chat_export = "\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in st.session_state.messages])
            st.download_button(
                label="Download Chat History",
                data=chat_export,
                file_name="chat_history.txt",
                mime="text/plain"
            )
    
    with col2:
        message_count = len(st.session_state.messages)
        st.metric("Messages", message_count)
    
    with col3:
        if "doc_info" in st.session_state and st.session_state.doc_info:
            st.metric("Document Chunks", st.session_state.doc_info.get("chunks", 0))
