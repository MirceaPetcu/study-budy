import streamlit as st
import os
import sys
import shutil
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "retriever")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "generator")))
from retriever.index import Index
from generator.generator import Generator
from datetime import datetime


if "index" not in st.session_state:
    st.session_state.index = Index()

if "retrieve" not in st.session_state:
    st.session_state.retrieve = st.session_state.index.index

if "generator" not in st.session_state:
    st.session_state.generator = Generator(st.session_state.retrieve)

st.title("Study Buddy")

st.sidebar.header("Manage Documents")

uploaded_files = st.sidebar.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
if st.sidebar.button("Store uploaded files"):
    temp_dir = 'temp_dir'
    os.makedirs(temp_dir, exist_ok=True)
    for file in uploaded_files:
        temp_file_path = os.path.join(temp_dir, file.name)
        with open(temp_file_path, "wb") as f:
            f.write(file.getbuffer())
    st.session_state.index.insert_documents('temp_dir', extensions=['.pdf'])
    shutil.rmtree('temp_dir')

st.sidebar.subheader("Delete Documents")
if st.session_state.index.num_docs > 0:
    selected_files = st.sidebar.multiselect("Select files to delete", st.session_state.index.doc_registry.keys())
    if st.sidebar.button("Delete Selected"):
        for file_name in selected_files:
            st.session_state.index.delete_documents([file_name])
            st.sidebar.warning(f"Deleted {file_name}")
else:
    st.sidebar.info("No files uploaded yet.")

st.header("Chat with the Document-based Generator")
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

user_query = st.text_input("Ask a question about the documents", placeholder="Type your question here...")

if st.button("Send") and user_query:
    response = st.session_state.generator.generate(user_query)

    st.session_state["chat_history"].append((datetime.now().strftime("%H:%M"), "User", user_query))
    st.session_state["chat_history"].append((datetime.now().strftime("%H:%M"), "Bot", response))

st.write("---")
for timestamp, speaker, message in st.session_state["chat_history"]:
    if speaker == "User":
        st.markdown(f"<div style='color: blue; font-weight: bold;'>[{timestamp}] **{speaker}**: {message}</div>",
                    unsafe_allow_html=True)
    else:
        st.markdown(f"<div style='color: green; font-weight: bold;'>[{timestamp}] **{speaker}**: {message}</div>",
                    unsafe_allow_html=True)
st.write("---")

st.info("This chat uses documents uploaded in the sidebar to answer questions.")
