import streamlit as st
from services.rag import get_astra_db_vector_store, load_document, split_document, embedder_by_huggingface
from utils.document_comparison import add_unique_documents
from routes.query_document import inference_chain_rag
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

st.title("Chat with Documents using Astra DB")
st.sidebar.title("Settings")

collection_name = st.sidebar.text_input("Enter collection name (default: astra_vectordb_langchain)", "").strip() or "astra_vectordb_langchain"
api_endpoint = st.sidebar.text_input("Astra DB API Endpoint", type="password")
token = st.sidebar.text_input("Astra DB Application Token", type="password")
namespace = st.sidebar.text_input("Astra DB Namespace")
model = st.sidebar.selectbox("Select model", ["llama3-8b-8192", "gemma2-9b-it", "mixtral-8x7b-32768"], index=0)
groq_api_key = st.sidebar.text_input("Groq API Key", type="password")
embeddings = embedder_by_huggingface()

template = [
        ("system", "You are a helpful assistant that answers concisely. You are given the following context: {context}."),
        ("human", "{input}"),
    ]
prompt_template = ChatPromptTemplate.from_messages(messages=template)
llm = ChatGroq(model=model, streaming=True, api_key=groq_api_key)

file = st.file_uploader("Upload file", type=["pdf", "txt"])
query = st.text_input("Enter query")

if st.button("Chat"):
    if file and query and api_endpoint and token and namespace and model and groq_api_key:
        documents = load_document(file)
        documents = split_document(documents)
        vector_store = get_astra_db_vector_store(
            collection_name=collection_name,
            api_endpoint=api_endpoint,
            token=token,
            namespace=namespace,
            embeddings=embeddings
        )
        add_unique_documents(documents, vector_store)
        st.success("Documents added to vector store")
        st.info("Now thinking on query...")
        chain = inference_chain_rag(
        vectorstorage=vector_store, 
        llm=llm,
        prompt_template=prompt_template,
    )
        response = chain.invoke({"input": query})
        st.success("Got the answer, Yea!")
        st.write(response["answer"])
    else:
        st.error("Please fill all the fields")


