from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_astradb import AstraDBVectorStore
from langchain_core.documents import Document
from fastapi import UploadFile as FastAPIUploadFile
from starlette.datastructures import UploadFile as FastAPIUploadFile
from streamlit.runtime.uploaded_file_manager import UploadedFile as StreamlitUploadedFile
import tempfile
from typing import List, Union
import os
from constants import ASTRA_DB_API_ENDPOINT, ASTRA_DB_APPLICATION_TOKEN, ASTRA_DB_NAMESPACE

def load_document(file: Union[FastAPIUploadFile, StreamlitUploadedFile]):
    """
    Handles file loading for both FastAPI's UploadFile and Streamlit's UploadedFile.

    Args:
        file: FastAPI's UploadFile or Streamlit's UploadedFile.

    Returns:
        Loaded document.
    """
    # Check file type
    if isinstance(file, FastAPIUploadFile):
        # FastAPI UploadFile handling
        file_name = file.filename
        file_content = file.file.read()
    elif isinstance(file, StreamlitUploadedFile):
        # Streamlit UploadedFile handling
        file_name = file.name
        file_content = file.read()
    else:
        raise ValueError("Unsupported file type: Must be FastAPI's UploadFile or Streamlit's UploadedFile.")

    # Create a temporary file for processing
    file_extension = file_name.split(".")[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as temp_file:
        temp_file.write(file_content)
        temp_file_path = temp_file.name

    # Determine the loader based on file type
    if file_extension == "pdf":
        loader = PyPDFLoader(temp_file_path)
    elif file_extension == "txt":
        loader = TextLoader(temp_file_path)
    else:
        os.remove(temp_file_path)  # Cleanup
        raise ValueError(f"Unsupported file type: {file_extension}")

    try:
        # Load the document
        document = loader.load()
    finally:
        # Ensure temporary file is cleaned up
        os.remove(temp_file_path)

    return document

def split_document(documents: List[Document]):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(documents)

def embedder_by_huggingface(model : str = "all-MiniLM-L12-v2") -> HuggingFaceEmbeddings:
    """
    This function is used to embed the documents using huggingface embedding model.
    """
    embedder = HuggingFaceEmbeddings(model_name=model)
    return embedder

def get_astra_db_vector_store(
        embeddings: Embeddings,
        collection_name: str = "astra_vectordb_langchain",
        api_endpoint: str = ASTRA_DB_API_ENDPOINT,
        token: str = ASTRA_DB_APPLICATION_TOKEN,
        namespace: str = ASTRA_DB_NAMESPACE,
    ):
    vector_store = AstraDBVectorStore(
        collection_name=collection_name,
        embedding=embeddings,
        api_endpoint=api_endpoint,
        token=token,
        namespace=namespace,
    )
    return vector_store