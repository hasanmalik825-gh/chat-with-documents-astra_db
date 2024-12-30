from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_astradb import AstraDBVectorStore
from langchain_core.documents import Document
from fastapi import UploadFile
from typing import List
import os
from constants import ASTRA_DB_API_ENDPOINT, ASTRA_DB_APPLICATION_TOKEN, ASTRA_DB_NAMESPACE

def load_document(file: UploadFile):
    # Save the uploaded file to a temporary file
    temp_file_path = f"temp_{file.filename}"
    
    with open(temp_file_path, "wb") as temp_file:
        temp_file.write(file.file.read())  # Write the file contents to disk

    file_extension = file.filename.split(".")[-1]
    if file_extension == "pdf":
        loader = PyPDFLoader(temp_file_path)
    elif file_extension == "txt":
        loader = TextLoader(temp_file_path)
    else:
        os.remove(temp_file_path)  # Cleanup
        raise ValueError(f"Unsupported file type: {file_extension}")

    try:
        # Load the document and then clean up the temporary file
        document = loader.load()
    finally:
        os.remove(temp_file_path)  # Cleanup the temporary file

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
        collection_name: str,
        embeddings: Embeddings,
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